"""Sequence-only three-label baselines for the PU split experiment."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from src.multimodal.torch_losses import estimate_class_priors, make_nnpu_loss


class OneHotCNNBaseline(nn.Module):
    """Local 31-residue one-hot baseline with no auxiliary inputs."""

    def __init__(self, alphabet_size):
        super().__init__()
        self.alphabet_size = int(alphabet_size)
        self.encoder = nn.Sequential(
            nn.Conv1d(self.alphabet_size, 128, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.25),
        )
        self.head = nn.Linear(128, 3)

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[-1] != self.alphabet_size:
            raise ValueError("one-hot sequence must have shape (rows, window, alphabet)")
        return torch.sigmoid(self.head(self.encoder(sequence.transpose(1, 2))))


class FuncPhosStyleCNNBaseline(nn.Module):
    """Three-task nnPU adaptation of the original local sequence-CNN idea."""

    def __init__(self, alphabet_size):
        super().__init__()
        self.alphabet_size = int(alphabet_size)
        self.encoder = nn.Sequential(
            nn.Conv1d(self.alphabet_size, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.head = nn.Linear(64, 3)

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[-1] != self.alphabet_size:
            raise ValueError("one-hot sequence must have shape (rows, window, alphabet)")
        return torch.sigmoid(self.head(self.encoder(sequence.transpose(1, 2))))


class ESMMLPBaseline(nn.Module):
    """Frozen ESM-2 cache baseline; motif columns are deliberately excluded."""

    def __init__(self, embedding_dim=640):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.network = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Linear(self.embedding_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 3),
        )

    def forward(self, embedding):
        if embedding.ndim != 2 or embedding.shape[1] != self.embedding_dim:
            raise ValueError("ESM input must have shape (rows, 640)")
        return torch.sigmoid(self.network(embedding))


def _macro_auprc(labels, probabilities):
    return float(
        sum(average_precision_score(labels[:, index], probabilities[:, index]) for index in range(3))
        / 3.0
    )


def train_sequence_baseline(
    model_name, sequence, sequence_aux, labels, split_names, site_ids, output_directory,
    seed, epochs=200, batch_size=512, patience=30, device="cuda",
):
    """Train one sequence-only nnPU baseline using an immutable explicit split vector."""
    if model_name not in {"onehot_cnn", "esm_mlp", "funcphos_seq_style_cnn"}:
        raise ValueError("unsupported sequence baseline model")
    labels = torch.as_tensor(labels, dtype=torch.float32)
    split_names = list(map(str, split_names))
    if labels.ndim != 2 or labels.shape[1] != 3 or not (
        len(labels) == len(split_names) == len(site_ids)
    ):
        raise ValueError("sequence baseline arrays do not align")
    rows = {name: [index for index, value in enumerate(split_names) if value == name] for name in ("train", "validation", "test")}
    if not all(rows.values()):
        raise ValueError("all three splits must contain rows")
    torch.manual_seed(int(seed))
    resolved = torch.device(device)
    cached_values = sequence if model_name != "esm_mlp" else sequence_aux[:, :640]
    # Processed arrays are read-only memory maps; training needs an owned buffer.
    values = torch.as_tensor(np.array(cached_values, dtype=np.float32, copy=True))
    if model_name == "onehot_cnn":
        model = OneHotCNNBaseline(values.shape[2])
    elif model_name == "funcphos_seq_style_cnn":
        model = FuncPhosStyleCNNBaseline(values.shape[2])
    else:
        model = ESMMLPBaseline()
    model = model.to(resolved)
    priors = estimate_class_priors(labels[rows["train"]], "train")
    loss_function = make_nnpu_loss(priors)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(TensorDataset(values[rows["train"]], labels[rows["train"]]), batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(TensorDataset(values[rows["validation"]]), batch_size=batch_size)
    best_state, best_score, remaining = None, float("-inf"), int(patience)
    epochs_trained = 0
    for epoch in range(int(epochs)):
        epochs_trained = epoch + 1
        model.train()
        for x, y in train_loader:
            loss = loss_function(y.to(resolved), model(x.to(resolved)))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = torch.cat([model(x.to(resolved)).cpu() for (x,) in validation_loader]).numpy()
        score = _macro_auprc(labels[rows["validation"]].numpy(), validation)
        if score > best_score:
            best_score, best_state, remaining = score, {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, int(patience)
        else:
            remaining -= 1
            if remaining == 0:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation = model(values[rows["validation"]].to(resolved)).cpu().numpy()
        test = model(values[rows["test"]].to(resolved)).cpu().numpy()
    return model, test, {
        "best_validation_macro_auprc": best_score,
        "class_priors": priors,
        "validation_rows": rows["validation"],
        "validation_probabilities": validation,
        "test_rows": rows["test"],
        "epochs_trained": epochs_trained,
    }
