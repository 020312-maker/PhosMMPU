import torch
from torch import nn


class SequenceEncoder(nn.Module):
    def __init__(self, alphabet_size, embedding_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(alphabet_size, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, value):
        return self.layers(value.transpose(1, 2))


class TabularEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, value):
        return self.layers(value)


class MaskedGatedFusion(nn.Module):
    def __init__(self, embedding_dim, modality_dropout=0.0):
        super().__init__()
        self.gate = nn.Linear(embedding_dim, 1)
        self.modality_dropout = float(modality_dropout)

    def forward(self, embeddings, mask):
        mask = mask.float()
        if self.training and self.modality_dropout > 0:
            keep = (torch.rand_like(mask[:, 1:]) >= self.modality_dropout).float()
            mask = torch.cat((mask[:, :1], mask[:, 1:] * keep), dim=1)
        logits = self.gate(embeddings).squeeze(-1)
        logits = logits.masked_fill(mask == 0, -torch.inf)
        weights = torch.softmax(logits, dim=1) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return (embeddings * weights.unsqueeze(-1)).sum(dim=1), weights


class MultimodalTorchModel(nn.Module):
    def __init__(
        self,
        window,
        alphabet_size,
        structure_dim,
        network_dim,
        proteomics_dim,
        embedding_dim=64,
        modality_dropout=0.2,
        fusion_mode="gated",
    ):
        super().__init__()
        if fusion_mode not in {"gated", "concatenate"}:
            raise ValueError(f"unsupported fusion_mode: {fusion_mode}")
        self.window = int(window)
        self.alphabet_size = int(alphabet_size)
        self.structure_dim = int(structure_dim)
        self.network_dim = int(network_dim)
        self.proteomics_dim = int(proteomics_dim)
        self.embedding_dim = int(embedding_dim)
        self.fusion_mode = fusion_mode
        self.sequence_encoder = SequenceEncoder(alphabet_size, embedding_dim)
        self.structure_encoder = TabularEncoder(structure_dim, embedding_dim)
        self.network_encoder = TabularEncoder(network_dim, embedding_dim)
        self.proteomics_encoder = TabularEncoder(proteomics_dim, embedding_dim)
        self.fusion = MaskedGatedFusion(embedding_dim, modality_dropout)
        self.concat_projection = nn.Linear(embedding_dim * 4, embedding_dim)
        self.shared = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Linear(64, 3)

    def forward(
        self,
        sequence,
        structure,
        network,
        proteomics,
        modality_mask,
        return_gates=False,
    ):
        embeddings = torch.stack(
            (
                self.sequence_encoder(sequence),
                self.structure_encoder(structure),
                self.network_encoder(network),
                self.proteomics_encoder(proteomics),
            ),
            dim=1,
        )
        if self.fusion_mode == "gated":
            fused, gates = self.fusion(embeddings, modality_mask)
        else:
            masked = embeddings * modality_mask.float().unsqueeze(-1)
            fused = torch.relu(self.concat_projection(masked.flatten(start_dim=1)))
            gates = modality_mask.float()
            gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-12)
        scores = torch.sigmoid(self.head(self.shared(fused)))
        return (scores, gates) if return_gates else scores

    def dimensions(self):
        return {
            "window": self.window,
            "alphabet_size": self.alphabet_size,
            "structure_dim": self.structure_dim,
            "network_dim": self.network_dim,
            "proteomics_dim": self.proteomics_dim,
            "embedding_dim": self.embedding_dim,
            "fusion_mode": self.fusion_mode,
        }
