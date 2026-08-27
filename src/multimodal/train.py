"""Training entry point for the enhanced human-only multimodal model."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.multimodal.config import MultimodalConfig
from src.multimodal.torch_losses import (
    estimate_class_priors,
    make_bce_loss,
    make_nnpu_loss,
    make_weighted_bce,
)
from src.multimodal.phosmmpu_model import PhosMMPUModel


INPUT_NAMES = (
    "sequence", "sequence_aux", "structure_scalars", "structure_vectors", "structure_mask",
    "network_node", "network_neighbors", "network_weights", "network_relations",
    "network_neighbor_mask", "network_global_context", "proteomics", "modality_mask",
)
SPLIT_NAMES = ("train", "validation", "test")
MODALITY_NAMES = ("sequence", "structure", "network", "proteomics")


def disable_modalities(arrays, disabled_modalities=()):
    """Return training arrays with requested fusion branches unavailable."""
    requested = tuple(dict.fromkeys(str(name) for name in disabled_modalities))
    unknown = set(requested).difference(MODALITY_NAMES)
    if unknown:
        raise ValueError(f"unsupported modalities: {', '.join(sorted(unknown))}")
    result = dict(arrays)
    result["modality_mask"] = arrays["modality_mask"].copy()
    for modality in requested:
        result["modality_mask"][:, MODALITY_NAMES.index(modality)] = 0.0
    if not np.all(result["modality_mask"].sum(axis=1) > 0):
        raise ValueError("at least one modality must remain available for every site")
    return result


def _device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _validate_inputs(inputs, labels=None):
    missing = set(INPUT_NAMES).difference(inputs)
    if missing:
        raise ValueError(f"missing model inputs: {sorted(missing)}")
    values = {name: np.asarray(inputs[name]) for name in INPUT_NAMES}
    rows = len(values["sequence"])
    if rows == 0 or values["sequence"].ndim != 3:
        raise ValueError("sequence must be a non-empty three-dimensional array")
    expected = {
        "sequence_aux": 2, "structure_scalars": 3, "structure_vectors": 3,
        "structure_mask": 2, "network_node": 2, "network_neighbors": 3,
        "network_weights": 2, "network_relations": 2, "network_neighbor_mask": 2,
        "network_global_context": 2, "proteomics": 2, "modality_mask": 2,
    }
    for name, dimensions in expected.items():
        if values[name].ndim != dimensions or len(values[name]) != rows:
            raise ValueError(f"{name} does not align with site rows")
    if values["structure_vectors"].shape[:2] != values["structure_scalars"].shape[:2]:
        raise ValueError("structure vectors and scalars must share graph dimensions")
    if values["structure_mask"].shape != values["structure_scalars"].shape[:2]:
        raise ValueError("structure mask must align with graph dimensions")
    if values["network_neighbors"].shape[:2] != values["network_weights"].shape:
        raise ValueError("network weights must align with neighbours")
    if values["network_neighbors"].shape[:2] != values["network_relations"].shape:
        raise ValueError("network relation ids must align with neighbours")
    if values["network_neighbors"].shape[:2] != values["network_neighbor_mask"].shape:
        raise ValueError("network neighbour mask must align with neighbours")
    if values["modality_mask"].shape != (rows, 4):
        raise ValueError("modality mask must have four columns")
    if not set(np.unique(values["modality_mask"])).issubset({0.0, 1.0}):
        raise ValueError("modality mask must be binary")
    if not np.isfinite(np.concatenate([
        values[name].astype(np.float32, copy=False).reshape(rows, -1)
        for name in INPUT_NAMES if name != "network_relations"
    ], axis=1)).all():
        raise ValueError("model inputs contain non-finite values")
    if (values["network_relations"] < 0).any() or (values["network_relations"] > 3).any():
        raise ValueError("network relation ids must be in [0, 3]")
    prepared = {
        name: np.asarray(value, dtype=np.int64 if name == "network_relations" else np.float32)
        for name, value in values.items()
    }
    if labels is None:
        return prepared
    labels = np.asarray(labels, dtype=np.float32)
    if labels.shape != (rows, 3) or not set(np.unique(labels)).issubset({0.0, 1.0}):
        raise ValueError("labels must be binary with shape (site_rows, 3)")
    return prepared, labels


def _tensors(inputs, labels=None):
    tensors = [torch.from_numpy(np.ascontiguousarray(inputs[name])) for name in INPUT_NAMES]
    if labels is not None:
        tensors.append(torch.from_numpy(np.ascontiguousarray(labels)))
    return tensors


def _macro_auprc(labels, probabilities):
    scores = [
        average_precision_score(labels[:, column], probabilities[:, column])
        for column in range(3)
        if labels[:, column].min() != labels[:, column].max()
    ]
    return float(np.mean(scores)) if scores else float("nan")


def _positive_weights(labels):
    positives = labels.sum(axis=0)
    return [float(max((len(labels) - value) / max(value, 1.0), 1.0)) for value in positives]


def _model_from_inputs(inputs, embedding_dim, modality_dropout, transformer_layers=2):
    return PhosMMPUModel(
        alphabet_size=inputs["sequence"].shape[2],
        sequence_aux_dim=inputs["sequence_aux"].shape[1],
        structure_scalar_dim=inputs["structure_scalars"].shape[2],
        network_node_dim=inputs["network_node"].shape[1],
        network_global_dim=inputs["network_global_context"].shape[1],
        proteomics_dim=inputs["proteomics"].shape[1],
        embedding_dim=embedding_dim,
        transformer_layers=transformer_layers,
        modality_dropout=modality_dropout,
    )


def _predict_loader(model, loader, device):
    output = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensors = [value.to(device, non_blocking=True) for value in batch[:len(INPUT_NAMES)]]
            output.append(model(*tensors).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def _fit(
    model, train_loader, train_labels, validation_loader, validation_labels, output_dir, epochs, seed,
    class_priors, objective, device, patience,
):
    if objective not in {"nnpu", "weighted_bce", "bce"}:
        raise ValueError(f"unsupported objective: {objective}")
    if objective == "nnpu":
        loss_function = make_nnpu_loss(class_priors)
    elif objective == "weighted_bce":
        loss_function = make_weighted_bce(_positive_weights(train_labels))
    else:
        loss_function = make_bce_loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    if int(patience) < 1:
        raise ValueError("early-stopping patience must be positive")
    history, best_state, best_score, remaining_patience = [], None, -np.inf, int(patience)
    for epoch in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for batch in train_loader:
            tensors = [value.to(device, non_blocking=True) for value in batch]
            prediction = model(*tensors[:len(INPUT_NAMES)])
            loss = loss_function(tensors[-1], prediction)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_probabilities = _predict_loader(model, validation_loader, device)
        score = _macro_auprc(validation_labels, validation_probabilities)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_auprc": score})
        if score > best_score:
            best_score, best_state, remaining_patience = score, copy.deepcopy(model.state_dict()), int(patience)
        else:
            remaining_patience -= 1
            if remaining_patience == 0:
                break
    model.load_state_dict(best_state)
    probabilities = _predict_loader(model, validation_loader, device)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    np.save(output_dir / "validation_probabilities.npy", probabilities)
    return probabilities, history, best_score


def train_arrays(
    train_inputs, train_labels, validation_inputs, validation_labels, output_dir, epochs,
    seed, class_priors, objective="nnpu", batch_size=512, embedding_dim=64,
    modality_dropout=0.0, transformer_layers=2, device="auto", patience=30,
):
    train_inputs, train_labels = _validate_inputs(train_inputs, train_labels)
    validation_inputs, validation_labels = _validate_inputs(validation_inputs, validation_labels)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    resolved = _device(device)
    model = _model_from_inputs(
        train_inputs, embedding_dim, modality_dropout, transformer_layers
    ).to(resolved)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(*_tensors(train_inputs, train_labels)), batch_size=batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(TensorDataset(*_tensors(validation_inputs)), batch_size=batch_size, shuffle=False)
    probabilities, history, best_score = _fit(
        model, train_loader, train_labels, validation_loader, validation_labels, output_dir, epochs, seed,
        class_priors, objective, resolved, patience,
    )
    manifest = {
        "backend": "pytorch", "seed": int(seed), "device": str(resolved),
        "objective": objective, "class_priors": [float(value) for value in class_priors],
        "epochs_requested": int(epochs), "epochs_completed": len(history),
        "early_stopping_patience": int(patience),
        "best_validation_macro_auprc": float(best_score), "model_dimensions": model.dimensions(),
    }
    torch.save({"state_dict": model.state_dict(), "model_dimensions": model.dimensions(), "run_manifest": manifest}, output_dir / "model.pt")
    (output_dir / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"model": model, "validation_probabilities": probabilities, "manifest": manifest}


class _GraphSiteDataset(Dataset):
    """Lazily gathers fixed neighbours, avoiding a site-times-neighbour copy."""

    def __init__(self, arrays, rows, labels=None):
        self.arrays = arrays
        self.rows = np.asarray(rows, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = int(self.rows[item])
        protein = int(self.arrays["protein_index"][row])
        neighbour_index = self.arrays["neighbor_index"][protein]
        safe_index = np.maximum(neighbour_index, 0)
        values = (
            self.arrays["sequence"][row], self.arrays["sequence_aux"][row],
            self.arrays["structure_scalars"][row], self.arrays["structure_vectors"][row],
            self.arrays["structure_mask"][row], self.arrays["node_features"][protein],
            self.arrays["node_features"][safe_index], self.arrays["neighbor_weights"][protein],
            self.arrays["neighbor_relations"][protein], self.arrays["neighbor_mask"][protein],
            self.arrays["global_context"][protein], self.arrays["proteomics"][row],
            self.arrays["modality_mask"][row],
        )
        tensors = tuple(torch.as_tensor(np.array(value, copy=True)) for value in values)
        if self.labels is not None:
            return (*tensors, torch.as_tensor(self.labels[row]))
        return tensors


def _load_processed(config, splits, network_dir=None):
    root = config.processed_data
    network_root = Path(network_dir or root)
    required = (
        "sequence_aux.npy", "structure_graph_scalars.npy", "structure_graph_vectors.npy",
        "structure_graph_mask.npy", "relation_graph.npz", "relation_node_features.npy",
        "relation_global_context.npy",
    )
    missing = [name for name in required if not ((network_root if name.startswith("relation_") else root) / name).is_file()]
    if missing:
        raise FileNotFoundError(f"preprocessing is incomplete: {', '.join(missing)}")
    site_index = pd.read_parquet(root / "site_index.parquet")
    site_ids = np.load(root / "dataset_site_ids.npy", allow_pickle=False)
    if not np.array_equal(site_ids, site_index["site_id"].astype(str).to_numpy()):
        raise ValueError("site index no longer aligns with dataset site ids")
    graph = np.load(network_root / "relation_graph.npz", allow_pickle=False)
    node_accessions = graph["node_accessions"].astype(str)
    lookup = {accession: index for index, accession in enumerate(node_accessions)}
    protein_index = site_index["accession"].astype(str).map(lookup).to_numpy()
    if pd.isna(protein_index).any():
        raise ValueError("some sites have no relation-graph protein node")
    protein_index = protein_index.astype(np.int64)
    node_features = np.load(network_root / "relation_node_features.npy").astype(np.float32)
    global_context = np.load(network_root / "relation_global_context.npy").astype(np.float32)
    train_nodes = np.unique(protein_index[splits == "train"])
    for values in (node_features, global_context):
        mean = values[train_nodes].mean(axis=0, keepdims=True)
        scale = values[train_nodes].std(axis=0, keepdims=True)
        values -= mean
        values /= np.maximum(scale, 1e-6)
    modality_mask = np.load(root / "modality_masks.npy").astype(np.float32)
    structure_mask = np.load(root / "structure_graph_mask.npy", mmap_mode="r")
    modality_mask[:, 1] *= structure_mask.any(axis=1).astype(np.float32)
    return {
        "sequence": np.load(root / "sequence.npy", mmap_mode="r"),
        "sequence_aux": np.load(root / "sequence_aux.npy", mmap_mode="r"),
        "structure_scalars": np.load(root / "structure_graph_scalars.npy", mmap_mode="r"),
        "structure_vectors": np.load(root / "structure_graph_vectors.npy", mmap_mode="r"),
        "structure_mask": structure_mask,
        "protein_index": protein_index,
        "node_features": node_features,
        "neighbor_index": graph["sampled_neighbor_index"],
        "neighbor_weights": graph["sampled_neighbor_weight"],
        "neighbor_relations": graph["sampled_neighbor_relation"],
        "neighbor_mask": graph["sampled_neighbor_mask"],
        "global_context": global_context,
        "proteomics": np.load(root / "proteomics_processed.npy", mmap_mode="r"),
        "modality_mask": modality_mask,
    }, site_ids


def _site_id_hash(values):
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_from_config(config, seeds, epochs=200, batch_size=512, objective="nnpu", device="auto", output_root=None, patience=30, transformer_layers=2, modality_dropout=None, network_dir=None, network_variant="complete", network_enabled=True, disabled_modalities=()):
    root = config.processed_data
    labels = np.load(root / "labels.npy").astype(np.float32)
    splits = np.load(root / "split_names.npy", allow_pickle=False)
    if len(labels) != len(splits) or set(np.unique(splits)).difference(SPLIT_NAMES):
        raise ValueError("labels or split names are invalid")
    network_root = Path(network_dir or root)
    arrays, site_ids = _load_processed(config, splits, network_root)
    disabled_modalities = tuple(disabled_modalities)
    if not network_enabled:
        disabled_modalities = (*disabled_modalities, "network")
    arrays = disable_modalities(arrays, disabled_modalities)
    groups = {name: np.flatnonzero(splits == name) for name in SPLIT_NAMES}
    if not all(len(rows) for rows in groups.values()):
        raise ValueError("all three group-aware splits must be non-empty")
    priors = estimate_class_priors(labels[groups["train"]], "train", config.pu_prior_multiplier)
    resolved = _device(device)
    modality_dropout = config.modality_dropout if modality_dropout is None else float(modality_dropout)
    output_root = Path(output_root or config.artifacts / "training_runs")
    output_root.mkdir(parents=True, exist_ok=True)
    results = []
    dimensions = {
        "sequence": np.empty((1, 31, arrays["sequence"].shape[2]), dtype=np.float32),
        "sequence_aux": np.empty((1, arrays["sequence_aux"].shape[1]), dtype=np.float32),
        "structure_scalars": np.empty((1, 1, arrays["structure_scalars"].shape[2]), dtype=np.float32),
        "network_node": np.empty((1, arrays["node_features"].shape[1]), dtype=np.float32),
        "network_global_context": np.empty((1, arrays["global_context"].shape[1]), dtype=np.float32),
        "proteomics": np.empty((1, arrays["proteomics"].shape[1]), dtype=np.float32),
    }
    for seed in seeds:
        run_dir = output_root / f"{objective}_seed{int(seed)}"
        run_dir.mkdir(parents=True, exist_ok=False)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = PhosMMPUModel(
            alphabet_size=dimensions["sequence"].shape[2], sequence_aux_dim=dimensions["sequence_aux"].shape[1],
            structure_scalar_dim=dimensions["structure_scalars"].shape[2], network_node_dim=dimensions["network_node"].shape[1],
            network_global_dim=dimensions["network_global_context"].shape[1], proteomics_dim=dimensions["proteomics"].shape[1],
            embedding_dim=config.embedding_dim, transformer_layers=transformer_layers, modality_dropout=modality_dropout,
        ).to(resolved)
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(_GraphSiteDataset(arrays, groups["train"], labels), batch_size=batch_size, shuffle=True, generator=generator, pin_memory=resolved.type == "cuda")
        validation_loader = DataLoader(_GraphSiteDataset(arrays, groups["validation"]), batch_size=batch_size, shuffle=False, pin_memory=resolved.type == "cuda")
        probabilities, history, best_score = _fit(model, train_loader, labels[groups["train"]], validation_loader, labels[groups["validation"]], run_dir, epochs, seed, priors, objective, resolved, patience)
        test_loader = DataLoader(_GraphSiteDataset(arrays, groups["test"]), batch_size=batch_size, shuffle=False, pin_memory=resolved.type == "cuda")
        test_probabilities = _predict_loader(model, test_loader, resolved)
        np.save(run_dir / "test_probabilities.npy", test_probabilities)
        pd.DataFrame({
            "site_id": site_ids[groups["test"]],
            "y_activity": labels[groups["test"], 0].astype(np.int8),
            "y_interaction": labels[groups["test"], 1].astype(np.int8),
            "y_proteostasis": labels[groups["test"], 2].astype(np.int8),
            "activity_probability": test_probabilities[:, 0],
            "interaction_probability": test_probabilities[:, 1],
            "proteostasis_probability": test_probabilities[:, 2],
        }).to_parquet(run_dir / "test_predictions.parquet", index=False)
        manifest = {
            "backend": "pytorch", "seed": int(seed), "device": str(resolved), "objective": objective,
            "class_priors": [float(value) for value in priors], "epochs_requested": int(epochs),
            "pu_prior_multiplier": float(config.pu_prior_multiplier),
            "epochs_completed": len(history), "best_validation_macro_auprc": float(best_score),
            "early_stopping_patience": int(patience),
            "model_dimensions": model.dimensions(), "split_counts": {name: int(len(rows)) for name, rows in groups.items()},
            "site_id_hashes": {name: _site_id_hash(site_ids[rows]) for name, rows in groups.items()},
            "network_scaling": "mean and standard deviation fitted on unique train-split proteins only",
            "network_variant": str(network_variant),
            "network_enabled": bool(network_enabled),
            "disabled_modalities": list(dict.fromkeys(disabled_modalities)),
            "network_sidecar_sha256": _file_hash(network_root / "network_variant_audit.json") if (network_root / "network_variant_audit.json").is_file() else None,
            "species_scope": "human_only",
        }
        torch.save({"state_dict": model.state_dict(), "model_dimensions": model.dimensions(), "run_manifest": manifest}, run_dir / "model.pt")
        (run_dir / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        results.append({"run_dir": str(run_dir), **manifest})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the enhanced human phosphosite multimodal model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--objective", choices=("nnpu", "weighted_bce", "bce"), default="nnpu")
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--modality-dropout", type=float)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-root")
    parser.add_argument("--network-dir")
    parser.add_argument("--network-variant", default="complete")
    parser.add_argument("--disable-network", action="store_true")
    parser.add_argument("--disable-modalities", nargs="*", choices=MODALITY_NAMES, default=())
    args = parser.parse_args(argv)
    config = MultimodalConfig.load(args.config)
    config.ensure_output_directories()
    print(json.dumps(train_from_config(config, args.seeds, args.epochs, args.batch_size, args.objective, args.device, args.output_root, args.patience, args.transformer_layers, args.modality_dropout, args.network_dir, args.network_variant, not args.disable_network, args.disable_modalities), indent=2))


if __name__ == "__main__":
    main()
