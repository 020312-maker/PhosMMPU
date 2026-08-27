"""Validation contracts shared by the upgraded multimodal pipeline."""

import numpy as np


SITE_ARRAYS = (
    "sequence",
    "sequence_aux",
    "structure_graph",
    "structure_vectors",
    "structure_neighbors",
    "proteomics",
    "modality_mask",
)


def _require_array(arrays, name):
    if name not in arrays:
        raise ValueError(f"missing required model array: {name}")
    return np.asarray(arrays[name])


def validate_arrays(arrays, site_rows):
    """Validate site-aligned inputs before model construction or training."""
    if int(site_rows) <= 0:
        raise ValueError("site_rows must be positive")
    for name in SITE_ARRAYS:
        values = _require_array(arrays, name)
        if values.ndim < 1 or values.shape[0] != site_rows:
            raise ValueError(f"{name} does not align with site rows")

    sequence = np.asarray(arrays["sequence"])
    if sequence.ndim != 3 or sequence.shape[1:] != (31, 21):
        raise ValueError("sequence must have shape (site_rows, 31, 21)")
    mask = np.asarray(arrays["modality_mask"])
    if mask.shape != (site_rows, 4) or not set(np.unique(mask)).issubset({0.0, 1.0}):
        raise ValueError("modality_mask must be binary with four columns")
    if not np.all(mask[:, 0] == 1.0):
        raise ValueError("sequence must be available for every site")

    edges = _require_array(arrays, "network_edges")
    weights = _require_array(arrays, "network_weights")
    relations = _require_array(arrays, "network_relations")
    nodes = _require_array(arrays, "network_nodes")
    if nodes.ndim != 2 or nodes.shape[0] == 0:
        raise ValueError("network_nodes must be a non-empty two-dimensional array")
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("network_edges must have shape (2, edge_count)")
    if weights.ndim != 1 or weights.shape[0] != edges.shape[1]:
        raise ValueError("network weights must align with network edges")
    if relations.ndim != 1 or relations.shape[0] != edges.shape[1]:
        raise ValueError("network relation ids must align with network edges")
    if edges.size and (edges.min() < 0 or edges.max() >= len(nodes)):
        raise ValueError("network edge index is outside network_nodes")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("network weights must be finite and non-negative")
    if (relations < 0).any():
        raise ValueError("network relation ids must be non-negative")
    return True
