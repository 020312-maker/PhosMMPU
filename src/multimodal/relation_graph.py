"""Typed, deterministic protein interaction graphs for PhosMMPU.

STRING and BioGRID are treated as undirected physical/functional evidence.
SIGNOR retains its signed causal direction, so activation and inhibition remain
separate relation types throughout sampling and model training.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


RELATION_NAMES = (
    "string",
    "biogrid",
    "signor_activation",
    "signor_inhibition",
)
_UNDIRECTED_RELATIONS = {"string", "biogrid"}
_REQUIRED_COLUMNS = ("source", "target", "weight")


def _validated_edges(frame: pd.DataFrame, index: Mapping[str, int]) -> pd.DataFrame:
    missing = set(_REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Relation edges are missing columns: {sorted(missing)}")
    selected = frame.loc[:, _REQUIRED_COLUMNS].copy()
    selected["source"] = selected["source"].astype(str)
    selected["target"] = selected["target"].astype(str)
    selected["weight"] = pd.to_numeric(selected["weight"], errors="coerce")
    selected = selected[
        selected["source"].isin(index)
        & selected["target"].isin(index)
        & selected["source"].ne(selected["target"])
        & selected["weight"].notna()
        & selected["weight"].gt(0)
    ].copy()
    selected["source_index"] = selected["source"].map(index).astype(np.int64)
    selected["target_index"] = selected["target"].map(index).astype(np.int64)
    return selected[["source_index", "target_index", "weight"]]


def build_relation_graph(
    relation_frames: Mapping[str, pd.DataFrame], accessions: list[str] | np.ndarray
) -> dict[str, np.ndarray]:
    """Build a typed edge list in the exact protein order supplied by callers."""
    unknown = set(relation_frames).difference(RELATION_NAMES)
    if unknown:
        raise ValueError(f"Unsupported relation names: {sorted(unknown)}")

    node_accessions = np.asarray(list(dict.fromkeys(str(value) for value in accessions)))
    if len(node_accessions) != len(accessions):
        raise ValueError("Protein accessions must be unique for a relation graph")
    index = {accession: position for position, accession in enumerate(node_accessions)}
    pieces: list[pd.DataFrame] = []

    for relation_id, relation_name in enumerate(RELATION_NAMES):
        frame = relation_frames.get(relation_name)
        if frame is None or frame.empty:
            continue
        edges = _validated_edges(frame, index)
        if edges.empty:
            continue
        if relation_name in _UNDIRECTED_RELATIONS:
            reverse = edges.rename(
                columns={"source_index": "target_index", "target_index": "source_index"}
            )
            edges = pd.concat([edges, reverse], ignore_index=True)
        edges["relation"] = relation_id
        pieces.append(edges)

    if not pieces:
        return {
            "node_accessions": node_accessions,
            "edge_index": np.empty((2, 0), dtype=np.int64),
            "edge_weight": np.empty(0, dtype=np.float32),
            "relation": np.empty(0, dtype=np.int64),
        }

    edges = pd.concat(pieces, ignore_index=True)
    edges = (
        edges.groupby(["source_index", "target_index", "relation"], as_index=False, sort=True)[
            "weight"
        ]
        .max()
        .sort_values(["relation", "source_index", "target_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    return {
        "node_accessions": node_accessions,
        "edge_index": edges[["source_index", "target_index"]].to_numpy(dtype=np.int64).T,
        "edge_weight": edges["weight"].to_numpy(dtype=np.float32),
        "relation": edges["relation"].to_numpy(dtype=np.int64),
    }


def sample_relation_neighbors(graph: Mapping[str, np.ndarray], limit: int = 64) -> dict[str, np.ndarray]:
    """Create a fixed-size outgoing neighbourhood for every protein.

    The deterministic ordering is relation type, decreasing evidence weight,
    then target index. This keeps batches reproducible while preserving the
    highest-confidence neighbour when the graph is truncated.
    """
    if limit <= 0:
        raise ValueError("Neighbour limit must be positive")
    node_count = len(graph["node_accessions"])
    edge_index = np.asarray(graph["edge_index"], dtype=np.int64)
    weights = np.asarray(graph["edge_weight"], dtype=np.float32)
    relations = np.asarray(graph["relation"], dtype=np.int64)
    if edge_index.shape != (2, len(weights)) or len(relations) != len(weights):
        raise ValueError("Relation graph edge arrays have inconsistent dimensions")

    result = {
        "index": np.full((node_count, limit), -1, dtype=np.int64),
        "weight": np.zeros((node_count, limit), dtype=np.float32),
        "relation": np.zeros((node_count, limit), dtype=np.int64),
        "mask": np.zeros((node_count, limit), dtype=np.float32),
    }
    for source in range(node_count):
        edge_positions = np.flatnonzero(edge_index[0] == source)
        if not len(edge_positions):
            continue
        order = np.lexsort(
            (
                edge_index[1, edge_positions],
                -weights[edge_positions],
                relations[edge_positions],
            )
        )[:limit]
        selected = edge_positions[order]
        count = len(selected)
        result["index"][source, :count] = edge_index[1, selected]
        result["weight"][source, :count] = weights[selected]
        result["relation"][source, :count] = relations[selected]
        result["mask"][source, :count] = 1.0
    return result
