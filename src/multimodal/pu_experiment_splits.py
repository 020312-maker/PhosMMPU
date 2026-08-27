"""Deterministic, leakage-audited split manifests for PU experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.multimodal.splits import LABEL_COLUMNS, assign_splits


STRATEGY_GROUPS = {
    "random_site": "site_id",
    "protein": "accession",
    "homology_cluster": "cluster_id",
}
SPLIT_NAMES = ("train", "validation", "test")


def _validate_frame(frame):
    required = {"site_id", "accession", "cluster_id", *LABEL_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"experiment frame is missing columns: {sorted(missing)}")
    if frame["site_id"].duplicated().any():
        raise ValueError("experiment frame contains duplicate site IDs")
    if frame[["site_id", "accession", "cluster_id"]].isna().any().any():
        raise ValueError("experiment frame contains missing grouping values")
    for name in LABEL_COLUMNS:
        if not set(frame[name].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{name} must be binary")


def _label_summary(frame):
    return {
        name: {
            "positive": int(frame[name].sum()),
            "unlabeled": int((frame[name] == 0).sum()),
            "positive_ratio": float(frame[name].mean()),
        }
        for name in LABEL_COLUMNS
    }


def _overlap(frame, group_column):
    split_count = frame.groupby(group_column, sort=True)["split"].nunique()
    return {
        "groups": int(split_count.size),
        "cross_split_groups": int((split_count > 1).sum()),
        "maximum_splits_per_group": int(split_count.max()),
    }


def _site_id_hash(site_ids):
    payload = "\n".join(map(str, site_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_experiment_split(frame, strategy, seed, fractions=(0.70, 0.15, 0.15)):
    """Return ordered split rows and an audit for one declared generalization regime."""
    if strategy not in STRATEGY_GROUPS:
        raise ValueError(f"unknown split strategy: {strategy}")
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("fractions must contain three positive values")
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to one")

    _validate_frame(frame)
    group_column = STRATEGY_GROUPS[strategy]
    working = frame[["site_id", "accession", "cluster_id", *LABEL_COLUMNS]].copy()
    working["_experiment_group"] = working[group_column].astype(str)
    assigned = assign_splits(
        working,
        seed=int(seed),
        group_col="_experiment_group",
        train_fraction=float(fractions[0]),
        validation_fraction=float(fractions[1]),
        test_fraction=float(fractions[2]),
    )
    assignment = assigned.set_index("site_id")
    result = working[["site_id", "accession", "cluster_id"]].copy()
    result["split"] = result["site_id"].map(assignment["split"])
    result["split_seed"] = result["site_id"].map(assignment["split_seed"])
    if result["split"].isna().any():
        raise AssertionError("split assignment did not cover every site")
    result["strategy"] = strategy
    result = result.sort_values("site_id", kind="mergesort").reset_index(drop=True)

    labelled = frame.copy()
    labelled["split"] = labelled["site_id"].map(
        result.set_index("site_id")["split"]
    )
    if labelled["split"].isna().any():
        raise AssertionError("label audit did not cover every site")
    split_counts = {}
    for name in SPLIT_NAMES:
        subset = labelled[labelled["split"].eq(name)]
        split_counts[name] = {
            "sites": int(len(subset)),
            "proteins": int(subset["accession"].nunique()),
            "homology_clusters": int(subset["cluster_id"].nunique()),
            "labels": _label_summary(subset),
        }
    audit = {
        "strategy": strategy,
        "seed": int(seed),
        "fractions": {"train": fractions[0], "validation": fractions[1], "test": fractions[2]},
        "sites": int(len(result)),
        "site_id_sha256": _site_id_hash(result["site_id"]),
        "split_counts": split_counts,
        "group_overlap": {
            "accession": _overlap(result, "accession"),
            "cluster_id": _overlap(result, "cluster_id"),
        },
    }
    if strategy == "protein" and audit["group_overlap"]["accession"]["cross_split_groups"]:
        raise AssertionError("protein split leaked proteins across partitions")
    if strategy == "homology_cluster" and audit["group_overlap"]["cluster_id"]["cross_split_groups"]:
        raise AssertionError("homology split leaked clusters across partitions")
    return result, audit


def write_experiment_split(frame, strategy, seed, output_directory):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    splits, audit = make_experiment_split(frame, strategy, seed)
    splits.to_parquet(output_directory / "splits.parquet", index=False)
    (output_directory / "split_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return splits, audit
