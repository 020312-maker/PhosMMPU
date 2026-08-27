import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.multimodal.config import MultimodalConfig


LABEL_COLUMNS = ["y_activity", "y_interaction", "y_proteostasis"]


def _split_once(groups, strata, test_size, seed):
    groups = np.asarray(groups)
    strata = np.asarray(strata)
    counts = pd.Series(strata).value_counts()
    stratify = strata if not counts.empty and counts.min() >= 2 else None
    try:
        return train_test_split(
            groups,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            groups,
            test_size=test_size,
            random_state=seed,
            stratify=None,
        )


def assign_splits(
    frame,
    seed,
    group_col="accession",
    train_fraction=0.70,
    validation_fraction=0.15,
    test_fraction=0.15,
):
    required = {"site_id", group_col, *LABEL_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if frame["site_id"].duplicated().any():
        raise ValueError("site_id must be unique before assigning splits")
    if frame[group_col].isna().any():
        raise ValueError(f"{group_col} contains missing values")
    fractions = train_fraction + validation_fraction + test_fraction
    if abs(fractions - 1.0) > 1e-9 or min(
        train_fraction, validation_fraction, test_fraction
    ) <= 0:
        raise ValueError("split fractions must be positive and sum to 1")

    grouped = frame.groupby(group_col, sort=True)[LABEL_COLUMNS].max().reset_index()
    if len(grouped) < 3:
        raise ValueError("at least three groups are required")
    grouped["stratum"] = (
        grouped["y_activity"]
        + 2 * grouped["y_interaction"]
        + 4 * grouped["y_proteostasis"]
    )
    holdout_fraction = validation_fraction + test_fraction
    train_groups, holdout_groups = _split_once(
        grouped[group_col].to_numpy(),
        grouped["stratum"].to_numpy(),
        holdout_fraction,
        seed,
    )
    holdout = grouped[grouped[group_col].isin(holdout_groups)]
    validation_groups, test_groups = _split_once(
        holdout[group_col].to_numpy(),
        holdout["stratum"].to_numpy(),
        test_fraction / holdout_fraction,
        seed + 1,
    )
    assignment = {group: "train" for group in train_groups}
    assignment.update({group: "validation" for group in validation_groups})
    assignment.update({group: "test" for group in test_groups})
    result = frame[["site_id", group_col]].copy()
    result["split"] = result[group_col].map(assignment)
    result["split_seed"] = int(seed)
    if result["split"].isna().any():
        raise ValueError("at least one group was not assigned")
    if (result.groupby(group_col)["split"].nunique() != 1).any():
        raise AssertionError(f"{group_col} crosses dataset splits")
    return result[["site_id", group_col, "split", "split_seed"]]


def _read_cluster_map(path):
    clusters = pd.read_csv(path, sep=None, engine="python")
    required = {"accession", "cluster_id"}
    missing = required - set(clusters.columns)
    if missing:
        raise ValueError(
            "cluster map must have accession and cluster_id columns; "
            f"missing {sorted(missing)}"
        )
    clusters = clusters[["accession", "cluster_id"]].copy()
    clusters["accession"] = clusters["accession"].astype(str)
    clusters["cluster_id"] = clusters["cluster_id"].astype(str)
    if clusters["accession"].duplicated().any():
        raise ValueError("cluster map contains duplicate accessions")
    return clusters


def _audit_splits(frame, splits, group_col, seed, cluster_map=None):
    annotated = frame.merge(
        splits[["site_id", "split"]], on="site_id", how="left", validate="one_to_one"
    )
    split_counts = {}
    for split in ("train", "validation", "test"):
        subset = annotated[annotated["split"].eq(split)]
        split_counts[split] = {
            "sites": int(len(subset)),
            "groups": int(subset[group_col].nunique()),
            "positive_counts": {
                label: int(subset[label].sum()) for label in LABEL_COLUMNS
            },
        }
    return {
        "seed": int(seed),
        "group_column": group_col,
        "cluster_map": str(Path(cluster_map).resolve()) if cluster_map else None,
        "sites": int(len(splits)),
        "groups": int(splits[group_col].nunique()),
        "maximum_splits_per_group": int(
            splits.groupby(group_col)["split"].nunique().max()
        ),
        "split_counts": split_counts,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create leakage-safe dataset splits")
    parser.add_argument("--config", required=True)
    parser.add_argument("--group-column", default="accession")
    parser.add_argument("--cluster-map")
    args = parser.parse_args(argv)

    config = MultimodalConfig.load(args.config)
    config.ensure_output_directories()
    index = pd.read_parquet(config.processed_data / "site_index.parquet")
    labels = pd.read_parquet(config.processed_data / "labels.parquet")
    frame = index[["site_id", "accession"]].merge(
        labels[["site_id", *LABEL_COLUMNS]],
        on="site_id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(index):
        raise ValueError("labels do not cover every indexed site")

    group_col = args.group_column
    if args.cluster_map:
        clusters = _read_cluster_map(args.cluster_map)
        frame = frame.merge(clusters, on="accession", how="left", validate="many_to_one")
        if frame["cluster_id"].isna().any():
            missing = int(frame.loc[frame["cluster_id"].isna(), "accession"].nunique())
            raise ValueError(f"cluster map is missing {missing} retained accessions")
        group_col = "cluster_id"
    elif group_col not in frame.columns:
        raise ValueError(f"group column is unavailable: {group_col}")

    splits = assign_splits(
        frame,
        seed=config.seed,
        group_col=group_col,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    if group_col != "accession":
        splits = splits.merge(
            frame[["site_id", "accession"]],
            on="site_id",
            how="left",
            validate="one_to_one",
        )[["site_id", "accession", group_col, "split", "split_seed"]]
    split_path = config.processed_data / "splits.parquet"
    audit_path = config.processed_data / "split_audit.json"
    splits.to_parquet(split_path, index=False)
    audit = _audit_splits(frame, splits, group_col, config.seed, args.cluster_map)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
