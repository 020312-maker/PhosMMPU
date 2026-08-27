"""Create fixed-test PhosMMPU subgroup reports from saved seed outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.multimodal.subgroup_analysis import (
    build_subgroup_tables,
    download_uniprot_family_annotations,
    ensemble_validation_thresholds,
    load_saved_seed_ensemble,
    render_subgroup_figures,
    write_subgroup_readme,
    write_subgroup_tables,
)


SEEDS = (11, 23, 37, 51, 73)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "multimodal" / "training_runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "multimodal" / "subgroup_analysis",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--uniprot-batch-size", type=int, default=100)
    args = parser.parse_args(argv)

    ensemble = load_saved_seed_ensemble(args.processed_dir, args.run_root, SEEDS)
    thresholds = ensemble_validation_thresholds(
        ensemble["validation_labels"], ensemble["validation_scores"]
    )
    index = pd.read_parquet(args.processed_dir / "site_index.parquet").set_index("site_id")
    test_index = index.loc[ensemble["test_site_ids"]].reset_index()
    if test_index["site_id"].astype(str).tolist() != ensemble["test_site_ids"].astype(str).tolist():
        raise ValueError("test site index order does not match saved seed predictions")

    families, uniprot_audit = download_uniprot_family_annotations(
        test_index["accession"].astype(str).unique(),
        batch_size=args.uniprot_batch_size,
    )
    clusters = pd.read_csv(args.processed_dir / "homology_clusters.csv")
    tables = build_subgroup_tables(
        test_index,
        ensemble["test_labels"],
        ensemble["test_scores"],
        thresholds,
        families,
        clusters,
        bootstrap_repeats=args.bootstrap_repeats,
    )

    output_dir = args.output_dir.resolve()
    paths = write_subgroup_tables(tables, output_dir)
    paths.update(render_subgroup_figures(tables, output_dir))
    uniprot_audit.update(
        {
            "analysis_type": "post_hoc_fixed_test_subgroup_analysis",
            "seed_ids": list(ensemble["seeds"]),
            "test_site_count": int(len(test_index)),
            "validation_thresholds": {
                "activity": thresholds[0],
                "interaction": thresholds[1],
                "proteostasis": thresholds[2],
            },
            "bootstrap_repeats": int(args.bootstrap_repeats),
        }
    )
    paths["readme"] = write_subgroup_readme(output_dir, uniprot_audit)
    paths["uniprot_metadata"] = output_dir / "uniprot_family_download_audit.json"
    paths["uniprot_metadata"].write_text(
        json.dumps(uniprot_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "files": {key: str(value) for key, value in paths.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

