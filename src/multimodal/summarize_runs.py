"""Summarize repeated nnPU runs and ensemble held-out predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.multimodal.client_ranking import LABEL_NAMES, WEIGHTS


def summarize_runs(run_root, expected_seeds=None):
    run_root = Path(run_root)
    run_dirs = sorted(path for path in run_root.glob("nnpu_seed*") if path.is_dir())
    if expected_seeds is not None:
        expected = {f"nnpu_seed{int(seed)}" for seed in expected_seeds}
        observed = {path.name for path in run_dirs}
        missing = expected.difference(observed)
        if missing:
            raise FileNotFoundError(f"missing repeated runs: {sorted(missing)}")
        run_dirs = [run_root / f"nnpu_seed{int(seed)}" for seed in expected_seeds]
    if not run_dirs:
        raise FileNotFoundError("no nnPU run directories were found")

    manifests = []
    predictions = []
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        frame = pd.read_parquet(run_dir / "test_predictions.parquet").sort_values("site_id")
        manifests.append({"run_dir": str(run_dir), **manifest})
        predictions.append(frame.reset_index(drop=True))
    site_ids = predictions[0]["site_id"].astype(str).to_numpy()
    if any(not np.array_equal(site_ids, frame["site_id"].astype(str).to_numpy()) for frame in predictions[1:]):
        raise ValueError("test site order differs across repeated runs")

    score_columns = [f"{name}_probability" for name in LABEL_NAMES]
    stack = np.stack([frame[score_columns].to_numpy(dtype=np.float64) for frame in predictions])
    ensemble = predictions[0][["site_id", "y_activity", "y_interaction", "y_proteostasis"]].copy()
    for column, mean, std in zip(score_columns, stack.mean(axis=0).T, stack.std(axis=0, ddof=0).T):
        ensemble[column] = mean
        ensemble[column.replace("_probability", "_seed_std")] = std
    ensemble["balanced_confidence"] = ensemble[score_columns].to_numpy().dot(WEIGHTS)
    ensemble = ensemble.sort_values("balanced_confidence", ascending=False, kind="stable").reset_index(drop=True)
    ensemble.insert(0, "rank", ensemble.index + 1)
    ensemble.insert(1, "prediction_split", "held_out_test")
    ensemble.insert(2, "seed_count", len(predictions))

    metrics = np.asarray([item["best_validation_macro_auprc"] for item in manifests], dtype=float)
    summary = {
        "seeds": [int(item["seed"]) for item in manifests],
        "run_count": len(manifests),
        "validation_macro_auprc_mean": float(metrics.mean()),
        "validation_macro_auprc_std": float(metrics.std(ddof=0)),
        "validation_macro_auprc_values": metrics.tolist(),
        "evaluation_split": "held_out_test",
        "species_scope": "human_only",
    }
    (run_root / "formal_5seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ensemble.to_parquet(run_root / "formal_5seed_ensemble_candidates.parquet", index=False)
    ensemble.to_csv(run_root / "formal_5seed_ensemble_candidates.csv", index=False)
    ensemble.head(10).to_csv(run_root / "formal_5seed_top10.csv", index=False)
    return summary, ensemble


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize repeated nnPU training runs")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+")
    args = parser.parse_args(argv)
    summary, _ = summarize_runs(args.run_root, args.expected_seeds)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
