"""Export held-out phosphosite candidates for client-facing review."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.multimodal.client_ranking import rank_by_balanced_confidence


def export_candidates(run_dir, top_k=10):
    run_dir = Path(run_dir)
    predictions = pd.read_parquet(run_dir / "test_predictions.parquet")
    required = {
        "site_id", "activity_probability", "interaction_probability", "proteostasis_probability"
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"test predictions are missing columns: {sorted(missing)}")
    ranked = rank_by_balanced_confidence(
        predictions["site_id"].to_numpy(),
        predictions[
            ["activity_probability", "interaction_probability", "proteostasis_probability"]
        ].to_numpy(),
    )
    ranked.insert(1, "prediction_split", "held_out_test")
    ranked.insert(2, "interpretation", "high_priority_candidate_not_experimental_proof")
    ranked.to_parquet(run_dir / "candidate_ranking.parquet", index=False)
    ranked.to_csv(run_dir / "candidate_ranking.csv", index=False)
    ranked.head(int(top_k)).to_csv(run_dir / f"top_{int(top_k)}_candidates.csv", index=False)
    return ranked


def main(argv=None):
    parser = argparse.ArgumentParser(description="Rank held-out phosphosite candidates")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    ranked = export_candidates(args.run_dir, args.top_k)
    print(ranked.head(args.top_k).to_csv(index=False))


if __name__ == "__main__":
    main()
