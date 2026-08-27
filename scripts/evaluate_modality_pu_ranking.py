"""Calculate PU ranking metrics for saved modality-ablation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.multimodal.pu_ranking_metrics import ranking_metrics


TASKS = ("activity", "interaction", "proteostasis")


def _rows(model, labels, scores, priors, seed=None, k_values=(100, 200, 500), g_k=100):
    records = []
    for column, task in enumerate(TASKS):
        values = ranking_metrics(
            labels[:, column], scores[:, column], priors[column], k_values, g_k
        )
        for metric, value in values.items():
            records.append(
                {
                    "model": model,
                    "task": task,
                    "metric": metric,
                    "value": float(value),
                    "class_prior": float(priors[column]),
                    "seed": seed,
                }
            )
    return records


def summarize_model_scores(
    model, labels, seed_scores, priors, seeds, k_values=(100, 200, 500), g_k=100
):
    """Return ensemble metrics and per-seed summary statistics."""
    seed_scores = np.asarray(seed_scores, dtype=float)
    if seed_scores.ndim != 3 or seed_scores.shape[1:] != labels.shape:
        raise ValueError("seed scores must align with labels")
    if len(seed_scores) != len(seeds):
        raise ValueError("seed scores must align with seed identifiers")

    records = []
    for seed, scores in zip(seeds, seed_scores):
        records.extend(
            _rows(model, labels, scores, priors, int(seed), k_values, g_k)
        )
    per_seed = pd.DataFrame(records)
    seed_summary = (
        per_seed.groupby(
            ["model", "task", "metric", "class_prior"], as_index=False
        )["value"]
        .agg(seed_mean="mean", seed_std="std", seed_count="count")
    )
    ensemble = pd.DataFrame(
        _rows(model, labels, seed_scores.mean(axis=0), priors, None, k_values, g_k)
    ).rename(columns={"value": "ensemble_value"})
    return ensemble, seed_summary


def random_baseline(
    labels, priors, repeats=1000, seed=20260817, k_values=(100, 200, 500), g_k=100
):
    """Estimate random-ranking statistics on the fixed test set."""
    generator = np.random.default_rng(seed)
    records = []
    for repeat in range(int(repeats)):
        scores = generator.random(labels.shape)
        records.extend(
            _rows("random", labels, scores, priors, repeat, k_values, g_k)
        )
    values = pd.DataFrame(records)
    return (
        values.groupby(
            ["model", "task", "metric", "class_prior"], as_index=False
        )["value"]
        .agg(random_mean="mean", random_std="std", repeat_count="count")
    )


def _load_model_specs(path):
    specs = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(specs, list) or not specs:
        raise ValueError("models JSON must contain a non-empty list")
    return specs


def evaluate(processed_dir, specs_path, output_dir, random_repeats=1000):
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)
    labels = np.load(processed_dir / "labels.npy").astype(np.int8)
    splits = np.load(processed_dir / "split_names.npy", allow_pickle=False)
    test_labels = labels[splits == "test"]

    ensemble_frames = []
    seed_frames = []
    expected_priors = None
    for spec in _load_model_specs(specs_path):
        seeds = tuple(int(seed) for seed in spec["seeds"])
        root = Path(spec["run_root"])
        manifests = [
            json.loads(
                (root / f"{spec['objective']}_seed{seed}" / "run.json").read_text(
                    encoding="utf-8"
                )
            )
            for seed in seeds
        ]
        priors = np.asarray(manifests[0]["class_priors"], dtype=float)
        if any(
            not np.allclose(priors, np.asarray(manifest["class_priors"], dtype=float))
            for manifest in manifests[1:]
        ):
            raise ValueError(f"class priors differ between seeds for {spec['model']}")
        if expected_priors is None:
            expected_priors = priors
        elif not np.allclose(priors, expected_priors):
            raise ValueError("modality ablations must use the same class priors")

        scores = np.stack(
            [
                np.load(
                    root
                    / f"{spec['objective']}_seed{seed}"
                    / "test_probabilities.npy"
                )
                for seed in seeds
            ]
        )
        ensemble, seed_summary = summarize_model_scores(
            spec["model"], test_labels, scores, priors, seeds
        )
        ensemble_frames.append(ensemble)
        seed_frames.append(seed_summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble = pd.concat(ensemble_frames, ignore_index=True).sort_values(
        ["model", "task", "metric"], kind="mergesort"
    )
    seed_summary = pd.concat(seed_frames, ignore_index=True).sort_values(
        ["model", "task", "metric"], kind="mergesort"
    )
    random = random_baseline(
        test_labels, expected_priors, repeats=random_repeats
    ).sort_values(["task", "metric"], kind="mergesort")

    ensemble.to_csv(output_dir / "pu_ranking_ensemble.csv", index=False)
    seed_summary.to_csv(output_dir / "pu_ranking_seed_summary.csv", index=False)
    random.to_csv(output_dir / "pu_ranking_random_baseline.csv", index=False)
    (output_dir / "METRICS.md").write_text(
        "# PU ranking metrics\n\n"
        "Known positives are sites whose task label equals one. Unlabeled sites "
        "are not interpreted as confirmed negatives. PR-AUC is therefore an "
        "observed positive-unlabeled estimate. Hits@K counts known positives in "
        "the top K predictions, Recall@K measures recovery of all known positives, "
        "and NDCG@K evaluates their ranked positions.\n",
        encoding="utf-8",
    )
    return {
        "ensemble": ensemble,
        "seed_summary": seed_summary,
        "random": random,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--models-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--random-repeats", type=int, default=1000)
    args = parser.parse_args(argv)
    outputs = evaluate(
        args.processed_dir, args.models_json, args.output_dir, args.random_repeats
    )
    print({name: len(frame) for name, frame in outputs.items()})


if __name__ == "__main__":
    main()
