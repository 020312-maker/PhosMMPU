"""Run a same-split LightGBM BCE comparator for all three PU tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.multimodal.baselines import fit_lightgbm_baseline
from src.multimodal.pu_evaluation import build_unified_report_table, ensemble_bootstrap_report


TASK_NAMES = {"activity": "活性调控", "interaction": "分子关联", "proteostasis": "稳定性/降解", "macro": "宏平均"}


def build_processed_features(processed_root):
    """Build deterministic tabular features from the current processed modalities."""
    processed_root = Path(processed_root)
    sequence = np.load(processed_root / "sequence.npy", mmap_mode="r")
    structure = np.load(processed_root / "structure_processed.npy", mmap_mode="r")
    network = np.load(processed_root / "network_processed.npy", mmap_mode="r")
    proteomics = np.load(processed_root / "proteomics_processed.npy", mmap_mode="r")
    masks = np.load(processed_root / "modality_masks.npy", mmap_mode="r")
    frequency = sequence.mean(axis=1, dtype=np.float32)
    center = np.asarray(sequence[:, sequence.shape[1] // 2, :], dtype=np.float32)
    features = np.concatenate([frequency, center, structure, network, proteomics, masks], axis=1).astype(np.float32, copy=False)
    names = (
        [f"sequence_frequency_{index}" for index in range(frequency.shape[1])]
        + [f"sequence_center_{index}" for index in range(center.shape[1])]
        + [f"structure_{index}" for index in range(structure.shape[1])]
        + [f"network_{index}" for index in range(network.shape[1])]
        + [f"proteomics_{index}" for index in range(proteomics.shape[1])]
        + [f"modality_available_{index}" for index in range(masks.shape[1])]
    )
    if len(names) != features.shape[1] or not np.isfinite(features).all():
        raise ValueError("processed LightGBM features are invalid")
    return features, names


def _write_summary(output_root, labels, splits, validation_scores, test_scores, repeats):
    validation_rows, test_rows = np.flatnonzero(splits == "validation"), np.flatnonzero(splits == "test")
    report = ensemble_bootstrap_report(
        labels[validation_rows], validation_scores, labels[test_rows], test_scores, repeats=repeats,
    )
    detail = build_unified_report_table(report)
    detail["模型"] = "LightGBM（BCE对照）"
    detail["任务"] = detail["task"].map(TASK_NAMES)
    detail.rename(columns={
        "ensemble_value": "五种子集成点估计", "mean": "五种子均值", "std": "五种子标准差", "seed_count": "种子数",
        "ci_low": "Bootstrap下限", "ci_high": "Bootstrap上限",
        "valid_repeats": "Bootstrap有效重复数",
    }, inplace=True)
    detail.to_csv(Path(output_root) / "五种子测试指标汇总.csv", index=False, encoding="utf-8-sig")
    return detail


def run_lightgbm_comparison(processed_root, output_root, seeds, n_estimators=500, bootstrap_repeats=1000):
    """Fit five BCE LightGBM comparators on the immutable homology-cluster split."""
    processed_root, output_root = Path(processed_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    features, feature_names = build_processed_features(processed_root)
    labels = np.load(processed_root / "labels.npy").astype(np.int8)
    splits = np.load(processed_root / "split_names.npy", allow_pickle=False)
    site_ids = np.load(processed_root / "dataset_site_ids.npy", allow_pickle=False)
    train_rows, validation_rows, test_rows = (np.flatnonzero(splits == name) for name in ("train", "validation", "test"))
    if not (len(features) == len(labels) == len(site_ids) and len(train_rows) and len(validation_rows) and len(test_rows)):
        raise ValueError("LightGBM inputs do not align with the fixed split")
    validation_scores, test_scores = [], []
    for seed in seeds:
        run_root = output_root / f"bce_seed{int(seed)}"
        run_root.mkdir()
        _, probabilities = fit_lightgbm_baseline(
            features[train_rows], labels[train_rows], features, seed=int(seed),
            n_estimators=n_estimators,
        )
        validation, test = probabilities[validation_rows], probabilities[test_rows]
        np.save(run_root / "validation_probabilities.npy", validation.astype(np.float32))
        np.save(run_root / "test_probabilities.npy", test.astype(np.float32))
        np.save(run_root / "test_site_ids.npy", site_ids[test_rows])
        (run_root / "run.json").write_text(json.dumps({
            "model": "LightGBM（BCE对照）", "objective": "bce", "seed": int(seed),
            "n_estimators": int(n_estimators), "feature_count": int(features.shape[1]),
            "split": "homology_cluster", "unlabeled_treated_as_negative": True,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        validation_scores.append(validation)
        test_scores.append(test)
    summary = _write_summary(output_root, labels, splits, np.stack(validation_scores), np.stack(test_scores), bootstrap_repeats)
    return {"output": str(output_root), "summary_rows": int(len(summary)), "feature_count": int(features.shape[1])}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run current-split LightGBM BCE comparator")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 51, 73])
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args(argv)
    print(json.dumps(run_lightgbm_comparison(
        args.processed_dir, args.output_root, args.seeds, args.n_estimators, args.bootstrap_repeats,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
