"""Recompute unified ensemble Bootstrap summaries from saved seed predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.multimodal.pu_evaluation import (
    build_unified_report_table,
    ensemble_bootstrap_report,
    paired_bootstrap_difference,
    validation_f1_thresholds,
)


TASK_NAMES = {"activity": "活性调控", "interaction": "分子关联", "proteostasis": "稳定性/降解", "macro": "宏平均"}


def load_saved_seed_predictions(processed_dir, run_root, objective, seeds):
    """Load fixed-split labels and saved validation/test probabilities without training."""
    processed_dir, run_root = Path(processed_dir), Path(run_root)
    labels = np.load(processed_dir / "labels.npy").astype(np.int8)
    splits = np.load(processed_dir / "split_names.npy", allow_pickle=False)
    validation_rows = np.flatnonzero(splits == "validation")
    test_rows = np.flatnonzero(splits == "test")
    validation = np.stack([np.load(run_root / f"{objective}_seed{int(seed)}" / "validation_probabilities.npy") for seed in seeds])
    test = np.stack([np.load(run_root / f"{objective}_seed{int(seed)}" / "test_probabilities.npy") for seed in seeds])
    return labels[validation_rows], validation, labels[test_rows], test


def _format_ensemble_detail(report, model_name):
    detail = build_unified_report_table(report)
    detail["模型"] = str(model_name)
    detail["任务"] = detail["task"].map(TASK_NAMES)
    detail["Bootstrap单位"] = "测试位点"
    detail.rename(columns={
        "ensemble_value": "五种子集成点估计", "mean": "五种子单种子均值", "std": "五种子单种子标准差",
        "seed_count": "种子数", "ci_low": "95%CI下限", "ci_high": "95%CI上限",
        "valid_repeats": "Bootstrap有效重复数",
    }, inplace=True)
    # Keep prior summary consumers working while the new headers state the
    # statistical unit explicitly.
    detail["五种子均值"] = detail["五种子单种子均值"]
    detail["五种子标准差"] = detail["五种子单种子标准差"]
    detail["Bootstrap下限"] = detail["95%CI下限"]
    detail["Bootstrap上限"] = detail["95%CI上限"]
    return detail


def _format_per_seed_detail(report, model_name):
    detail = report["per_seed_bootstrap"].copy()
    detail["模型"] = str(model_name)
    detail["任务"] = detail["task"].map(TASK_NAMES)
    detail.rename(columns={
        "ensemble_value": "单种子点估计", "ci_low": "95%CI下限", "ci_high": "95%CI上限",
        "valid_repeats": "Bootstrap有效重复数",
    }, inplace=True)
    return detail.sort_values(["seed", "task", "metric"], kind="mergesort").reset_index(drop=True)


def recompute_saved_run_statistics(processed_dir, run_root, objective, seeds, model_name, output_dir, repeats=1000):
    """Write separate single-seed and five-seed test-site Bootstrap reports."""
    validation_labels, validation, test_labels, test = load_saved_seed_predictions(
        processed_dir, run_root, objective, seeds,
    )
    report = ensemble_bootstrap_report(
        validation_labels, validation, test_labels, test, repeats=repeats, seed_ids=seeds,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensemble = _format_ensemble_detail(report, model_name)
    per_seed = _format_per_seed_detail(report, model_name)
    seed_summary = report["seed_summary"].copy()
    seed_summary["模型"] = str(model_name)
    seed_summary["任务"] = seed_summary["task"].map(TASK_NAMES)
    seed_summary.rename(columns={"mean": "单种子均值", "std": "单种子标准差", "seed_count": "种子数"}, inplace=True)
    ensemble.to_csv(output_dir / "五种子集成_测试位点Bootstrap_95CI.csv", index=False, encoding="utf-8-sig")
    per_seed.to_csv(output_dir / "单种子_测试位点Bootstrap_95CI.csv", index=False, encoding="utf-8-sig")
    seed_summary.to_csv(output_dir / "五种子随机种子统计.csv", index=False, encoding="utf-8-sig")
    return {"report": report, "ensemble": ensemble, "per_seed": per_seed, "seed_summary": seed_summary}


def paired_saved_run_difference(
    processed_dir, reference_root, reference_objective, reference_model,
    candidate_root, candidate_objective, candidate_model, seeds, repeats=1000,
):
    """Compare saved five-seed ensembles using identical Bootstrap test-site draws."""
    ref_validation_labels, ref_validation, ref_test_labels, ref_test = load_saved_seed_predictions(
        processed_dir, reference_root, reference_objective, seeds,
    )
    candidate_validation_labels, candidate_validation, candidate_test_labels, candidate_test = load_saved_seed_predictions(
        processed_dir, candidate_root, candidate_objective, seeds,
    )
    if not (
        np.array_equal(ref_validation_labels, candidate_validation_labels)
        and np.array_equal(ref_test_labels, candidate_test_labels)
    ):
        raise ValueError("paired comparison requires identical validation and test labels")
    comparison = paired_bootstrap_difference(
        ref_test_labels,
        ref_test.mean(axis=0), validation_f1_thresholds(ref_validation_labels, ref_validation.mean(axis=0)),
        candidate_test.mean(axis=0), validation_f1_thresholds(candidate_validation_labels, candidate_validation.mean(axis=0)),
        repeats=repeats,
    )
    comparison["参考模型"] = str(reference_model)
    comparison["对照模型"] = str(candidate_model)
    comparison["任务"] = comparison["task"].map(TASK_NAMES)
    comparison.rename(columns={
        "reference_value": "参考模型五种子集成点估计",
        "candidate_value": "对照模型五种子集成点估计",
        "difference": "差值（完整模型-对照模型）",
        "ci_low": "差值95%CI下限",
        "ci_high": "差值95%CI上限",
        "valid_repeats": "Bootstrap有效重复数",
        "bootstrap_unit": "Bootstrap单位",
    }, inplace=True)
    comparison["Bootstrap单位"] = comparison["Bootstrap单位"].replace({"paired_test_sites": "配对测试位点"})
    return comparison.sort_values(["task", "metric"], kind="mergesort").reset_index(drop=True)


def recompute_saved_run_summary(processed_dir, run_root, objective, seeds, model_name, output_file, repeats=1000):
    """Compatibility wrapper writing the five-seed ensemble statistics to one CSV."""
    output_file = Path(output_file)
    reports = recompute_saved_run_statistics(
        processed_dir, run_root, objective, seeds, model_name, output_file.parent, repeats,
    )
    detail = reports["ensemble"]
    detail.to_csv(output_file, index=False, encoding="utf-8-sig")
    return detail


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recompute unified Bootstrap summaries from saved predictions")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 51, 73])
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args(argv)
    detail = recompute_saved_run_summary(
        args.processed_dir, args.run_root, args.objective, args.seeds, args.model,
        args.output, args.bootstrap_repeats,
    )
    print(f"rows={len(detail)} output={args.output}")


if __name__ == "__main__":
    main()
