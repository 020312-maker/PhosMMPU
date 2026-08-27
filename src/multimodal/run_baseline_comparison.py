"""Run auditable three-task baselines on the fixed homology-cluster split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.multimodal.pu_evaluation import build_unified_report_table, ensemble_bootstrap_report
from src.multimodal.sequence_baselines import train_sequence_baseline


MODEL_NAMES = {
    "onehot_cnn": "One-hot + CNN",
    "esm_mlp": "ESM-2 + MLP",
    "funcphos_seq_style_cnn": "FuncPhos-SEQ式序列CNN（适配）",
}
TASK_NAMES = {"activity": "活性调控", "interaction": "分子关联", "proteostasis": "稳定性/降解", "macro": "宏平均"}


def summarize_score_runs(validation_labels, validation_scores, test_labels, test_scores, repeats=1000):
    """Return five-seed summaries and stratified bootstrap intervals."""
    report = ensemble_bootstrap_report(
        validation_labels, validation_scores, test_labels, test_scores, repeats=repeats,
    )
    return report["seed_summary"], report["bootstrap"]


def _write_model_summary(model_root, model_name, validation_labels, validation_scores, test_labels, test_scores, repeats):
    report = ensemble_bootstrap_report(
        validation_labels, validation_scores, test_labels, test_scores, repeats=repeats,
    )
    detail = build_unified_report_table(report)
    detail["模型"] = MODEL_NAMES[model_name]
    detail["任务"] = detail["task"].map(TASK_NAMES)
    detail.rename(columns={
        "ensemble_value": "五种子集成点估计", "mean": "五种子均值", "std": "五种子标准差", "seed_count": "种子数",
        "ci_low": "Bootstrap下限", "ci_high": "Bootstrap上限",
        "valid_repeats": "Bootstrap有效重复数",
    }, inplace=True)
    detail.to_csv(model_root / "五种子测试指标汇总.csv", index=False, encoding="utf-8-sig")
    return detail


def run_sequence_baselines(processed_root, output_root, models, seeds, epochs, patience, batch_size, device, bootstrap_repeats=1000):
    """Train sequence baselines with nnPU and save per-seed prediction scores."""
    processed_root, output_root = Path(processed_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    sequence = np.load(processed_root / "sequence.npy", mmap_mode="r")
    sequence_aux = np.load(processed_root / "sequence_aux.npy", mmap_mode="r")
    labels = np.load(processed_root / "labels.npy").astype(np.int8)
    splits = np.load(processed_root / "split_names.npy", allow_pickle=False)
    site_ids = np.load(processed_root / "dataset_site_ids.npy", allow_pickle=False)
    validation_rows = np.flatnonzero(splits == "validation")
    test_rows = np.flatnonzero(splits == "test")
    if not (len(validation_rows) and len(test_rows)):
        raise ValueError("fixed split must contain validation and test rows")
    outputs = []
    for model_name in models:
        if model_name not in MODEL_NAMES:
            raise ValueError(f"unknown model: {model_name}")
        model_root = output_root / model_name
        model_root.mkdir(parents=True, exist_ok=True)
        validation_scores, test_scores = [], []
        for seed in seeds:
            run_root = model_root / f"nnpu_seed{int(seed)}"
            run_root.mkdir(parents=True, exist_ok=True)
            _, test, detail = train_sequence_baseline(
                model_name, sequence, sequence_aux, labels, splits, site_ids, run_root,
                seed=seed, epochs=epochs, batch_size=batch_size, patience=patience, device=device,
            )
            validation = np.asarray(detail["validation_probabilities"], dtype=np.float32)
            np.save(run_root / "validation_probabilities.npy", validation)
            np.save(run_root / "test_probabilities.npy", test)
            (run_root / "run.json").write_text(json.dumps({
                "model": MODEL_NAMES[model_name], "objective": "nnpu", "seed": int(seed),
                "device": str(device), "epochs_trained": int(detail["epochs_trained"]),
                "best_validation_macro_auprc": float(detail["best_validation_macro_auprc"]),
                "class_priors": [float(value) for value in detail["class_priors"]],
                "split": "homology_cluster",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            validation_scores.append(validation)
            test_scores.append(test)
        validation_scores, test_scores = np.stack(validation_scores), np.stack(test_scores)
        summary = _write_model_summary(
            model_root, model_name, labels[validation_rows], validation_scores, labels[test_rows], test_scores,
            bootstrap_repeats,
        )
        outputs.append({"model": model_name, "rows": int(len(summary)), "output": str(model_root)})
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run three-task nnPU sequence baselines")
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--models", nargs="+", choices=tuple(MODEL_NAMES), default=tuple(MODEL_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 51, 73])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args(argv)
    print(json.dumps(run_sequence_baselines(
        args.processed_dir, args.output_root, args.models, args.seeds, args.epochs, args.patience,
        args.batch_size, args.device, args.bootstrap_repeats,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
