"""Run fixed-split nnPU positive-prior sensitivity conditions."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.multimodal.config import MultimodalConfig
from src.multimodal.pu_evaluation import build_unified_report_table, ensemble_bootstrap_report
from src.multimodal.train import train_from_config


PRIOR_MULTIPLIERS = (1.0, 1.5, 2.0, 3.0)
TASK_NAMES = {"activity": "活性调控", "interaction": "分子关联", "proteostasis": "稳定性/降解", "macro": "宏平均"}


def prior_name(multiplier):
    if float(multiplier) not in PRIOR_MULTIPLIERS:
        raise ValueError(f"unsupported prior multiplier: {multiplier}")
    return f"prior_{float(multiplier):.1f}x"


def _summary(root, multiplier, labels, splits, seeds, repeats):
    validation_rows = np.flatnonzero(splits == "validation")
    test_rows = np.flatnonzero(splits == "test")
    validation = np.stack([np.load(root / f"nnpu_seed{seed}" / "validation_probabilities.npy") for seed in seeds])
    test = np.stack([np.load(root / f"nnpu_seed{seed}" / "test_probabilities.npy") for seed in seeds])
    report = ensemble_bootstrap_report(labels[validation_rows], validation, labels[test_rows], test, repeats=repeats)
    detail = build_unified_report_table(report)
    detail["先验倍数"] = float(multiplier)
    detail["任务"] = detail["task"].map(TASK_NAMES)
    detail.rename(columns={
        "ensemble_value": "五种子集成点估计", "mean": "五种子均值", "std": "五种子标准差", "seed_count": "种子数",
        "ci_low": "Bootstrap下限", "ci_high": "Bootstrap上限",
        "valid_repeats": "Bootstrap有效重复数",
    }, inplace=True)
    detail.to_csv(root / "五种子测试指标汇总.csv", index=False, encoding="utf-8-sig")
    return detail


def run_prior_sensitivity(config, output_root, network_dir, seeds, epochs, patience, batch_size, device, bootstrap_repeats=1000):
    """Train four nnPU priors on the unchanged homology split and complete network."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    labels = np.load(config.processed_data / "labels.npy").astype(np.int8)
    splits = np.load(config.processed_data / "split_names.npy", allow_pickle=False)
    details = []
    for multiplier in PRIOR_MULTIPLIERS:
        root = output_root / prior_name(multiplier)
        root.mkdir()
        prior_config = replace(config, pu_prior_multiplier=float(multiplier))
        train_from_config(
            prior_config, seeds=seeds, epochs=epochs, batch_size=batch_size, objective="nnpu", device=device,
            output_root=root, patience=patience, network_dir=network_dir, network_variant="complete", network_enabled=True,
        )
        details.append(_summary(root, multiplier, labels, splits, seeds, bootstrap_repeats))
    result = pd.concat(details, ignore_index=True)
    result.to_csv(output_root / "nnPU阳性先验敏感性汇总.csv", index=False, encoding="utf-8-sig")
    return {"output": str(output_root), "conditions": [prior_name(value) for value in PRIOR_MULTIPLIERS]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run nnPU positive-prior sensitivity study")
    parser.add_argument("--config", required=True)
    parser.add_argument("--processed-dir")
    parser.add_argument("--network-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 51, 73])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args(argv)
    config = MultimodalConfig.load(args.config)
    if args.processed_dir:
        config = replace(config, processed_data=Path(args.processed_dir).resolve())
    print(json.dumps(run_prior_sensitivity(
        config, args.output_root, args.network_dir, args.seeds, args.epochs, args.patience,
        args.batch_size, args.device, args.bootstrap_repeats,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
