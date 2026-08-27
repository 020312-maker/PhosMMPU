"""Run the requested PhosMMPU architectural and objective ablations."""

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


ABLATION_SETTINGS = {
    "remove_crossmodal_transformer": {
        "name": "去跨模态Transformer", "objective": "nnpu", "transformer_layers": 0,
        "modality_dropout": 0.2, "network_enabled": True,
    },
    "remove_modality_dropout": {
        "name": "去modality dropout", "objective": "nnpu", "transformer_layers": 2,
        "modality_dropout": 0.0, "network_enabled": True,
    },
    "bce_instead_nnpu": {
        "name": "nnPU替换为BCE", "objective": "bce", "transformer_layers": 2,
        "modality_dropout": 0.2, "network_enabled": True,
    },
}
TASK_NAMES = {"activity": "活性调控", "interaction": "分子关联", "proteostasis": "稳定性/降解", "macro": "宏平均"}


def _write_summary(root, setting, labels, splits, seeds, repeats):
    validation_rows, test_rows = np.flatnonzero(splits == "validation"), np.flatnonzero(splits == "test")
    objective = setting["objective"]
    validation = np.stack([np.load(root / f"{objective}_seed{seed}" / "validation_probabilities.npy") for seed in seeds])
    test = np.stack([np.load(root / f"{objective}_seed{seed}" / "test_probabilities.npy") for seed in seeds])
    report = ensemble_bootstrap_report(labels[validation_rows], validation, labels[test_rows], test, repeats=repeats)
    detail = build_unified_report_table(report)
    detail["模型"] = setting["name"]
    detail["任务"] = detail["task"].map(TASK_NAMES)
    detail.rename(columns={
        "ensemble_value": "五种子集成点估计", "mean": "五种子均值", "std": "五种子标准差", "seed_count": "种子数",
        "ci_low": "Bootstrap下限", "ci_high": "Bootstrap上限",
        "valid_repeats": "Bootstrap有效重复数",
    }, inplace=True)
    detail.to_csv(root / "五种子测试指标汇总.csv", index=False, encoding="utf-8-sig")
    return detail


def run_ablations(config, output_root, network_dir, ablations, seeds, epochs, patience, batch_size, device, bootstrap_repeats=1000):
    """Train the three requested ablations using the fixed complete-network sidecar."""
    output_root = Path(output_root)
    labels = np.load(config.processed_data / "labels.npy").astype(np.int8)
    splits = np.load(config.processed_data / "split_names.npy", allow_pickle=False)
    outputs = []
    for ablation in ablations:
        if ablation not in ABLATION_SETTINGS:
            raise ValueError(f"unknown ablation: {ablation}")
        setting = ABLATION_SETTINGS[ablation]
        root = output_root / ablation
        root.mkdir(parents=True, exist_ok=False)
        train_from_config(
            config, seeds=seeds, epochs=epochs, batch_size=batch_size, objective=setting["objective"],
            device=device, output_root=root, patience=patience,
            transformer_layers=setting["transformer_layers"], modality_dropout=setting["modality_dropout"],
            network_dir=network_dir, network_variant="complete", network_enabled=setting["network_enabled"],
        )
        detail = _write_summary(root, setting, labels, splits, seeds, bootstrap_repeats)
        outputs.append({"ablation": ablation, "rows": int(len(detail)), "output": str(root)})
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run required PhosMMPU ablations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--processed-dir")
    parser.add_argument("--network-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ablations", nargs="+", choices=tuple(ABLATION_SETTINGS), default=tuple(ABLATION_SETTINGS))
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
    print(json.dumps(run_ablations(
        config, args.output_root, args.network_dir, args.ablations, args.seeds, args.epochs,
        args.patience, args.batch_size, args.device, args.bootstrap_repeats,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
