"""Run auditable network-evidence sensitivity conditions for PhosMMPU."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.multimodal.config import MultimodalConfig
from src.multimodal.features.network import NETWORK_VARIANTS, build_network_variant
from src.multimodal.pu_evaluation import build_unified_report_table, ensemble_bootstrap_report, random_reference
from src.multimodal.train import train_from_config


VARIANT_NAMES = {
    "complete": "完整网络",
    "direct_evidence_masked": "直接证据屏蔽网络",
    "without_signor": "去SIGNOR网络",
    "ordinary_ppi": "普通PPI网络",
    "without_network": "无网络",
}
TASK_NAMES = {"activity": "活性调控", "interaction": "分子关联", "proteostasis": "稳定性/降解"}
FEATURE_SOURCES = "One-hot序列; ESM-2/motif; AlphaFold局部结构; STRING/BioGRID/SIGNOR网络; PXD蛋白组学"


@dataclass(frozen=True)
class OutputPaths:
    network_dir: Path
    training_dir: Path


def output_paths(output_root, variant):
    if variant not in VARIANT_NAMES:
        raise ValueError(f"unknown network variant: {variant}")
    root = Path(output_root)
    name = VARIANT_NAMES[variant]
    return OutputPaths(root / "网络版本" / name, root / "训练运行" / name)


def _labels_and_groups(processed_root):
    labels = np.load(processed_root / "labels.npy").astype(np.int8)
    splits = np.load(processed_root / "split_names.npy", allow_pickle=False)
    return labels, {name: np.flatnonzero(splits == name) for name in ("train", "validation", "test")}


def _metrics_tables(paths, labels, groups, seeds, repeats):
    validation = np.stack([np.load(paths.training_dir / f"nnpu_seed{seed}" / "validation_probabilities.npy") for seed in seeds])
    test = np.stack([np.load(paths.training_dir / f"nnpu_seed{seed}" / "test_probabilities.npy") for seed in seeds])
    report = ensemble_bootstrap_report(
        labels[groups["validation"]], validation, labels[groups["test"]], test, repeats=repeats
    )
    detail = build_unified_report_table(report)
    detail["网络版本"] = paths.training_dir.name
    return detail


def _write_reports(output_root, edge_rows, detail_frames):
    output_root = Path(output_root)
    pd.DataFrame(edge_rows).to_csv(output_root / "网络版本边数审计.csv", index=False, encoding="utf-8-sig")
    detail = pd.concat(detail_frames, ignore_index=True)
    detail["任务"] = detail["task"].map(TASK_NAMES).fillna("宏平均")
    detail["指标"] = detail["metric"]
    detail.rename(columns={"ensemble_value": "五种子集成点估计", "mean": "五种子均值", "std": "五种子标准差", "seed_count": "种子数", "ci_low": "Bootstrap下限", "ci_high": "Bootstrap上限", "valid_repeats": "Bootstrap有效重复数"}, inplace=True)
    detail.to_csv(output_root / "网络版本训练结果汇总.csv", index=False, encoding="utf-8-sig")


def write_random_baseline_report(validation_labels, test_labels, output_root, repeats=100, seed=20260806):
    """Write the validation-thresholded random-score reference for the fixed split."""
    values = random_reference(validation_labels, test_labels, repeats=repeats, seed=seed)
    values["任务"] = values["task"].map(TASK_NAMES)
    values.rename(
        columns={
            "repeat": "重复编号", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC",
            "f1": "F1", "mcc": "MCC", "precision": "Precision", "recall": "Recall",
        },
        inplace=True,
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    values.to_csv(output_root / "随机基线结果.csv", index=False, encoding="utf-8-sig")
    return values


def build_site_audit_table(site_index, labels, splits, clusters, data_version):
    table = site_index[["site_id", "accession", "residue", "position"]].merge(
        labels[["site_id", "y_activity", "y_interaction", "y_proteostasis"]], on="site_id", validate="one_to_one"
    ).merge(splits[["site_id", "split"]], on="site_id", validate="one_to_one").merge(
        clusters[["accession", "cluster_id"]], on="accession", how="left", validate="many_to_one"
    )
    if table["cluster_id"].isna().any():
        raise ValueError("site audit table has missing homology clusters")
    table["活性调控标签状态"] = np.where(table.pop("y_activity").eq(1), "P", "U")
    table["分子关联标签状态"] = np.where(table.pop("y_interaction").eq(1), "P", "U")
    table["稳定性/降解标签状态"] = np.where(table.pop("y_proteostasis").eq(1), "P", "U")
    table.rename(columns={"accession": "蛋白", "residue": "残基", "position": "位置", "split": "预测划分", "cluster_id": "同源簇"}, inplace=True)
    table["特征来源"] = FEATURE_SOURCES
    table["数据版本"] = str(data_version)
    return table


def write_site_audit_table(processed_root, output_root, data_version):
    processed_root = Path(processed_root)
    table = build_site_audit_table(
        pd.read_parquet(processed_root / "site_index.parquet"),
        pd.read_parquet(processed_root / "labels.parquet"),
        pd.read_parquet(processed_root / "splits.parquet"),
        pd.read_csv(processed_root / "homology_clusters.csv"),
        data_version,
    )
    table.to_csv(Path(output_root) / "位点_PU_同源簇_特征来源_数据版本总表.csv", index=False, encoding="utf-8-sig")
    return table


def run(config, output_root, seeds, epochs, patience, batch_size, device, network_only=False, bootstrap_repeats=1000, resume=False):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=resume)
    split_path = config.processed_data / "splits.parquet"
    edge_rows, detail_frames = [], []
    labels, groups = _labels_and_groups(config.processed_data)
    for variant in NETWORK_VARIANTS:
        paths = output_paths(output_root, variant)
        audit_path = paths.network_dir / "network_variant_audit.json"
        if resume and audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        else:
            audit = build_network_variant(config, split_path, variant, paths.network_dir)
        edge_rows.append({"网络版本": VARIANT_NAMES[variant], **audit})
        if network_only:
            continue
        train_from_config(
            config, seeds=seeds, epochs=epochs, batch_size=batch_size, objective="nnpu", device=device,
            output_root=paths.training_dir, patience=patience, network_dir=paths.network_dir,
            network_variant=variant, network_enabled=variant != "without_network",
        )
        detail_frames.append(_metrics_tables(paths, labels, groups, seeds, bootstrap_repeats))
    pd.DataFrame(edge_rows).to_csv(output_root / "网络版本边数审计.csv", index=False, encoding="utf-8-sig")
    write_random_baseline_report(labels[groups["validation"]], labels[groups["test"]], output_root)
    if detail_frames:
        _write_reports(output_root, edge_rows, detail_frames)
    (output_root / "网络证据泄漏审计说明.md").write_text(
        "# 网络证据泄漏审计\n\n0 表示未标注 U，不是真负例。直接证据屏蔽仅删除 validation/test 位点精确匹配且 DIRECT=t 的 SIGNOR 原始记录。\n",
        encoding="utf-8",
    )
    return {"output_root": str(output_root), "network_only": bool(network_only), "variants": list(VARIANT_NAMES)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run PhosMMPU network evidence sensitivity study")
    parser.add_argument("--config", required=True)
    parser.add_argument("--processed-dir")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 51, 73])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--network-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    args = parser.parse_args(argv)
    config = MultimodalConfig.load(args.config)
    if args.processed_dir:
        config = replace(config, processed_data=Path(args.processed_dir).resolve())
    print(json.dumps(run(config, args.output_root, args.seeds, args.epochs, args.patience, args.batch_size, args.device, args.network_only, args.bootstrap_repeats, args.resume), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
