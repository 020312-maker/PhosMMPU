"""Merge all same-split model summaries into one auditable comparison table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_summary_files(summary_files):
    """Merge ``(csv_path, category)`` sources while retaining metric provenance."""
    frames = []
    for path, category in summary_files:
        frame = pd.read_csv(path)
        if "模型" not in frame:
            if "网络版本" not in frame:
                raise ValueError(f"summary lacks model identity: {path}")
            frame["模型"] = "PhosMMPU（" + frame["网络版本"].astype(str) + "）"
        frame["实验类别"] = category
        frames.append(frame)
    if not frames:
        raise ValueError("at least one summary file is required")
    return pd.concat(frames, ignore_index=True, sort=False)


def build_final_comparison(network_summary, baseline_root, ablation_root, lightgbm_summary, output_file):
    """Write a combined result table for the fixed three-task homology split."""
    baseline_root, ablation_root = Path(baseline_root), Path(ablation_root)
    sources = [(Path(network_summary), "网络证据敏感性")]
    sources.extend((path, "baseline") for path in sorted(baseline_root.glob("*/五种子测试指标汇总.csv")))
    sources.extend((path, "模型消融") for path in sorted(ablation_root.glob("*/五种子测试指标汇总.csv")))
    sources.append((Path(lightgbm_summary), "baseline"))
    merged = merge_summary_files(sources)
    columns = [
        "实验类别", "模型", "任务", "task", "指标", "metric", "五种子集成点估计", "五种子均值", "五种子标准差", "种子数",
        "Bootstrap下限", "Bootstrap上限", "Bootstrap有效重复数", "requested_repeats",
    ]
    selected = [column for column in columns if column in merged]
    merged = merged[selected].sort_values(["实验类别", "模型", "task", "metric"], kind="mergesort").reset_index(drop=True)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_file, index=False, encoding="utf-8-sig")
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser(description="Merge PhosMMPU comparison summaries")
    parser.add_argument("--network-summary", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--lightgbm-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build_final_comparison(
        args.network_summary, args.baseline_root, args.ablation_root, args.lightgbm_summary, args.output,
    )
    print(f"rows={len(result)} output={args.output}")


if __name__ == "__main__":
    main()
