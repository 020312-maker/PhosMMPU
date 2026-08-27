"""Export held-out candidates whose three functional labels are all zero.

This is a candidate-discovery export: a zero denotes unlabelled for the PU
tasks, not a verified negative.  Ranking uses the fixed client-facing balance
of 0.40 activity, 0.35 interaction, and 0.25 proteostasis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd


SEEDS = (11, 23, 37, 51, 73)
SCORE_COLUMNS = ("activity_probability", "interaction_probability", "proteostasis_probability")
WEIGHTS = np.asarray((0.40, 0.35, 0.25), dtype=np.float64)


def build_triple_zero_top_candidates(prediction_root: Path, site_index_path: Path, top_k: int) -> pd.DataFrame:
    """Return the Top-K held-out sites with all three observed labels equal to zero."""
    prediction_root = Path(prediction_root)
    seed_tables = [
        pd.read_parquet(prediction_root / f"nnpu_seed{seed}" / "test_predictions.parquet")
        for seed in SEEDS
    ]
    first = seed_tables[0]
    site_ids = first["site_id"].astype(str).to_numpy()
    labels = first[["y_activity", "y_interaction", "y_proteostasis"]].to_numpy(dtype=np.int8)
    for seed, table in zip(SEEDS[1:], seed_tables[1:], strict=True):
        if not np.array_equal(table["site_id"].astype(str).to_numpy(), site_ids):
            raise ValueError(f"seed {seed} test-site order differs from seed {SEEDS[0]}")
        if not np.array_equal(table[["y_activity", "y_interaction", "y_proteostasis"]].to_numpy(dtype=np.int8), labels):
            raise ValueError(f"seed {seed} test labels differ from seed {SEEDS[0]}")

    scores = np.stack([table.loc[:, SCORE_COLUMNS].to_numpy(dtype=np.float64) for table in seed_tables])
    score_mean = scores.mean(axis=0)
    score_std = scores.std(axis=0, ddof=1)
    all_zero = (labels == 0).all(axis=1)
    if int(all_zero.sum()) < top_k:
        raise ValueError("triple-zero held-out candidates are fewer than top_k")

    site_index = pd.read_parquet(site_index_path)
    metadata = site_index.loc[:, ["site_id", "accession", "gene", "residue", "position"]].copy()
    result = pd.DataFrame(
        {
            "位点 ID": site_ids,
            "活性调控已知标签": labels[:, 0],
            "相互作用调控已知标签": labels[:, 1],
            "稳定性/降解已知标签": labels[:, 2],
            "活性调控预测分数": score_mean[:, 0],
            "活性调控预测分数标准差": score_std[:, 0],
            "分子关联预测分数": score_mean[:, 1],
            "分子关联预测分数标准差": score_std[:, 1],
            "稳定性/降解预测分数": score_mean[:, 2],
            "稳定性/降解预测分数标准差": score_std[:, 2],
        }
    )
    result["加权综合预测分数"] = score_mean @ WEIGHTS
    result = result.loc[all_zero].merge(metadata, left_on="位点 ID", right_on="site_id", how="left", validate="one_to_one")
    if result[["accession", "gene", "residue", "position"]].isna().any().any():
        raise ValueError("some candidates lack site-index metadata")
    result = result.sort_values(["加权综合预测分数", "位点 ID"], ascending=[False, True], kind="mergesort").head(top_k).copy()
    result.insert(0, "候选排名", np.arange(1, len(result) + 1))
    result.insert(1, "预测划分", "同源簇独立测试集")
    result.insert(2, "标签状态", "U/U/U（三项均未标注，非负例）")
    result.insert(3, "随机种子数", len(SEEDS))
    result.rename(columns={"accession": "UniProt 蛋白", "gene": "基因", "residue": "残基", "position": "位置"}, inplace=True)
    result.drop(columns="site_id", inplace=True)
    result["PubMed 精确检索链接"] = [
        "https://pubmed.ncbi.nlm.nih.gov/?term="
        + quote_plus(f'"{row.基因}" AND ("{row.残基}{row.位置}" OR "{row.残基} {row.位置}") AND phosphorylation')
        for row in result.itertuples(index=False)
    ]
    result["外部 PMID"] = ""
    result["外部文献日期"] = ""
    result["外部证据等级"] = "待检索"
    result["精确位点磷酸化证据"] = "待检索"
    result["功能关联证据"] = "待检索"
    result["核查结论"] = "待检索"
    return result


def export_triple_zero_candidates(prediction_root: Path, site_index_path: Path, output_root: Path, top_k: int) -> dict[str, object]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    candidates = build_triple_zero_top_candidates(prediction_root, site_index_path, top_k)
    candidates.to_csv(output_root / f"同源簇测试集_三标签全0_Top{top_k}_文献核查表.csv", index=False, encoding="utf-8-sig")
    candidates.to_excel(output_root / f"同源簇测试集_三标签全0_Top{top_k}_文献核查表.xlsx", index=False)
    summary = {
        "candidate_definition": "held-out test sites with activity=0, interaction=0, proteostasis=0",
        "pu_interpretation": "all-zero means unlabelled across three tasks, not verified negative",
        "ranking": {"activity": 0.40, "interaction": 0.35, "proteostasis": 0.25},
        "seeds": list(SEEDS),
        "top_k": int(top_k),
        "model": "PhosMMPU full multimodal, nnPU, modality_dropout=0.0, complete network",
        "literature_boundary": "PubMed results must be classified as exact-site, protein-level only, or no matching evidence; this export alone is not independent time-split validation.",
    }
    (output_root / "实验与文献核查说明.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"top_k": int(top_k), "output_root": str(output_root)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export Top-K triple-zero held-out phosphosite candidates.")
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--site-index", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--top-k", default=30, type=int)
    args = parser.parse_args(argv)
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    print(json.dumps(
        export_triple_zero_candidates(
            prediction_root=args.prediction_root,
            site_index_path=args.site_index,
            output_root=args.output_root,
            top_k=args.top_k,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
