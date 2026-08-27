"""Post-hoc fixed-test subgroup analysis for saved PhosMMPU predictions."""

from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from src.multimodal.pu_evaluation import METRICS, evaluate_test, validation_f1_thresholds


MIN_SITES = 20
MIN_POSITIVES = 3
MIN_UNLABELED = 3
TASKS = ("activity", "interaction", "proteostasis")
TASK_NAMES = {"activity": "活性调控", "interaction": "分子关联", "proteostasis": "稳定性/降解"}


def _validate_vectors(labels, scores, groups):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups, dtype=str)
    if labels.ndim != 1 or scores.ndim != 1 or groups.ndim != 1:
        raise ValueError("labels, scores, and groups must be one-dimensional")
    if not (len(labels) == len(scores) == len(groups)):
        raise ValueError("labels, scores, and groups must have equal length")
    if not set(np.unique(labels)).issubset({0, 1}) or not np.isfinite(scores).all():
        raise ValueError("labels must be binary and scores finite")
    return labels, scores, groups


def _one_task_metrics(labels, scores, threshold):
    repeated_labels = np.repeat(labels[:, None], 3, axis=1)
    repeated_scores = np.repeat(scores[:, None], 3, axis=1)
    return evaluate_test(repeated_labels, repeated_scores, [threshold] * 3)["activity"]


def summarize_groups(labels, scores, groups, threshold, task, level):
    """Summarize observed P/U classification metrics for mutually labelled groups.

    Metrics are only evaluated for groups with enough observed P and U sites.
    Every group is retained so readers can distinguish unavailable evidence from
    a zero result.
    """
    labels, scores, groups = _validate_vectors(labels, scores, groups)
    rows = []
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        group_labels, group_scores = labels[positions], scores[positions]
        positive_count = int(group_labels.sum())
        unlabeled_count = int(len(group_labels) - positive_count)
        eligible = bool(
            len(group_labels) >= MIN_SITES
            and positive_count >= MIN_POSITIVES
            and unlabeled_count >= MIN_UNLABELED
        )
        row = {
            "level": str(level),
            "subgroup": str(group),
            "task": str(task),
            "site_count": int(len(group_labels)),
            "positive_count": positive_count,
            "unlabeled_count": unlabeled_count,
            "observed_positive_ratio": float(group_labels.mean()) if len(group_labels) else float("nan"),
            "mean_prediction_score": float(group_scores.mean()) if len(group_scores) else float("nan"),
            "threshold": float(threshold),
            "eligible_for_metrics": eligible,
            "ineligible_reason": "" if eligible else "样本不足或单一标签",
        }
        if eligible:
            row.update({metric: _one_task_metrics(group_labels, group_scores, threshold)[metric] for metric in METRICS})
        else:
            row.update({metric: float("nan") for metric in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def load_saved_seed_ensemble(processed_dir, run_root, seeds, objective="nnpu"):
    """Load one fixed split and validate then average saved seed predictions."""
    processed_dir, run_root = Path(processed_dir), Path(run_root)
    labels = np.load(processed_dir / "labels.npy").astype(np.int8)
    split_names = np.load(processed_dir / "split_names.npy", allow_pickle=False).astype(str)
    site_ids = np.load(processed_dir / "dataset_site_ids.npy", allow_pickle=False).astype(str)
    if not (len(labels) == len(split_names) == len(site_ids)):
        raise ValueError("processed labels, split names, and site IDs must align")
    validation_positions = np.flatnonzero(split_names == "validation")
    test_positions = np.flatnonzero(split_names == "test")
    validation_scores, test_scores = [], []
    expected_test_ids, expected_test_labels = site_ids[test_positions], labels[test_positions]
    probability_columns = ["activity_probability", "interaction_probability", "proteostasis_probability"]
    label_columns = ["y_activity", "y_interaction", "y_proteostasis"]
    for seed in seeds:
        seed_dir = run_root / f"{objective}_seed{int(seed)}"
        validation = np.load(seed_dir / "validation_probabilities.npy")
        test = np.load(seed_dir / "test_probabilities.npy")
        if validation.shape != (len(validation_positions), 3) or test.shape != (len(test_positions), 3):
            raise ValueError(f"seed {seed} probability shape does not match fixed split")
        frame = pd.read_parquet(seed_dir / "test_predictions.parquet")
        actual_ids = frame["site_id"].astype(str).to_numpy()
        actual_labels = frame[label_columns].to_numpy(dtype=np.int8)
        actual_scores = frame[probability_columns].to_numpy(dtype=float)
        if not np.array_equal(actual_ids, expected_test_ids):
            raise ValueError(f"seed {seed} test site_id order does not match fixed split")
        if not np.array_equal(actual_labels, expected_test_labels):
            raise ValueError(f"seed {seed} test labels do not match fixed split")
        if not np.allclose(actual_scores, test, rtol=0.0, atol=1e-12):
            raise ValueError(f"seed {seed} test parquet probabilities do not match NumPy output")
        validation_scores.append(validation.astype(float))
        test_scores.append(test.astype(float))
    if not validation_scores:
        raise ValueError("at least one saved seed is required")
    return {
        "validation_labels": labels[validation_positions],
        "validation_scores": np.stack(validation_scores).mean(axis=0),
        "test_labels": expected_test_labels,
        "test_scores": np.stack(test_scores).mean(axis=0),
        "test_site_ids": expected_test_ids,
        "seeds": tuple(int(seed) for seed in seeds),
    }


def ensemble_validation_thresholds(validation_labels, validation_scores):
    """Select the three fixed decision thresholds from ensemble validation scores."""
    return validation_f1_thresholds(validation_labels, validation_scores)


def bootstrap_group_metrics(labels, scores, threshold, repeats=1000, seed=20260811):
    """Use within-group observed P/U stratified resampling for metric intervals."""
    labels, scores, _ = _validate_vectors(labels, scores, np.repeat("group", len(labels)))
    if len(labels) < 2 or labels.min() == labels.max():
        raise ValueError("Bootstrap requires both observed P and U sites")
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("Bootstrap repeats must be positive")
    generator = np.random.default_rng(seed)
    positive_positions = np.flatnonzero(labels == 1)
    unlabeled_positions = np.flatnonzero(labels == 0)
    samples = {metric: [] for metric in METRICS}
    for _ in range(repeats):
        positions = np.concatenate((
            generator.choice(positive_positions, size=len(positive_positions), replace=True),
            generator.choice(unlabeled_positions, size=len(unlabeled_positions), replace=True),
        ))
        metrics = _one_task_metrics(labels[positions], scores[positions], threshold)
        for metric in METRICS:
            samples[metric].append(metrics[metric])
    point = _one_task_metrics(labels, scores, threshold)
    result = {
        "bootstrap_unit": "subgroup_test_sites_stratified_by_observed_pu",
        "bootstrap_repeats": repeats,
    }
    for metric in METRICS:
        values = np.asarray(samples[metric], dtype=float)
        result[metric] = point[metric]
        result[f"{metric}_ci_low"] = float(np.quantile(values, 0.025))
        result[f"{metric}_ci_high"] = float(np.quantile(values, 0.975))
    return result


def parse_uniprot_family_tsv(tsv_text, requested_accessions):
    """Parse UniProt's returned family annotation without reclassifying it."""
    requested = [str(accession) for accession in requested_accessions]
    returned = pd.read_csv(StringIO(tsv_text), sep="\t", dtype=str).fillna("")
    required = {"Entry", "Protein families"}
    if required.difference(returned.columns):
        raise ValueError("UniProt TSV must contain Entry and Protein families columns")
    lookup = returned.set_index("Entry")["Protein families"].to_dict()
    rows = []
    for accession in requested:
        raw = str(lookup.get(accession, "")).strip()
        rows.append({
            "accession": accession,
            "protein_family": raw if raw else "未注释",
            "returned_by_uniprot": accession in lookup,
        })
    return pd.DataFrame(rows)


def download_uniprot_family_annotations(accessions, batch_size=100, timeout=60, opener=urlopen):
    """Download UniProt protein_families annotations in deterministic accession batches."""
    ordered = tuple(dict.fromkeys(map(str, accessions)))
    if not ordered:
        return pd.DataFrame(columns=["accession", "protein_family", "returned_by_uniprot"]), {
            "source": "https://rest.uniprot.org/uniprotkb/search",
            "download_date": date.today().isoformat(), "request_count": 0,
        }
    batches = [ordered[start:start + int(batch_size)] for start in range(0, len(ordered), int(batch_size))]
    frames, urls, releases, release_dates = [], [], [], []
    for batch in batches:
        query = " OR ".join(f"accession:{accession}" for accession in batch)
        parameters = urlencode({"query": f"({query})", "format": "tsv", "fields": "accession,protein_families", "size": len(batch)})
        url = f"https://rest.uniprot.org/uniprotkb/search?{parameters}"
        request = Request(url, headers={"User-Agent": "PhosMMPU-subgroup-analysis/1.0"})
        with opener(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
            headers = response.headers
        frames.append(parse_uniprot_family_tsv(text, batch))
        urls.append(url)
        releases.append(headers.get("x-uniprot-release", ""))
        release_dates.append(headers.get("x-uniprot-release-date", ""))
    annotations = pd.concat(frames, ignore_index=True)
    audit = {
        "source": "https://rest.uniprot.org/uniprotkb/search",
        "download_date": date.today().isoformat(),
        "request_count": len(batches),
        "request_urls": urls,
        "uniprot_releases": sorted({value for value in releases if value}),
        "uniprot_release_dates": sorted({value for value in release_dates if value}),
        "requested_accessions": len(ordered),
        "returned_accessions": int(annotations["returned_by_uniprot"].sum()),
        "annotated_accessions": int(annotations["protein_family"].ne("未注释").sum()),
    }
    return annotations, audit


def _residue_summary(labels, scores, residues, thresholds, bootstrap_repeats):
    frames = []
    for task_index, task in enumerate(TASKS):
        summary = summarize_groups(labels[:, task_index], scores[:, task_index], residues, thresholds[task_index], task, "residue")
        for row_index, row in summary.loc[summary["eligible_for_metrics"]].iterrows():
            positions = np.flatnonzero(residues == row["subgroup"])
            interval = bootstrap_group_metrics(
                labels[positions, task_index], scores[positions, task_index], thresholds[task_index],
                repeats=bootstrap_repeats, seed=20260811 + task_index,
            )
            for metric in METRICS:
                summary.loc[row_index, f"{metric}_ci_low"] = interval[f"{metric}_ci_low"]
                summary.loc[row_index, f"{metric}_ci_high"] = interval[f"{metric}_ci_high"]
            summary.loc[row_index, "bootstrap_repeats"] = interval["bootstrap_repeats"]
            summary.loc[row_index, "bootstrap_unit"] = interval["bootstrap_unit"]
        frames.append(summary)
    return pd.concat(frames, ignore_index=True).sort_values(["task", "subgroup"], kind="mergesort").reset_index(drop=True)


def _level_summary(labels, scores, groups, thresholds, level):
    frames = [
        summarize_groups(labels[:, task_index], scores[:, task_index], groups, thresholds[task_index], task, level)
        for task_index, task in enumerate(TASKS)
    ]
    return pd.concat(frames, ignore_index=True).sort_values(["task", "subgroup"], kind="mergesort").reset_index(drop=True)


def build_subgroup_tables(
    test_site_index, test_labels, test_scores, thresholds, family_annotations, homology_clusters,
    bootstrap_repeats=1000,
):
    """Build residual, protein, UniProt-family and MMseqs-cluster analysis tables."""
    index = test_site_index.copy().reset_index(drop=True)
    test_labels = np.asarray(test_labels, dtype=np.int8)
    test_scores = np.asarray(test_scores, dtype=float)
    required = {"site_id", "accession", "gene", "residue"}
    if required.difference(index.columns):
        raise ValueError(f"test site index is missing columns: {sorted(required.difference(index.columns))}")
    if test_labels.shape != test_scores.shape or test_labels.shape != (len(index), 3):
        raise ValueError("test site index, labels, and scores must align as (rows, 3)")
    if len(thresholds) != 3:
        raise ValueError("three validation-derived thresholds are required")
    family = family_annotations.copy()
    if {"accession", "protein_family"}.difference(family.columns):
        raise ValueError("family annotations must contain accession and protein_family")
    family = family.drop_duplicates("accession", keep="last")
    clusters = homology_clusters.copy()
    if {"accession", "cluster_id"}.difference(clusters.columns):
        raise ValueError("homology clusters must contain accession and cluster_id")
    clusters = clusters.drop_duplicates("accession", keep="last")
    indexed = index.merge(family[["accession", "protein_family"]], on="accession", how="left", validate="many_to_one")
    indexed = indexed.merge(clusters[["accession", "cluster_id"]], on="accession", how="left", validate="many_to_one")
    indexed["protein_family"] = indexed["protein_family"].fillna("未注释").replace("", "未注释")
    indexed["cluster_id"] = indexed["cluster_id"].fillna("未分配同源簇").replace("", "未分配同源簇")
    audit = family.copy()
    audit["protein_family"] = audit["protein_family"].fillna("未注释").replace("", "未注释")
    site_counts = indexed.groupby("accession", as_index=False).size().rename(columns={"size": "test_site_count"})
    audit = audit.merge(site_counts, on="accession", how="left", validate="one_to_one")
    audit["test_site_count"] = audit["test_site_count"].fillna(0).astype(int)
    return {
        "residue": _residue_summary(test_labels, test_scores, indexed["residue"].astype(str).to_numpy(), thresholds, bootstrap_repeats),
        "protein": _level_summary(test_labels, test_scores, indexed["accession"].astype(str).to_numpy(), thresholds, "protein"),
        "family": _level_summary(test_labels, test_scores, indexed["protein_family"].astype(str).to_numpy(), thresholds, "uniprot_family"),
        "homology": _level_summary(test_labels, test_scores, indexed["cluster_id"].astype(str).to_numpy(), thresholds, "mmseqs_homology_cluster"),
        "family_audit": audit.sort_values("accession", kind="mergesort").reset_index(drop=True),
    }


def _chinese_table(frame):
    output = frame.copy()
    output["任务"] = output["task"].map(TASK_NAMES)
    output["亚组"] = output["subgroup"]
    rename = {
        "level": "分析层级", "site_count": "测试位点数", "positive_count": "已知正例数",
        "unlabeled_count": "未标注位点数", "observed_positive_ratio": "观测正例比例",
        "mean_prediction_score": "平均预测分数", "threshold": "验证集固定阈值",
        "eligible_for_metrics": "纳入分类指标", "ineligible_reason": "未纳入原因",
        "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1", "mcc": "MCC",
        "precision": "Precision", "recall": "Recall", "bootstrap_repeats": "Bootstrap重复次数",
        "bootstrap_unit": "Bootstrap单位",
    }
    for metric in METRICS:
        rename[f"{metric}_ci_low"] = f"{rename[metric]} 95%CI下限"
        rename[f"{metric}_ci_high"] = f"{rename[metric]} 95%CI上限"
    return output.rename(columns=rename)


def write_subgroup_tables(tables, output_dir):
    """Write the five specified Chinese-named CSV tables."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "residue": "残基亚组测试集指标.csv",
        "protein": "蛋白亚组测试集汇总.csv",
        "family": "UniProt蛋白家族亚组测试集汇总.csv",
        "homology": "MMseqs同源簇亚组测试集汇总.csv",
        "family_audit": "UniProt蛋白家族注释审计.csv",
    }
    paths = {}
    for key, name in names.items():
        frame = tables[key]
        if key == "family_audit":
            frame = frame.rename(columns={"accession": "UniProt accession", "protein_family": "UniProt 蛋白家族", "returned_by_uniprot": "UniProt返回记录", "test_site_count": "测试位点数"})
        else:
            frame = _chinese_table(frame)
        path = output_dir / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    return paths


def _prepare_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def format_family_plot_label(value, limit=42):
    """Shorten only plot labels; CSV tables retain the complete UniProt value."""
    value = str(value)
    return value if len(value) <= int(limit) else f"{value[:int(limit) - 3]}..."


def render_subgroup_figures(tables, output_dir):
    """Render the two planned Chinese subgroup comparison figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt = _prepare_matplotlib()
    paths = {}

    residue = tables["residue"].copy()
    residues = [item for item in ("S", "T", "Y") if item in set(residue["subgroup"])]
    tasks = [task for task in TASKS if task in set(residue["task"])]
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    positions = np.arange(len(residues), dtype=float)
    width = 0.75 / max(len(tasks), 1)
    for task_index, task in enumerate(tasks):
        data = residue.loc[residue["task"].eq(task)].set_index("subgroup").reindex(residues)
        values = data["pr_auc"].to_numpy(dtype=float)
        lower = data.get("pr_auc_ci_low", pd.Series(np.nan, index=data.index)).to_numpy(dtype=float)
        upper = data.get("pr_auc_ci_high", pd.Series(np.nan, index=data.index)).to_numpy(dtype=float)
        offset = (task_index - (len(tasks) - 1) / 2) * width
        errors = np.vstack((np.maximum(values - lower, 0), np.maximum(upper - values, 0)))
        errors[:, ~np.isfinite(errors).all(axis=0)] = 0
        axis.bar(positions + offset, values, width=width, label=TASK_NAMES[task], yerr=errors, capsize=3)
    axis.set(title="残基亚组：测试集 PR-AUC", xlabel="中心磷酸化残基", ylabel="PR-AUC", xticks=positions, xticklabels=residues, ylim=(0, 1))
    axis.legend()
    figure.tight_layout()
    paths["residue_pr_auc"] = output_dir / "图_残基亚组_PR-AUC.png"
    figure.savefig(paths["residue_pr_auc"], dpi=220, bbox_inches="tight")
    plt.close(figure)

    family = tables["family"].copy()
    figure, axes = plt.subplots(1, len(TASKS), figsize=(25, 7), sharex=True, layout="constrained")
    for axis, task in zip(axes, TASKS):
        eligible = family.loc[family["task"].eq(task) & family["eligible_for_metrics"].astype(bool)]
        eligible = eligible.sort_values(["site_count", "subgroup"], ascending=[False, True], kind="mergesort").head(10)
        if eligible.empty:
            axis.text(0.5, 0.5, "没有满足指标\n纳入条件的蛋白家族", ha="center", va="center")
            axis.set_axis_off()
            continue
        axis.barh(np.arange(len(eligible)), eligible["pr_auc"].to_numpy(dtype=float), color="#3a8fb7")
        axis.set(
            title=TASK_NAMES[task], xlabel="PR-AUC", xlim=(0, 1), yticks=np.arange(len(eligible)),
            yticklabels=[format_family_plot_label(value) for value in eligible["subgroup"]],
        )
    figure.suptitle("主要 UniProt 蛋白家族：测试集 PR-AUC", fontsize=14)
    paths["family_pr_auc"] = output_dir / "图_主要蛋白家族_PR-AUC.png"
    figure.savefig(paths["family_pr_auc"], dpi=220, bbox_inches="tight")
    plt.close(figure)
    return paths


def write_subgroup_readme(output_dir, uniprot_audit):
    """Write the statistical scope alongside the post-hoc subgroup artefacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text = (
        "# PhosMMPU 测试集亚组分析口径\n\n"
        "- 预测来源：完整网络 PhosMMPU 固定测试集的五种子概率均值；本目录不训练模型。\n"
        "- 阈值来源：每项任务仅在固定验证集的五种子均值上选择最大 F1 阈值，随后固定应用到全部测试亚组。\n"
        "- 标签解释：P 为已有功能证据，U 为未标注而不是确认阴性；指标仅衡量已知 P 相对于 U 的观测区分。\n"
        "- 残基表：S/T/Y 使用组内按观测 P/U 分层的测试位点 Bootstrap（1,000 次）报告 95% CI。\n"
        "- 蛋白、UniProt 家族、MMseqs 序列同源簇：仅在每任务 n>=20、P>=3、U>=3 时报告分类指标；其余行保留描述性统计。\n"
        "- UniProt 家族字段只在预测完成后用于分组，不进入模型、损失函数、阈值选择或训练。\n"
        "- MMseqs 簇使用既有 40% 序列一致性和 70% 覆盖度定义，仅称为序列同源簇，不等同于 UniProt 生物学家族。\n\n"
        f"UniProt REST 请求批次数：{uniprot_audit.get('request_count', 0)}；有家族注释的 accession 数：{uniprot_audit.get('annotated_accessions', 0)}。\n"
    )
    path = output_dir / "亚组分析口径说明.md"
    path.write_text(text, encoding="utf-8")
    return path
