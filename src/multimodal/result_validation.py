"""Evaluate five-seed held-out predictions without treating PU labels as negatives."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.multimodal.client_ranking import WEIGHTS


LABEL_COLUMNS = ("y_activity", "y_interaction", "y_proteostasis")
SCORE_COLUMNS = (
    "activity_probability",
    "interaction_probability",
    "proteostasis_probability",
)
TASKS = (
    ("activity", "活性调控", "y_activity", "activity_probability"),
    ("interaction", "相互作用调控", "y_interaction", "interaction_probability"),
    ("proteostasis", "稳定性/降解调控", "y_proteostasis", "proteostasis_probability"),
)
WORKPOINT_RANKS = (10, 50, 100, 500, 1000, 5000)
FIGURE_STEMS = {
    "activity": "活性调控_验证曲线",
    "interaction": "相互作用调控_验证曲线",
    "proteostasis": "稳定性与降解调控_验证曲线",
    "balanced_composite": "综合排序_验证曲线",
    "auc_comparison": "四项指标_AUC对比",
}


def _required_columns(frame: pd.DataFrame) -> None:
    required = {"site_id", *LABEL_COLUMNS, *SCORE_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"prediction frame is missing columns: {sorted(missing)}")


def _arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _required_columns(frame)
    if frame.empty:
        raise ValueError("prediction frame must not be empty")
    site_ids = frame["site_id"].astype(str).to_numpy()
    if len(set(site_ids)) != len(site_ids):
        raise ValueError("site_id values must be unique within every seed prediction")
    labels = frame.loc[:, LABEL_COLUMNS].to_numpy(dtype=np.int8)
    scores = frame.loc[:, SCORE_COLUMNS].to_numpy(dtype=np.float64)
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be binary")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("prediction scores must be finite probabilities in [0, 1]")
    return site_ids, labels, scores


def _validated_seed_arrays(
    seed_frames: list[pd.DataFrame], seeds: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not seed_frames or len(seed_frames) != len(seeds):
        raise ValueError("seed_frames and seeds must be non-empty and have matching length")
    site_ids, labels, scores = _arrays(seed_frames[0])
    seed_scores = [scores]
    for seed, frame in zip(seeds[1:], seed_frames[1:]):
        next_site_ids, next_labels, next_scores = _arrays(frame)
        if not np.array_equal(site_ids, next_site_ids):
            raise ValueError(f"site_id order differs for seed {int(seed)}")
        if not np.array_equal(labels, next_labels):
            raise ValueError(f"labels differ for seed {int(seed)}")
        seed_scores.append(next_scores)
    return site_ids, labels, np.stack(seed_scores, axis=0)


def _task_arrays(labels: np.ndarray, scores: np.ndarray):
    for index, (task, task_label, label_column, score_column) in enumerate(TASKS):
        yield task, task_label, label_column, score_column, labels[:, index], scores[:, index]
    yield (
        "balanced_composite",
        "综合排序（任一已知功能）",
        "any_known_function",
        "balanced_confidence",
        labels.any(axis=1).astype(np.int8),
        scores.dot(WEIGHTS),
    )


def summarize_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    if len(labels) < 2 or labels.min() == labels.max():
        raise ValueError("each metric requires both known-positive and unlabeled rows")
    positive_count = int(labels.sum())
    return {
        "total_count": int(len(labels)),
        "positive_count": positive_count,
        "positive_prevalence": float(positive_count / len(labels)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
    }


def observed_fdr_tpr_curve(labels, scores) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("observed FDR curve requires at least one known-positive row")
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    selected = np.arange(1, len(labels) + 1)
    true_positive = np.cumsum(ordered_labels)
    last_at_score = np.r_[ordered_scores[:-1] != ordered_scores[1:], True]
    selected = selected[last_at_score]
    true_positive = true_positive[last_at_score]
    thresholds = ordered_scores[last_at_score]
    precision = true_positive / selected
    return pd.DataFrame(
        {
            "threshold_score": thresholds,
            "selected_count": selected.astype(int),
            "known_positive_count": true_positive.astype(int),
            "unlabeled_count": (selected - true_positive).astype(int),
            "tpr": true_positive / positives,
            "observed_precision": precision,
            "observed_fdr": 1.0 - precision,
        }
    )


def threshold_workpoints(labels, scores, ranks=WORKPOINT_RANKS) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("labels and scores must be aligned one-dimensional arrays")
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("threshold workpoints require at least one known-positive row")
    order = np.argsort(-scores, kind="mergesort")
    rows = []
    for rank in ranks:
        count = min(int(rank), len(order))
        selected_labels = labels[order[:count]]
        true_positive = int(selected_labels.sum())
        precision = true_positive / count
        rows.append(
            {
                "requested_rank": int(rank),
                "threshold_score": float(scores[order[count - 1]]),
                "selected_count": count,
                "known_positive_count": true_positive,
                "unlabeled_count": count - true_positive,
                "tpr": true_positive / positives,
                "observed_precision": precision,
                "observed_fdr": 1.0 - precision,
                "selection_rule": "top_k_stable_rank",
            }
        )
    return pd.DataFrame(rows)


def _metric_frame(labels: np.ndarray, scores: np.ndarray, seed: int | str) -> pd.DataFrame:
    rows = []
    for task, task_label, label_column, score_column, task_labels, task_scores in _task_arrays(labels, scores):
        rows.append(
            {
                "seed": seed,
                "task": task,
                "task_label": task_label,
                "label_column": label_column,
                "score_column": score_column,
                **summarize_metrics(task_labels, task_scores),
            }
        )
    return pd.DataFrame(rows)


def build_evaluation_frames(
    seed_frames: list[pd.DataFrame], seeds: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, labels, seed_scores = _validated_seed_arrays(seed_frames, seeds)
    ensemble = _metric_frame(labels, seed_scores.mean(axis=0), "ensemble")
    per_seed = pd.concat(
        [_metric_frame(labels, scores, int(seed)) for seed, scores in zip(seeds, seed_scores)],
        ignore_index=True,
    )
    return ensemble, per_seed


def _load_seed_frames(run_root: Path, expected_seeds: list[int]) -> list[pd.DataFrame]:
    frames = []
    for seed in expected_seeds:
        path = run_root / f"nnpu_seed{int(seed)}" / "test_predictions.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing seed prediction: {path}")
        frames.append(pd.read_parquet(path))
    return frames


def _plot_task(task_row, labels, scores, output_path: Path, test_count: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    fdr_curve = observed_fdr_tpr_curve(labels, scores)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    title = f"{task_row['task_label']}：独立测试集（n={test_count}）"
    axes[0].plot(fpr, tpr, color="#1677b8", linewidth=2, label=f"ROC-AUC = {task_row['roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], color="#808080", linestyle="--", linewidth=1)
    axes[0].set(xlabel="未标注位点比例", ylabel="TPR（已知正例召回率）", title="ROC 曲线", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(loc="lower right")
    axes[1].plot(recall, precision, color="#2c9c69", linewidth=2, label=f"PR-AUC = {task_row['pr_auc']:.3f}")
    axes[1].axhline(task_row["positive_prevalence"], color="#808080", linestyle="--", linewidth=1, label="已知正例比例")
    axes[1].set(xlabel="TPR（已知正例召回率）", ylabel="已知标签精确率", title="PR 曲线", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend(loc="upper right")
    axes[2].plot(fdr_curve["observed_fdr"], fdr_curve["tpr"], color="#b65a34", linewidth=2)
    axes[2].set(xlabel="观测 FDR（未标注暂作非正例）", ylabel="TPR（已知正例召回率）", title="阈值覆盖曲线", xlim=(0, 1), ylim=(0, 1))
    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _plot_auc_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = metrics["task_label"].tolist()
    positions = np.arange(len(metrics))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for axis, column, title in zip(
        axes,
        ("roc_auc", "pr_auc", "positive_prevalence"),
        ("ROC-AUC", "PR-AUC", "已知正例比例"),
    ):
        values = metrics[column].to_numpy(dtype=float)
        bars = axis.bar(positions, values, color="#1677b8")
        axis.set(title=title, ylim=(0, 1), xticks=positions, xticklabels=labels)
        axis.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    figure.suptitle("四项排序视图：独立测试集指标比较", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def _readme(seed_count: int, test_count: int) -> str:
    return f"""# 五种子保持测试集验证结果

- 评估划分：`held_out_test`，共 {test_count} 个位点。
- 最终分数：{seed_count} 个随机种子原始 sigmoid 输出的算术平均。
- 单任务标签：FuncPhos-SEQ 汇总注释中对应功能的已有公开证据。
- 综合分数：`0.40 * 活性 + 0.35 * 相互作用 + 0.25 * 稳定性/降解`；综合已知正例表示至少一种功能已有注释。
- `0` 表示未标注，而非确认阴性。因此图中的“观测 FDR”将未标注暂作非正例，只是保守的标签覆盖指标，不是真实生物学 FDR。
- ROC-AUC 和 PR-AUC 衡量已知正例相对未标注位点的排序能力，不可解释为外部临床诊断性能或湿实验因果证明。
"""


def generate_validation_report(run_root, expected_seeds, output_dir) -> Path:
    run_root = Path(run_root)
    expected_seeds = [int(seed) for seed in expected_seeds]
    output_dir = Path(output_dir)
    frames = _load_seed_frames(run_root, expected_seeds)
    site_ids, labels, seed_scores = _validated_seed_arrays(frames, expected_seeds)
    metrics, per_seed = build_evaluation_frames(frames, expected_seeds)
    metrics.insert(0, "prediction_split", "held_out_test")
    metrics.insert(1, "seed_count", len(expected_seeds))
    per_seed.insert(0, "prediction_split", "held_out_test")
    thresholds = []
    ensemble_scores = seed_scores.mean(axis=0)
    for task, task_label, label_column, score_column, task_labels, task_scores in _task_arrays(labels, ensemble_scores):
        table = threshold_workpoints(task_labels, task_scores)
        table.insert(0, "prediction_split", "held_out_test")
        table.insert(1, "task", task)
        table.insert(2, "task_label", task_label)
        table.insert(3, "label_column", label_column)
        table.insert(4, "score_column", score_column)
        thresholds.append(table)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    per_seed.to_csv(output_dir / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(thresholds, ignore_index=True).to_csv(
        output_dir / "threshold_table.csv", index=False, encoding="utf-8-sig"
    )
    for task, _, _, _, task_labels, task_scores in _task_arrays(labels, ensemble_scores):
        task_row = metrics.loc[metrics["task"].eq(task)].iloc[0]
        _plot_task(task_row, task_labels, task_scores, output_dir / FIGURE_STEMS[task], len(site_ids))
    _plot_auc_comparison(metrics, output_dir / FIGURE_STEMS["auc_comparison"])
    (output_dir / "README.md").write_text(_readme(len(expected_seeds), len(site_ids)), encoding="utf-8")
    return output_dir
