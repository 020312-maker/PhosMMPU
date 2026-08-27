"""Validation-thresholded PU evaluation and random-score reference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


TASKS = ("activity", "interaction", "proteostasis")
METRICS = ("roc_auc", "pr_auc", "f1", "mcc", "precision", "recall")


def _arrays(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError("labels and scores must share shape (rows, 3)")
    if not set(np.unique(labels)).issubset({0, 1}) or not np.isfinite(scores).all():
        raise ValueError("labels must be binary and scores finite")
    return labels, scores


def validation_f1_thresholds(labels, scores):
    labels, scores = _arrays(labels, scores)
    thresholds = []
    for column in range(3):
        task_scores = scores[:, column]
        task_labels = labels[:, column]
        candidates = np.unique(np.concatenate(([0.0], task_scores, [1.0])))
        order = np.argsort(task_scores, kind="mergesort")
        ordered_scores = task_scores[order]
        prefix_positive = np.concatenate(([0], np.cumsum(task_labels[order], dtype=np.int64)))
        positions = np.searchsorted(ordered_scores, candidates, side="left")
        true_positive = int(task_labels.sum()) - prefix_positive[positions]
        predicted_positive = len(task_labels) - positions
        false_positive = predicted_positive - true_positive
        false_negative = int(task_labels.sum()) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        values = np.divide(2 * true_positive, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)
        thresholds.append(float(candidates[np.flatnonzero(values == values.max())[0]]))
    return thresholds


def _task_metrics(labels, scores, threshold):
    predicted = scores >= threshold
    if labels.min() == labels.max():
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(labels, scores))
        pr_auc = float(average_precision_score(labels, scores))
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "threshold": float(threshold),
    }


def evaluate_test(labels, scores, thresholds):
    labels, scores = _arrays(labels, scores)
    if len(thresholds) != 3:
        raise ValueError("three validation-derived thresholds are required")
    output = {
        task: _task_metrics(labels[:, column], scores[:, column], thresholds[column])
        for column, task in enumerate(TASKS)
    }
    output["macro"] = {
        metric: float(np.nanmean([output[task][metric] for task in TASKS]))
        for metric in METRICS
    }
    return output


def random_reference(validation_labels, test_labels, repeats=100, seed=20260720):
    validation_labels = np.asarray(validation_labels, dtype=np.int8)
    test_labels = np.asarray(test_labels, dtype=np.int8)
    _arrays(validation_labels, validation_labels.astype(float))
    _arrays(test_labels, test_labels.astype(float))
    if int(repeats) < 1:
        raise ValueError("random repeats must be positive")
    generator = np.random.default_rng(seed)
    rows = []
    for repeat in range(int(repeats)):
        validation_scores = generator.random(validation_labels.shape)
        test_scores = generator.random(test_labels.shape)
        thresholds = validation_f1_thresholds(validation_labels, validation_scores)
        metrics = evaluate_test(test_labels, test_scores, thresholds)
        for task in TASKS:
            rows.append(
                {
                    "source": "random_prevalence_matched",
                    "repeat": repeat,
                    "task": task,
                    "threshold": thresholds[TASKS.index(task)],
                    "observed_positive_ratio": float(test_labels[:, TASKS.index(task)].mean()),
                    **metrics[task],
                }
            )
    return pd.DataFrame(rows)


def _metric_rows(metrics, seed):
    rows = []
    for task, values in metrics.items():
        for metric in METRICS:
            rows.append({"seed": int(seed), "task": task, "metric": metric, "value": float(values[metric])})
    return rows


def _seed_mean_std(seed_rows):
    values = pd.DataFrame(seed_rows)
    return (
        values.groupby(["task", "metric"], as_index=False)["value"]
        .agg(mean="mean", std="std", seed_count="count")
        .sort_values(["task", "metric"], kind="mergesort")
        .reset_index(drop=True)
    )


def _stratified_bootstrap_positions(labels, repeats, seed):
    """Draw joint label-pattern-stratified test-site samples for paired analysis."""
    generator = np.random.default_rng(seed)
    label_patterns = np.asarray(labels, dtype=np.int8) @ np.asarray([1, 2, 4], dtype=np.int8)
    strata = [np.flatnonzero(label_patterns == pattern) for pattern in np.unique(label_patterns)]
    return [
        np.concatenate([generator.choice(rows, size=len(rows), replace=True) for rows in strata])
        for _ in range(int(repeats))
    ]


def _stratified_bootstrap(labels, scores, thresholds, repeats, seed):
    # Preserve the joint three-task label composition so task and macro intervals
    # originate from the same bootstrap replicate.
    values = {task: {metric: [] for metric in METRICS} for task in (*TASKS, "macro")}
    for positions in _stratified_bootstrap_positions(labels, repeats, seed):
        metrics = evaluate_test(labels[positions], scores[positions], thresholds)
        for task, task_metrics in metrics.items():
            for metric in METRICS:
                if np.isfinite(task_metrics[metric]):
                    values[task][metric].append(task_metrics[metric])
    rows = []
    for task, task_values in values.items():
        for metric, samples in task_values.items():
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "ci_low": float(np.quantile(samples, 0.025)) if samples else float("nan"),
                    "ci_high": float(np.quantile(samples, 0.975)) if samples else float("nan"),
                    "valid_repeats": int(len(samples)),
                    "requested_repeats": int(repeats),
                }
            )
    return pd.DataFrame(rows)


def paired_bootstrap_difference(
    labels, reference_scores, reference_thresholds, candidate_scores, candidate_thresholds,
    repeats=1000, seed=20260806,
):
    """Return a test-site paired Bootstrap CI for reference minus candidate metrics.

    Every replicate samples the same test-site indices for both models.  Thresholds
    are supplied separately because each model selects them from its own validation
    predictions before the held-out test evaluation.
    """
    labels, reference_scores = _arrays(labels, reference_scores)
    _, candidate_scores = _arrays(labels, candidate_scores)
    if len(reference_thresholds) != 3 or len(candidate_thresholds) != 3:
        raise ValueError("three validation-derived thresholds are required for both models")
    reference_metrics = evaluate_test(labels, reference_scores, reference_thresholds)
    candidate_metrics = evaluate_test(labels, candidate_scores, candidate_thresholds)
    values = {task: {metric: [] for metric in METRICS} for task in (*TASKS, "macro")}
    for positions in _stratified_bootstrap_positions(labels, repeats, seed):
        reference_sample = evaluate_test(labels[positions], reference_scores[positions], reference_thresholds)
        candidate_sample = evaluate_test(labels[positions], candidate_scores[positions], candidate_thresholds)
        for task in values:
            for metric in METRICS:
                difference = reference_sample[task][metric] - candidate_sample[task][metric]
                if np.isfinite(difference):
                    values[task][metric].append(float(difference))
    rows = []
    for task in (*TASKS, "macro"):
        for metric in METRICS:
            samples = values[task][metric]
            rows.append(
                {
                    "task": task,
                    "metric": metric,
                    "reference_value": float(reference_metrics[task][metric]),
                    "candidate_value": float(candidate_metrics[task][metric]),
                    "difference": float(reference_metrics[task][metric] - candidate_metrics[task][metric]),
                    "ci_low": float(np.quantile(samples, 0.025)) if samples else float("nan"),
                    "ci_high": float(np.quantile(samples, 0.975)) if samples else float("nan"),
                    "valid_repeats": int(len(samples)),
                    "requested_repeats": int(repeats),
                    "bootstrap_unit": "paired_test_sites",
                }
            )
    return pd.DataFrame(rows)


def _ensemble_point_table(metrics):
    return pd.DataFrame(
        [
            {"task": task, "metric": metric, "ensemble_value": float(values[metric])}
            for task, values in metrics.items()
            for metric in METRICS
        ]
    )


def build_unified_report_table(report):
    """Combine one ensemble point estimate, its CI, and seed stability statistics."""
    required = {"ensemble_summary", "bootstrap", "seed_summary"}
    if required.difference(report):
        raise ValueError("report is missing unified evaluation components")
    point = report["ensemble_summary"]
    bootstrap = report["bootstrap"]
    seed_summary = report["seed_summary"]
    detail = point.merge(bootstrap, on=["task", "metric"], how="inner", validate="one_to_one")
    detail = detail.merge(seed_summary, on=["task", "metric"], how="left", validate="one_to_one")
    if len(detail) != len(point) or detail[["ci_low", "ci_high"]].isna().any().any():
        raise ValueError("every ensemble metric must have a bootstrap confidence interval")
    return detail.sort_values(["task", "metric"], kind="mergesort").reset_index(drop=True)


def ensemble_bootstrap_report(
    validation_labels, validation_seed_scores, test_labels, test_seed_scores, repeats=1000, seed=20260806,
    seed_ids=None,
):
    validation_labels, _ = _arrays(validation_labels, np.asarray(validation_seed_scores)[0])
    test_labels, _ = _arrays(test_labels, np.asarray(test_seed_scores)[0])
    validation_seed_scores = np.asarray(validation_seed_scores, dtype=float)
    test_seed_scores = np.asarray(test_seed_scores, dtype=float)
    if validation_seed_scores.ndim != 3 or test_seed_scores.ndim != 3:
        raise ValueError("seed scores must have shape (seeds, rows, 3)")
    if len(validation_seed_scores) != len(test_seed_scores):
        raise ValueError("validation and test must contain the same seed count")
    if validation_seed_scores.shape[1:] != validation_labels.shape or test_seed_scores.shape[1:] != test_labels.shape:
        raise ValueError("seed score shapes must align with labels")
    if seed_ids is None:
        seed_ids = tuple(range(len(validation_seed_scores)))
    if len(seed_ids) != len(validation_seed_scores):
        raise ValueError("seed_ids must align with seed score arrays")
    validation_mean = validation_seed_scores.mean(axis=0)
    test_mean = test_seed_scores.mean(axis=0)
    ensemble_thresholds = validation_f1_thresholds(validation_labels, validation_mean)
    seed_rows = []
    per_seed_bootstrap = []
    for ordinal, (seed_id, validation_scores, test_scores) in enumerate(
        zip(seed_ids, validation_seed_scores, test_seed_scores)
    ):
        seed_thresholds = validation_f1_thresholds(validation_labels, validation_scores)
        seed_metrics = evaluate_test(test_labels, test_scores, seed_thresholds)
        seed_rows.extend(_metric_rows(seed_metrics, seed_id))
        seed_bootstrap = _stratified_bootstrap(
            test_labels, test_scores, seed_thresholds, repeats, int(seed) + ordinal + 1,
        ).merge(_ensemble_point_table(seed_metrics), on=["task", "metric"], how="inner", validate="one_to_one")
        seed_bootstrap["seed"] = int(seed_id)
        seed_bootstrap["bootstrap_unit"] = "test_sites"
        per_seed_bootstrap.append(seed_bootstrap)
    ensemble_bootstrap = _stratified_bootstrap(test_labels, test_mean, ensemble_thresholds, repeats, seed)
    ensemble_bootstrap["bootstrap_unit"] = "test_sites"
    return {
        "thresholds": ensemble_thresholds,
        "seed_summary": _seed_mean_std(seed_rows),
        "per_seed_bootstrap": pd.concat(per_seed_bootstrap, ignore_index=True),
        "bootstrap": ensemble_bootstrap,
        "ensemble_summary": _ensemble_point_table(evaluate_test(test_labels, test_mean, ensemble_thresholds)),
        "ensemble_metrics": evaluate_test(test_labels, test_mean, ensemble_thresholds),
    }
