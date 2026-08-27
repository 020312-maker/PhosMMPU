"""Known-positive ranking metrics for positive-unlabeled evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def ranking_metrics(labels, scores, class_prior, k_values=(100, 200, 500), g_k=100):
    """Measure recovery of known positives without treating U as confirmed negative."""
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if labels.shape != scores.shape or not len(labels):
        raise ValueError("labels and scores must be non-empty aligned vectors")
    if not set(np.unique(labels)).issubset({0, 1}) or not np.isfinite(scores).all():
        raise ValueError("labels must be binary and scores finite")
    if not 0 < float(class_prior) <= 1:
        raise ValueError("class_prior must lie in (0, 1]")
    positives = int(labels.sum())
    if positives == 0:
        raise ValueError("known-positive ranking metrics require at least one positive")
    ranking = np.argsort(-scores, kind="mergesort")
    relevance = labels[ranking]
    output = {"pr_auc_observed_lower_bound": float(average_precision_score(labels, scores))}
    for requested_k in k_values:
        k = min(int(requested_k), len(labels))
        hits = int(relevance[:k].sum())
        output[f"top_hits_at_{requested_k}"] = hits
        output[f"recall_at_{requested_k}"] = hits / positives
    g_k = int(g_k)
    k = min(g_k, len(labels))
    recall = output[f"recall_at_{g_k}"]
    selected_fraction = k / len(labels)
    output[f"ndcg_at_{g_k}"] = float(
        (relevance[:k] / np.log2(np.arange(2, k + 2))).sum()
        / (np.ones(min(positives, k)) / np.log2(np.arange(2, min(positives, k) + 2))).sum()
    )
    first_positive = int(np.flatnonzero(relevance)[0])
    output["mrr"] = 1.0 / (first_positive + 1)
    output[f"g_metric_at_{g_k}"] = recall**2 / selected_fraction
    # This prior-based estimate is reported as a probability, so retain its valid range.
    output[f"adjusted_precision_at_{g_k}"] = min(
        1.0, float(class_prior) * recall / selected_fraction
    )
    return output
