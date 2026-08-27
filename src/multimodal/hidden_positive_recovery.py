"""Label-hidden known-positive recovery utilities for PU experiments."""

from __future__ import annotations

import numpy as np


TASKS = ("activity", "interaction", "proteostasis")


def _stratified_choice(eligible, strata, count, generator):
    """Sample proportionally while retaining every stratum with available quota."""
    values = np.asarray(strata)[eligible]
    names, inverse, available = np.unique(values, return_inverse=True, return_counts=True)
    target = available / available.sum() * int(count)
    allocation = np.floor(target).astype(int)
    allocation = np.minimum(allocation, available)
    remaining = int(count) - int(allocation.sum())
    priority = np.argsort(-(target - allocation), kind="mergesort")
    for group in priority:
        if not remaining:
            break
        if allocation[group] < available[group]:
            allocation[group] += 1
            remaining -= 1
    if remaining:
        raise ValueError("not enough eligible values for stratified sampling")
    samples = [
        generator.choice(eligible[inverse == group], size=allocation[group], replace=False)
        for group in range(len(names))
        if allocation[group]
    ]
    return np.sort(np.concatenate(samples))


def sample_hidden_positives(labels, train_positions, count_per_task=100, seed=20260814, strata=None):
    """Select task-exclusive observed positives to hide from the training labels.

    Requiring a site to be positive for one task only prevents its other task
    labels from disclosing the hidden target label to the multi-task model.
    """
    labels = np.asarray(labels, dtype=np.int8)
    train_positions = np.asarray(train_positions, dtype=np.int64)
    if labels.ndim != 2 or labels.shape[1] != len(TASKS):
        raise ValueError("labels must have shape (rows, 3)")
    if int(count_per_task) < 1:
        raise ValueError("count_per_task must be positive")
    if train_positions.ndim != 1 or len(np.unique(train_positions)) != len(train_positions):
        raise ValueError("train_positions must be one-dimensional and unique")
    if len(train_positions) and (train_positions.min() < 0 or train_positions.max() >= len(labels)):
        raise ValueError("train_positions are outside label rows")
    if strata is not None and len(np.asarray(strata)) != len(labels):
        raise ValueError("strata must align with label rows")
    generator = np.random.default_rng(int(seed))
    exclusive = labels.sum(axis=1) == 1
    selected = {}
    for task_index, task in enumerate(TASKS):
        eligible = train_positions[(labels[train_positions, task_index] == 1) & exclusive[train_positions]]
        if len(eligible) < int(count_per_task):
            raise ValueError(f"{task} has only {len(eligible)} task-exclusive training positives")
        selected[task_index] = (
            _stratified_choice(eligible, strata, int(count_per_task), generator)
            if strata is not None
            else np.sort(generator.choice(eligible, size=int(count_per_task), replace=False))
        )
    return selected


def mask_hidden_positives(labels, hidden_by_task):
    """Convert selected observed positives to U (0) in their target task only."""
    labels = np.asarray(labels, dtype=np.int8)
    if labels.ndim != 2 or labels.shape[1] != len(TASKS):
        raise ValueError("labels must have shape (rows, 3)")
    masked = labels.copy()
    for task_index in range(len(TASKS)):
        if task_index not in hidden_by_task:
            raise ValueError("hidden_by_task must contain all three task indices")
        positions = np.asarray(hidden_by_task[task_index], dtype=np.int64)
        if positions.ndim != 1 or len(np.unique(positions)) != len(positions):
            raise ValueError("hidden positions must be one-dimensional and unique")
        if len(positions) and (positions.min() < 0 or positions.max() >= len(labels)):
            raise ValueError("hidden positions are outside label rows")
        if not np.all(labels[positions, task_index] == 1):
            raise ValueError("only observed positives may be hidden")
        masked[positions, task_index] = 0
    return masked


def candidate_positions_after_mask(masked_labels, train_positions):
    """Return the per-task training candidate pools after P labels are hidden."""
    masked_labels = np.asarray(masked_labels, dtype=np.int8)
    train_positions = np.asarray(train_positions, dtype=np.int64)
    if masked_labels.ndim != 2 or masked_labels.shape[1] != len(TASKS):
        raise ValueError("masked_labels must have shape (rows, 3)")
    if train_positions.ndim != 1 or len(np.unique(train_positions)) != len(train_positions):
        raise ValueError("train_positions must be one-dimensional and unique")
    if len(train_positions) and (train_positions.min() < 0 or train_positions.max() >= len(masked_labels)):
        raise ValueError("train_positions are outside masked label rows")
    return {
        task_index: train_positions[masked_labels[train_positions, task_index] == 0]
        for task_index in range(len(TASKS))
    }


def evaluate_hidden_recovery(hidden_positions, candidate_positions, candidate_scores, top_ks=(50, 100, 200, 500)):
    """Evaluate whether hidden positives are recovered from the U candidate pool."""
    hidden_positions = np.asarray(hidden_positions, dtype=np.int64)
    candidate_positions = np.asarray(candidate_positions, dtype=np.int64)
    candidate_scores = np.asarray(candidate_scores, dtype=float)
    if hidden_positions.ndim != 1 or candidate_positions.ndim != 1 or candidate_scores.ndim != 1:
        raise ValueError("hidden positions, candidate positions, and scores must be one-dimensional")
    if not len(hidden_positions) or not len(candidate_positions):
        raise ValueError("hidden positives and candidates must be non-empty")
    if len(np.unique(hidden_positions)) != len(hidden_positions) or len(np.unique(candidate_positions)) != len(candidate_positions):
        raise ValueError("hidden positives and candidates must be unique")
    if len(candidate_positions) != len(candidate_scores) or not np.isfinite(candidate_scores).all():
        raise ValueError("candidate scores must be finite and align with candidates")
    if not np.isin(hidden_positions, candidate_positions).all():
        raise ValueError("every hidden positive must be included in the candidate pool")
    top_ks = tuple(int(value) for value in top_ks)
    if not top_ks or min(top_ks) < 1:
        raise ValueError("top_ks must contain positive integers")

    order = np.lexsort((candidate_positions, -candidate_scores))
    ranked_positions = candidate_positions[order]
    hidden_set = set(hidden_positions.tolist())
    relevant = np.fromiter((int(position in hidden_set) for position in ranked_positions), dtype=np.int8)
    ranks = np.flatnonzero(relevant) + 1
    output = {
        "hidden_positive_count": int(len(hidden_positions)),
        "candidate_count": int(len(candidate_positions)),
        "mean_hidden_rank": float(ranks.mean()),
        "mrr": float((1.0 / ranks).mean()),
    }
    for top_k in top_ks:
        limit = min(top_k, len(candidate_positions))
        hit_count = int(relevant[:limit].sum())
        ideal = min(len(hidden_positions), limit)
        discounts = 1.0 / np.log2(np.arange(2, limit + 2, dtype=float))
        dcg = float((relevant[:limit] * discounts).sum())
        idcg = float(discounts[:ideal].sum())
        output[f"hits_at_{top_k}"] = hit_count
        output[f"recall_at_{top_k}"] = float(hit_count / len(hidden_positions))
        output[f"random_recall_at_{top_k}"] = float(limit / len(candidate_positions))
        output[f"ndcg_at_{top_k}"] = float(dcg / idcg) if idcg else 0.0
    return output
