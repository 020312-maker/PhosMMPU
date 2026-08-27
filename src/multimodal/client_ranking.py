import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


LABEL_NAMES = ("activity", "interaction", "proteostasis")
WEIGHTS = np.asarray((0.40, 0.35, 0.25), dtype=np.float64)


class ConstantCalibrator:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, values):
        return np.full(len(values), self.value, dtype=np.float64)


def _arrays(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 2 or scores.shape != labels.shape or labels.shape[1] != 3:
        raise ValueError("labels and scores must have matching shape (rows, 3)")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("scores must be finite probabilities")
    return labels, scores


def fit_isotonic_calibrators(validation_labels, validation_scores):
    labels, scores = _arrays(validation_labels, validation_scores)
    calibrators = []
    for column in range(3):
        target = labels[:, column]
        if target.min() == target.max():
            calibrators.append(ConstantCalibrator(target[0]))
            continue
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(scores[:, column], target)
        calibrators.append(model)
    return calibrators


def apply_calibrators(calibrators, scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 3 or len(calibrators) != 3:
        raise ValueError("three calibrators and score columns are required")
    calibrated = np.column_stack(
        [calibrator.predict(scores[:, column]) for column, calibrator in enumerate(calibrators)]
    )
    return np.clip(calibrated, 0.0, 1.0)


def rank_by_balanced_confidence(site_ids, calibrated_scores):
    scores = np.asarray(calibrated_scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != 3 or len(site_ids) != len(scores):
        raise ValueError("site IDs and calibrated scores must align")
    result = pd.DataFrame(
        {
            "site_id": np.asarray(site_ids, dtype=str),
            "activity_confidence": scores[:, 0],
            "interaction_confidence": scores[:, 1],
            "proteostasis_confidence": scores[:, 2],
        }
    )
    result["balanced_confidence"] = result[
        ["activity_confidence", "interaction_confidence", "proteostasis_confidence"]
    ].to_numpy().dot(WEIGHTS)
    result = result.sort_values("balanced_confidence", ascending=False, kind="stable")
    result = result.reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)
    return result
