import numpy as np
import torch


def _prior_tensor(values, name):
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (3,) or not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError(f"{name} must contain three finite positive values")
    return torch.tensor(array, dtype=torch.float32)


def make_nnpu_loss(class_priors):
    priors = _prior_tensor(class_priors, "class_priors")

    def loss(y_true, y_pred):
        y_pred = y_pred.clamp(1e-7, 1.0 - 1e-7)
        priors_on_device = priors.to(y_pred.device)
        positive = y_true.float()
        unlabeled = 1.0 - positive
        positive_count = positive.sum(dim=0).clamp_min(1.0)
        unlabeled_count = unlabeled.sum(dim=0).clamp_min(1.0)
        positive_risk = priors_on_device * (positive * -y_pred.log()).sum(dim=0)
        positive_risk /= positive_count
        negative_risk = (unlabeled * -(1.0 - y_pred).log()).sum(dim=0)
        negative_risk /= unlabeled_count
        negative_risk -= priors_on_device * (
            positive * -(1.0 - y_pred).log()
        ).sum(dim=0) / positive_count
        return (positive_risk + negative_risk.clamp_min(0.0)).mean()

    return loss


def make_weighted_bce(positive_weights):
    weights = _prior_tensor(positive_weights, "positive_weights")

    def loss(y_true, y_pred):
        y_pred = y_pred.clamp(1e-7, 1.0 - 1e-7)
        y_true = y_true.float()
        weights_on_device = weights.to(y_pred.device)
        value = -(
            weights_on_device * y_true * y_pred.log()
            + (1.0 - y_true) * (1.0 - y_pred).log()
        )
        return value.mean()

    return loss


def make_bce_loss():
    """Ordinary BCE control that deliberately treats unlabeled zeros as negatives."""

    def loss(y_true, y_pred):
        return torch.nn.functional.binary_cross_entropy(
            y_pred.clamp(1e-7, 1.0 - 1e-7), y_true.float()
        )

    return loss


def estimate_class_priors(
    train_labels, split_name, multiplier=2.0, lower=0.01, upper=0.5
):
    if split_name != "train":
        raise ValueError("class priors may only be estimated from the training split")
    if not 0 < lower <= upper <= 1 or multiplier <= 0:
        raise ValueError("invalid class-prior bounds or multiplier")
    labels = torch.as_tensor(train_labels, dtype=torch.float32)
    if labels.ndim != 2 or labels.shape[1] != 3 or labels.shape[0] == 0:
        raise ValueError("train_labels must have shape (rows, 3)")
    values = (labels.mean(dim=0) * float(multiplier)).clamp(lower, upper)
    return [round(float(value), 8) for value in values]
