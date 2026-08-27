import argparse
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from src.multimodal.config import MultimodalConfig
from src.multimodal.features.sequence import ALPHABET


LABEL_NAMES = ("activity", "interaction", "proteostasis")


def fit_lightgbm_baseline(
    train_x,
    train_y,
    predict_x,
    seed,
    n_estimators=500,
    feature_names=None,
):
    models = {}
    probabilities = []
    for column, name in enumerate(LABEL_NAMES):
        positives = max(int(train_y[:, column].sum()), 1)
        negatives = max(len(train_y) - positives, 1)
        model = LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=negatives / positives,
            random_state=seed + column,
            n_jobs=-1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        fit_kwargs = {"feature_name": feature_names} if feature_names else {}
        model.fit(train_x, train_y[:, column], **fit_kwargs)
        models[name] = model
        probabilities.append(model.predict_proba(predict_x)[:, 1])
    return models, np.stack(probabilities, axis=1)


def build_baseline_features(config):
    sequence = np.load(config.processed_data / "sequence.npy", mmap_mode="r")
    structure = np.load(config.processed_data / "structure_processed.npy")
    network = np.load(config.processed_data / "network_processed.npy")
    proteomics = np.load(config.processed_data / "proteomics_processed.npy")
    masks = np.load(config.processed_data / "modality_masks.npy")
    frequency = sequence.mean(axis=1, dtype=np.float32)
    center = np.asarray(sequence[:, sequence.shape[1] // 2, :], dtype=np.float32)
    features = np.concatenate(
        [frequency, center, structure, network, proteomics, masks], axis=1
    ).astype(np.float32, copy=False)

    column_path = config.artifacts / "preprocessors" / "feature_columns.json"
    tabular_columns = json.loads(column_path.read_text(encoding="utf-8"))
    feature_names = (
        [f"sequence_frequency_{residue}" for residue in ALPHABET]
        + [f"sequence_center_{residue}" for residue in ALPHABET]
        + [f"structure_{column}" for column in tabular_columns["structure"]]
        + [f"network_{column}" for column in tabular_columns["network"]]
        + [f"proteomics_{column}" for column in tabular_columns["proteomics"]]
        + [f"modality_available_{name}" for name in ("sequence", "structure", "network", "proteomics")]
    )
    if len(feature_names) != features.shape[1]:
        raise AssertionError("baseline feature names do not match the matrix width")
    if not np.isfinite(features).all():
        raise AssertionError("baseline feature matrix contains non-finite values")
    return features, feature_names


def _split_metrics(labels, probabilities):
    metrics = {}
    for column, name in enumerate(LABEL_NAMES):
        metrics[name] = {
            "auprc": float(average_precision_score(labels[:, column], probabilities[:, column])),
            "auroc": float(roc_auc_score(labels[:, column], probabilities[:, column])),
            "positives": int(labels[:, column].sum()),
            "rows": int(len(labels)),
        }
    metrics["macro_auprc"] = float(
        np.mean([metrics[name]["auprc"] for name in LABEL_NAMES])
    )
    metrics["macro_auroc"] = float(
        np.mean([metrics[name]["auroc"] for name in LABEL_NAMES])
    )
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train a shared-split LightGBM baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-estimators", type=int, default=500)
    args = parser.parse_args(argv)

    config = MultimodalConfig.load(args.config)
    config.ensure_output_directories()
    features, feature_names = build_baseline_features(config)
    labels = np.load(config.processed_data / "labels.npy")
    splits = np.load(config.processed_data / "split_names.npy", allow_pickle=False)
    site_ids = np.load(config.processed_data / "dataset_site_ids.npy", allow_pickle=False)
    if not (len(features) == len(labels) == len(splits) == len(site_ids)):
        raise AssertionError("baseline inputs have different row counts")
    train = splits == "train"
    models, probabilities = fit_lightgbm_baseline(
        features[train],
        labels[train],
        features,
        seed=config.seed,
        n_estimators=args.n_estimators,
        feature_names=feature_names,
    )

    output_directory = config.artifacts / "baseline"
    output_directory.mkdir(parents=True, exist_ok=True)
    package = {
        "models": models,
        "feature_names": feature_names,
        "label_names": LABEL_NAMES,
        "unlabeled_treated_as_negative": True,
        "seed": config.seed,
    }
    joblib.dump(package, output_directory / "lightgbm_models.joblib")
    prediction_frame = pd.DataFrame(
        {
            "site_id": site_ids,
            "split": splits,
            **{
                f"{name}_probability": probabilities[:, column]
                for column, name in enumerate(LABEL_NAMES)
            },
        }
    )
    for split_name in ("validation", "test"):
        prediction_frame[prediction_frame["split"].eq(split_name)].to_parquet(
            output_directory / f"{split_name}_predictions.parquet", index=False
        )
    validation = splits == "validation"
    test = splits == "test"
    report = {
        "model": "LightGBM one-vs-rest",
        "feature_count": int(features.shape[1]),
        "n_estimators": int(args.n_estimators),
        "seed": int(config.seed),
        "unlabeled_treated_as_negative": True,
        "scope_note": "Comparator only; the multimodal neural model uses nnPU risk.",
        "validation": _split_metrics(labels[validation], probabilities[validation]),
        "test": _split_metrics(labels[test], probabilities[test]),
    }
    (output_directory / "baseline_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
