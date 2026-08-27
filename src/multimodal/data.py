import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.multimodal.config import MultimodalConfig
from src.multimodal.labels import LABEL_COLUMNS


MODALITY_NAMES = ("sequence", "structure", "network", "proteomics")


@dataclass(frozen=True)
class MultimodalArrays:
    site_ids: np.ndarray
    sequence: np.ndarray
    structure: np.ndarray
    network: np.ndarray
    proteomics: np.ndarray
    masks: np.ndarray
    feature_columns: dict


def _infer_numeric_columns(table, presence_column=None):
    excluded = {"site_id", "accession", presence_column}
    return [
        column
        for column in table.columns
        if column not in excluded
        and (is_numeric_dtype(table[column]) or is_bool_dtype(table[column]))
    ]


def _align_table(
    sites,
    table,
    prefix,
    feature_columns=None,
    presence_column=None,
):
    if table["site_id"].duplicated().any():
        raise ValueError(f"{prefix} table contains duplicate site_id values")
    columns = list(
        feature_columns
        if feature_columns is not None
        else _infer_numeric_columns(table, presence_column)
    )
    missing_columns = set(columns) - set(table.columns)
    if missing_columns:
        raise ValueError(f"{prefix} table is missing columns: {sorted(missing_columns)}")
    working = table[["site_id", *columns]].copy()
    if presence_column and presence_column in table:
        working[presence_column] = table[presence_column].to_numpy()
    working["__row_present"] = 1.0
    aligned = sites[["site_id"]].merge(
        working, on="site_id", how="left", validate="one_to_one"
    )
    if presence_column and presence_column in aligned:
        available = aligned[presence_column].fillna(0).to_numpy(dtype=np.float32)
    else:
        available = aligned["__row_present"].fillna(0).to_numpy(dtype=np.float32)
    available = (available > 0).astype(np.float32)

    if not columns:
        columns = [f"{prefix}0"]
        values = np.full((len(sites), 1), np.nan, dtype=np.float32)
    else:
        values = aligned[columns].to_numpy(dtype=np.float32)
    values[available == 0] = 0.0
    return values, available, columns


def _sequence_array(site_ids, sequence):
    if isinstance(sequence, dict):
        missing = [site_id for site_id in site_ids if site_id not in sequence]
        if missing:
            raise ValueError(f"sequence features are missing {len(missing)} sites")
        return np.stack([sequence[site_id] for site_id in site_ids]).astype(np.float32)
    array = np.asarray(sequence, dtype=np.float32)
    if len(array) != len(site_ids):
        raise ValueError("sequence array length differs from site index")
    return array


def assemble_modalities(
    sites,
    sequence,
    structure,
    network,
    proteomics,
    feature_columns=None,
):
    feature_columns = feature_columns or {}
    site_ids = np.asarray(sites["site_id"].astype(str).tolist(), dtype=np.str_)
    sequence_array = _sequence_array(site_ids, sequence)
    structure_array, structure_mask, structure_columns = _align_table(
        sites,
        structure,
        "s",
        feature_columns.get("structure"),
        "structure_present" if "structure_present" in structure else None,
    )
    network_array, network_mask, network_columns = _align_table(
        sites,
        network,
        "n",
        feature_columns.get("network"),
        "network_present" if "network_present" in network else None,
    )
    proteomics_array, proteomics_mask, proteomics_columns = _align_table(
        sites,
        proteomics,
        "p",
        feature_columns.get("proteomics"),
        "proteomics_present" if "proteomics_present" in proteomics else None,
    )
    sequence_mask = np.ones(len(sites), dtype=np.float32)
    masks = np.stack(
        [sequence_mask, structure_mask, network_mask, proteomics_mask], axis=1
    )
    return MultimodalArrays(
        site_ids=site_ids,
        sequence=sequence_array,
        structure=structure_array,
        network=network_array,
        proteomics=proteomics_array,
        masks=masks,
        feature_columns={
            "structure": structure_columns,
            "network": network_columns,
            "proteomics": proteomics_columns,
        },
    )


class FeaturePreprocessor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.fitted_split = None
        self.feature_columns = None
        self.empty_feature_mask = None

    def _values_and_columns(self, values):
        if isinstance(values, pd.DataFrame):
            columns = list(values.columns)
            array = values.to_numpy(dtype=np.float64)
        else:
            columns = None
            array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError("feature values must be a two-dimensional matrix")
        return array, columns

    def fit(self, values, split_name, feature_columns=None):
        if split_name != "train":
            raise ValueError("feature preprocessing may only fit on train")
        array, dataframe_columns = self._values_and_columns(values)
        columns = dataframe_columns or list(feature_columns or [])
        if not columns:
            columns = [f"feature_{index}" for index in range(array.shape[1])]
        if len(columns) != array.shape[1]:
            raise ValueError("feature column count differs from feature matrix width")
        self.feature_columns = columns
        self.empty_feature_mask = np.isnan(array).all(axis=0)
        fit_values = array.copy()
        fit_values[:, self.empty_feature_mask] = 0.0
        imputed = self.imputer.fit_transform(fit_values)
        if imputed.shape[1] != array.shape[1]:
            raise AssertionError("imputation changed the feature width")
        self.scaler.fit(imputed)
        self.fitted_split = split_name
        return self

    def transform(self, values, feature_columns=None):
        if self.fitted_split != "train":
            raise RuntimeError("preprocessor has not been fitted on train")
        array, dataframe_columns = self._values_and_columns(values)
        columns = dataframe_columns or list(feature_columns or self.feature_columns)
        if columns != self.feature_columns:
            raise ValueError("feature column order differs from fitted column order")
        transformed_values = array.copy()
        transformed_values[:, self.empty_feature_mask] = 0.0
        transformed = self.scaler.transform(
            self.imputer.transform(transformed_values)
        ).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise AssertionError("preprocessed feature matrix contains non-finite values")
        return transformed


def _save_array(path, values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Align and validate multimodal arrays")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    config = MultimodalConfig.load(args.config)
    config.ensure_output_directories()
    sites = pd.read_parquet(config.processed_data / "site_index.parquet")
    labels = pd.read_parquet(config.processed_data / "labels.parquet")
    splits = pd.read_parquet(config.processed_data / "splits.parquet")
    structure = pd.read_parquet(config.processed_data / "structure_features.parquet")
    network = pd.read_parquet(config.processed_data / "network_features.parquet")
    proteomics = pd.read_parquet(config.processed_data / "proteomics_features.parquet")
    sequence = np.load(config.processed_data / "sequence.npy", mmap_mode="r")
    sequence_ids = np.load(
        config.processed_data / "sequence_site_ids.npy", allow_pickle=False
    )
    site_ids = np.asarray(sites["site_id"].astype(str).tolist(), dtype=np.str_)
    if not np.array_equal(sequence_ids, site_ids):
        raise ValueError("sequence feature order differs from site index")

    from src.multimodal.features.structure import STRUCTURE_FEATURE_COLUMNS

    network_columns = [
        column
        for column in network.columns
        if column not in {"site_id", "accession", "network_present"}
        and (is_numeric_dtype(network[column]) or is_bool_dtype(network[column]))
    ]
    proteomics_columns = [
        column
        for column in proteomics.columns
        if column not in {"site_id", "proteomics_present"}
        and (is_numeric_dtype(proteomics[column]) or is_bool_dtype(proteomics[column]))
    ]
    arrays = assemble_modalities(
        sites,
        sequence,
        structure,
        network,
        proteomics,
        feature_columns={
            "structure": STRUCTURE_FEATURE_COLUMNS,
            "network": network_columns,
            "proteomics": proteomics_columns,
        },
    )
    if not set(np.unique(arrays.masks)).issubset({0.0, 1.0}):
        raise AssertionError("modality masks are not binary")
    if not np.all(arrays.masks[:, 0] == 1):
        raise AssertionError("at least one site is missing sequence features")
    if not np.isfinite(arrays.sequence).all():
        raise AssertionError("sequence array contains non-finite values")

    aligned_labels = sites[["site_id"]].merge(
        labels[["site_id", *LABEL_COLUMNS]],
        on="site_id",
        how="left",
        validate="one_to_one",
    )
    aligned_splits = sites[["site_id"]].merge(
        splits[["site_id", "split"]],
        on="site_id",
        how="left",
        validate="one_to_one",
    )
    if aligned_labels[LABEL_COLUMNS].isna().any().any():
        raise AssertionError("labels do not cover the site index")
    if aligned_splits["split"].isna().any():
        raise AssertionError("splits do not cover the site index")
    train_mask = aligned_splits["split"].eq("train").to_numpy()

    preprocessors = {}
    processed_arrays = {}
    modality_values = {
        "structure": arrays.structure,
        "network": arrays.network,
        "proteomics": arrays.proteomics,
    }
    modality_mask_columns = {"structure": 1, "network": 2, "proteomics": 3}
    preprocessor_directory = config.artifacts / "preprocessors"
    preprocessor_directory.mkdir(parents=True, exist_ok=True)
    fit_counts = {}
    for modality, values in modality_values.items():
        available_train = train_mask & arrays.masks[:, modality_mask_columns[modality]].astype(bool)
        if not available_train.any():
            raise ValueError(f"no available training rows for {modality}")
        preprocessor = FeaturePreprocessor().fit(
            values[available_train],
            split_name="train",
            feature_columns=arrays.feature_columns[modality],
        )
        processed = preprocessor.transform(
            values, feature_columns=arrays.feature_columns[modality]
        )
        preprocessors[modality] = preprocessor
        processed_arrays[modality] = processed
        fit_counts[modality] = int(available_train.sum())
        joblib.dump(preprocessor, preprocessor_directory / f"{modality}.joblib")

    _save_array(config.processed_data / "dataset_site_ids.npy", arrays.site_ids)
    _save_array(config.processed_data / "modality_masks.npy", arrays.masks)
    _save_array(config.processed_data / "structure_raw.npy", arrays.structure)
    _save_array(config.processed_data / "network_raw.npy", arrays.network)
    _save_array(config.processed_data / "proteomics_raw.npy", arrays.proteomics)
    for modality, values in processed_arrays.items():
        _save_array(config.processed_data / f"{modality}_processed.npy", values)
    _save_array(
        config.processed_data / "labels.npy",
        aligned_labels[LABEL_COLUMNS].to_numpy(dtype=np.int8),
    )
    split_values = np.asarray(aligned_splits["split"].astype(str).tolist(), dtype=np.str_)
    _save_array(config.processed_data / "split_names.npy", split_values)

    column_path = preprocessor_directory / "feature_columns.json"
    column_path.write_text(
        json.dumps(arrays.feature_columns, indent=2), encoding="utf-8"
    )
    coverage = arrays.masks.mean(axis=0)
    audit = {
        "sites": int(len(sites)),
        "sequence_shape": list(arrays.sequence.shape),
        "structure_shape": list(arrays.structure.shape),
        "network_shape": list(arrays.network.shape),
        "proteomics_shape": list(arrays.proteomics.shape),
        "modality_coverage": {
            name: float(value) for name, value in zip(MODALITY_NAMES, coverage)
        },
        "preprocessor_fit_split": "train",
        "preprocessor_fit_rows": fit_counts,
        "labels_aligned": True,
        "splits_aligned": True,
        "validate_only_requested": bool(args.validate_only),
    }
    (config.processed_data / "data_integrity_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
