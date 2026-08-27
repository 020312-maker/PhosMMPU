import argparse
import json
import re

import numpy as np
import pandas as pd

from src.multimodal.config import MultimodalConfig


EXPERIMENTS = ("A", "B", "C")


def aggregate_human_sites(frame):
    grouped = frame.groupby("site_id", sort=True)
    aggregations = {
        "detection_count": ("site_id", "size"),
        "localization_max": ("localization_probability", "max"),
        "localization_mean": ("localization_probability", "mean"),
        "log_ratio_mean": ("log_ratio", "mean"),
        "log_ratio_std": ("log_ratio", "std"),
        "occupancy_mean": ("occupancy", "mean"),
    }
    if "log_intensity" in frame.columns:
        aggregations["log_intensity_mean"] = ("log_intensity", "mean")
    result = grouped.agg(**aggregations).reset_index()
    result["log_ratio_std"] = result["log_ratio_std"].fillna(0.0)
    return result


def map_mouse_evidence(mouse, orthology):
    validated = orthology[
        orthology["residue_conserved"].astype(bool)
        & orthology["human_site_id"].notna()
        & orthology["mouse_site_id"].notna()
    ].copy()
    mouse_counts = validated["mouse_site_id"].value_counts()
    human_counts = validated["human_site_id"].value_counts()
    validated = validated[
        validated["mouse_site_id"].map(mouse_counts).eq(1)
        & validated["human_site_id"].map(human_counts).eq(1)
    ]
    return mouse.merge(validated, on="mouse_site_id", how="inner")


def paired_protein_positions(proteins, positions):
    protein_values = [value.strip() for value in str(proteins).split(";")]
    position_values = [value.strip() for value in str(positions).split(";")]
    if len(protein_values) != len(position_values):
        raise ValueError("proteins and positions have different lengths")
    return list(zip(protein_values, position_values))


def _is_flagged(value):
    return str(value).strip().lower() in {"+", "1", "true", "yes"}


def _number(value):
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def _mean_available(values):
    numeric = np.asarray(values, dtype=float)
    return float(np.nanmean(numeric)) if np.isfinite(numeric).any() else np.nan


def _canonical_accession(value):
    return str(value).strip().split("-", 1)[0]


def map_human_maxquant(frame, site_index):
    retained_accessions = set(site_index["accession"].astype(str))
    retained_sites = set(site_index["site_id"].astype(str))
    measurements = []
    audit = []
    for source_row, row in frame.reset_index(drop=True).iterrows():
        if _is_flagged(row.get("Reverse")):
            audit.append({"source_row": source_row, "reason": "reverse"})
            continue
        if _is_flagged(row.get("Contaminant")):
            audit.append({"source_row": source_row, "reason": "contaminant"})
            continue
        localization = _number(row.get("Localization prob"))
        if not np.isfinite(localization):
            audit.append(
                {"source_row": source_row, "reason": "missing_localization_probability"}
            )
            continue
        if localization < 0.75:
            audit.append({"source_row": source_row, "reason": "low_localization"})
            continue
        try:
            pairs = paired_protein_positions(
                row.get("Proteins", ""), row.get("Positions within proteins", "")
            )
        except ValueError:
            audit.append(
                {"source_row": source_row, "reason": "protein_position_count_mismatch"}
            )
            continue

        residue = str(row.get("Amino acid", "")).strip().upper()
        mapped_sites = set()
        for source_accession, raw_position in pairs:
            accession = _canonical_accession(source_accession)
            if accession not in retained_accessions:
                audit.append(
                    {
                        "source_row": source_row,
                        "source_accession": source_accession,
                        "raw_position": raw_position,
                        "reason": "accession_not_retained",
                    }
                )
                continue
            try:
                position = int(float(raw_position))
            except (TypeError, ValueError):
                audit.append(
                    {
                        "source_row": source_row,
                        "source_accession": source_accession,
                        "raw_position": raw_position,
                        "reason": "invalid_position",
                    }
                )
                continue
            site_id = f"{accession}_{residue}{position}"
            if site_id not in retained_sites:
                audit.append(
                    {
                        "source_row": source_row,
                        "source_accession": source_accession,
                        "raw_position": raw_position,
                        "reason": "sequence_or_site_mismatch",
                    }
                )
                continue
            mapped_sites.add(site_id)

        for site_id in sorted(mapped_sites):
            emitted = False
            for experiment in EXPERIMENTS:
                reported_localization = _number(
                    row.get(f"Localization prob {experiment}")
                )
                ratio = _number(row.get(f"Ratio H/L normalized {experiment}"))
                occupancy_l = _number(row.get(f"Occupancy L {experiment}"))
                occupancy_h = _number(row.get(f"Occupancy H {experiment}"))
                intensity = _number(row.get(f"Intensity {experiment}"))
                if np.isfinite(intensity) and intensity <= 0:
                    intensity = np.nan
                if not any(
                    np.isfinite(value)
                    for value in (
                        reported_localization,
                        ratio,
                        occupancy_l,
                        occupancy_h,
                        intensity,
                    )
                ):
                    continue
                experiment_localization = (
                    reported_localization
                    if np.isfinite(reported_localization)
                    else localization
                )
                if experiment_localization < 0.75:
                    audit.append(
                        {
                            "source_row": source_row,
                            "site_id": site_id,
                            "reason": "low_experiment_localization",
                        }
                    )
                    continue
                log_ratio = (
                    float(np.log2(ratio)) if np.isfinite(ratio) and ratio > 0 else np.nan
                )
                occupancy = _mean_available([occupancy_l, occupancy_h])
                log_intensity = (
                    float(np.log1p(intensity))
                    if np.isfinite(intensity) and intensity >= 0
                    else np.nan
                )
                emitted = True
                measurements.append(
                    {
                        "site_id": site_id,
                        "source_row": source_row,
                        "experiment": experiment,
                        "localization_probability": experiment_localization,
                        "log_ratio": log_ratio,
                        "occupancy": occupancy,
                        "log_intensity": log_intensity,
                    }
                )
            if not emitted:
                audit.append(
                    {
                        "source_row": source_row,
                        "site_id": site_id,
                        "reason": "no_experiment_measurement",
                    }
                )
    measurement_columns = [
        "site_id",
        "source_row",
        "experiment",
        "localization_probability",
        "log_ratio",
        "occupancy",
        "log_intensity",
    ]
    audit_columns = [
        "source_row",
        "source_accession",
        "raw_position",
        "site_id",
        "reason",
    ]
    return (
        pd.DataFrame(measurements, columns=measurement_columns),
        pd.DataFrame(audit, columns=audit_columns),
    )


def build_human_features(measurements):
    merged = None
    for experiment in EXPERIMENTS:
        selected = measurements[measurements["experiment"].eq(experiment)]
        if selected.empty:
            continue
        aggregated = aggregate_human_sites(selected)
        aggregated = aggregated.rename(
            columns={
                column: f"experiment_{experiment.lower()}_{column}"
                for column in aggregated.columns
                if column != "site_id"
            }
        )
        merged = (
            aggregated
            if merged is None
            else merged.merge(aggregated, on="site_id", how="outer", validate="one_to_one")
        )
    if merged is None:
        return pd.DataFrame(columns=["site_id", "proteomics_present"])

    detection_columns = [
        f"experiment_{experiment.lower()}_detection_count" for experiment in EXPERIMENTS
    ]
    localization_columns = [
        f"experiment_{experiment.lower()}_localization_max" for experiment in EXPERIMENTS
    ]
    ratio_columns = [
        f"experiment_{experiment.lower()}_log_ratio_mean" for experiment in EXPERIMENTS
    ]
    occupancy_columns = [
        f"experiment_{experiment.lower()}_occupancy_mean" for experiment in EXPERIMENTS
    ]
    intensity_columns = [
        f"experiment_{experiment.lower()}_log_intensity_mean"
        for experiment in EXPERIMENTS
    ]
    for column in (
        detection_columns
        + localization_columns
        + ratio_columns
        + occupancy_columns
        + intensity_columns
    ):
        if column not in merged:
            merged[column] = np.nan
    merged["detection_count_total"] = merged[detection_columns].sum(
        axis=1, min_count=1
    )
    merged["experiments_detected"] = merged[detection_columns].notna().sum(axis=1)
    merged["localization_max"] = merged[localization_columns].max(axis=1)
    merged["log_ratio_across_experiments_mean"] = merged[ratio_columns].mean(axis=1)
    merged["log_ratio_across_experiments_std"] = merged[ratio_columns].std(axis=1)
    merged["occupancy_across_experiments_mean"] = merged[occupancy_columns].mean(axis=1)
    merged["log_intensity_across_experiments_mean"] = merged[intensity_columns].mean(
        axis=1
    )
    merged["proteomics_present"] = np.int8(1)
    return merged.sort_values("site_id").reset_index(drop=True)


def _first_float(value):
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(value))
    return float(match.group(0)) if match else np.nan


def build_mouse_external_evidence(frame):
    rows = []
    for source_row, row in frame.reset_index(drop=True).iterrows():
        accession = _canonical_accession(row.get("UniProt", ""))
        for residue, position in re.findall(r"([STY])(\d+)", str(row.get("Site(s)", ""))):
            rows.append(
                {
                    "mouse_site_id": f"{accession}_{residue}{int(position)}",
                    "mouse_accession": accession,
                    "gene": row.get("Gene Symbol"),
                    "residue": residue,
                    "position": int(position),
                    "log2fc_pka_null": _first_float(
                        row.get("Phospho-site:  log2 (PKA-null / PKA-intact)")
                    ),
                    "p_value": _first_float(row.get("Phospho-site P value")),
                    "total_protein_log2fc": _first_float(
                        row.get("Total protein:  log2 (PKA-null / PKA-intact)")
                    ),
                    "annotation": row.get("Annotation"),
                    "centralized_sequence": row.get("Centralized Sequence"),
                    "source_row": source_row,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "mouse_site_id",
                "mouse_accession",
                "gene",
                "residue",
                "position",
                "evidence_count",
                "log2fc_pka_null",
                "log2fc_pka_null_std",
                "p_value",
                "total_protein_log2fc",
                "annotation",
                "centralized_sequence",
                "source_rows",
            ]
        )
    result = pd.DataFrame(rows)

    def combine_text(values):
        available = sorted(
            {str(value).strip() for value in values if str(value).strip() not in {"", "nan"}}
        )
        return ";".join(available)

    grouped = result.groupby("mouse_site_id", sort=True)
    aggregated = grouped.agg(
        mouse_accession=("mouse_accession", "first"),
        gene=("gene", "first"),
        residue=("residue", "first"),
        position=("position", "first"),
        evidence_count=("mouse_site_id", "size"),
        log2fc_pka_null=("log2fc_pka_null", "mean"),
        log2fc_pka_null_std=("log2fc_pka_null", "std"),
        p_value=("p_value", "min"),
        total_protein_log2fc=("total_protein_log2fc", "mean"),
        annotation=("annotation", combine_text),
        centralized_sequence=("centralized_sequence", "first"),
        source_rows=("source_row", lambda values: ";".join(map(str, sorted(values)))),
    ).reset_index()
    aggregated["log2fc_pka_null_std"] = aggregated[
        "log2fc_pka_null_std"
    ].fillna(0.0)
    return aggregated


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build optional quantitative proteomics features")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = MultimodalConfig.load(args.config)
    config.ensure_output_directories()
    site_index = pd.read_parquet(config.processed_data / "site_index.parquet")
    dynamic_root = config.source_data / "05_dynamic"
    human_path = dynamic_root / "PXD001559_MaxQuant_tables" / "Phospho (STY)Sites.txt"
    human = pd.read_csv(human_path, sep="\t", low_memory=False)
    measurements, mapping_audit = map_human_maxquant(human, site_index)
    human_features = build_human_features(measurements)
    if human_features["site_id"].duplicated().any():
        raise AssertionError("human proteomics features contain duplicate site_id values")
    human_features.to_parquet(
        config.processed_data / "proteomics_features.parquet", index=False
    )
    mapping_audit.to_csv(
        config.processed_data / "proteomics_mapping_audit.csv", index=False
    )

    mouse_path = dynamic_root / "PXD005938_PNAS_supplements" / "PKA-KO_database.xlsx"
    mouse_raw = pd.read_excel(mouse_path, sheet_name="sort by gene symbol")
    mouse_evidence = build_mouse_external_evidence(mouse_raw)
    mouse_evidence.to_parquet(
        config.processed_data / "mouse_external_evidence.parquet", index=False
    )
    exclusion_counts = mapping_audit["reason"].value_counts().sort_index()
    audit = {
        "human_source_rows": int(len(human)),
        "human_mapped_measurements": int(len(measurements)),
        "human_feature_sites": int(len(human_features)),
        "human_exclusion_reasons": {
            str(key): int(value) for key, value in exclusion_counts.items()
        },
        "mouse_external_evidence_rows": int(len(mouse_evidence)),
        "mouse_merged_into_human_features": False,
        "orthology_validation_available": False,
    }
    (config.processed_data / "proteomics_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
