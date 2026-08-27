import argparse
import json

import pandas as pd

from src.multimodal.config import MultimodalConfig


LABEL_COLUMNS = ["y_activity", "y_interaction", "y_proteostasis"]


def _contains(series, pattern):
    return series.fillna("").astype(str).str.contains(pattern, case=False, regex=True)


def _direction(text, positive_word, negative_word):
    value = str(text).lower()
    found = [name for name in (positive_word, negative_word) if name in value]
    if len(found) == 1:
        return found[0]
    if len(found) == 2:
        return "mixed"
    return "not_reported"


def _combine_directions(values):
    reported = {value for value in values if value != "not_reported"}
    if not reported:
        return "not_reported"
    if "mixed" in reported or len(reported) > 1:
        return "mixed"
    return next(iter(reported))


def _combine_sources(values):
    sources = {str(value).strip() for value in values if str(value).strip()}
    return ";".join(sorted(sources))


def _derive_row_labels(frame):
    required = {"site_id", "ON_FUNCTION", "ON_PROT_INTERACT", "Database"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    function = frame["ON_FUNCTION"].fillna("").astype(str)
    interaction = frame["ON_PROT_INTERACT"].fillna("").astype(str)
    interaction_present = ~interaction.str.strip().str.lower().isin(
        {"", "-", "nan", "none"}
    )
    result = pd.DataFrame({"site_id": frame["site_id"].astype(str)})
    result["y_activity"] = _contains(function, r"\bactivity\b").astype("int8")
    result["y_interaction"] = (
        interaction_present | _contains(function, r"molecular association")
    ).astype("int8")
    result["y_proteostasis"] = _contains(
        function, r"protein stabilization|protein degradation"
    ).astype("int8")
    result["activity_direction"] = function.map(
        lambda value: _direction(value, "induced", "inhibited")
    )
    result["interaction_direction"] = interaction.map(
        lambda value: _direction(value, "induces", "disrupts")
    )
    result["proteostasis_direction"] = function.map(
        lambda value: _direction(value, "stabilization", "degradation")
    )
    result["label_sources"] = frame["Database"].fillna("").astype(str)
    return result


def build_labels(frame):
    rows = _derive_row_labels(frame)
    labels = rows.groupby("site_id", sort=False, as_index=False)[LABEL_COLUMNS].max()
    for column in (
        "activity_direction",
        "interaction_direction",
        "proteostasis_direction",
    ):
        directions = (
            rows.groupby("site_id", sort=False)[column]
            .agg(_combine_directions)
            .rename(column)
        )
        labels = labels.merge(directions, on="site_id", how="left", validate="one_to_one")
    sources = (
        rows.groupby("site_id", sort=False)["label_sources"]
        .agg(_combine_sources)
        .rename("label_sources")
    )
    labels = labels.merge(sources, on="site_id", how="left", validate="one_to_one")
    labels[LABEL_COLUMNS] = labels[LABEL_COLUMNS].astype("int8")
    return labels


def _validated_source_mapping(index, audit):
    mapping = index[["source_row", "site_id"]].copy()
    duplicates = audit.loc[
        audit["reason"].eq("duplicate_site_id") & audit["site_id"].notna(),
        ["source_row", "site_id"],
    ]
    mapping = pd.concat([mapping, duplicates], ignore_index=True)
    if mapping["source_row"].duplicated().any():
        raise ValueError("source rows map to more than one validated site")
    return mapping


def _positive_counts(frame):
    return {column: int(frame[column].sum()) for column in LABEL_COLUMNS}


def _overlap_counts(labels):
    patterns = labels[LABEL_COLUMNS].astype(str).agg("".join, axis=1).value_counts()
    return {format(value, "03b"): int(patterns.get(format(value, "03b"), 0)) for value in range(8)}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build three PU functional labels")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = MultimodalConfig.load(args.config)
    config.ensure_output_directories()
    raw = pd.read_excel(
        config.source_data / "01_core" / "FuncPhos-SEQ_Phosphosite.xlsx",
        sheet_name="Sheet1",
    )
    raw = raw.reset_index().rename(columns={"index": "source_row"})
    index = pd.read_parquet(config.processed_data / "site_index.parquet")
    mapping_audit = pd.read_csv(config.processed_data / "site_mapping_audit.csv")
    mapping = _validated_source_mapping(index, mapping_audit)
    mapped = mapping.merge(raw, on="source_row", how="left", validate="one_to_one")
    labels = build_labels(mapped)
    if set(labels["site_id"]) != set(index["site_id"]):
        raise ValueError("label output does not cover the validated site index exactly")

    labels_path = config.processed_data / "labels.parquet"
    audit_path = config.processed_data / "label_audit.json"
    labels.to_parquet(labels_path, index=False)

    raw_for_count = raw.copy()
    raw_for_count["site_id"] = raw_for_count["source_row"].astype(str)
    raw_row_labels = _derive_row_labels(raw_for_count)
    audit = {
        "validated_unique_sites": int(len(labels)),
        "mapped_source_rows": int(len(mapped)),
        "raw_row_positive_counts": _positive_counts(raw_row_labels),
        "filtered_unique_site_positive_counts": _positive_counts(labels),
        "overlap_patterns": _overlap_counts(labels),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
