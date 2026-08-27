"""Audit and filter site-level direct evidence before building SIGNOR graphs."""

from __future__ import annotations

import re

import pandas as pd


DIRECT_TRUE = {"t", "true", "1", "yes"}
AUDIT_COLUMNS = (
    "SIGNOR_ID",
    "PMID",
    "IDA",
    "IDB",
    "RESIDUE",
    "EFFECT",
    "MECHANISM",
    "DIRECT",
    "site_id",
    "split",
)
_RESIDUE = re.compile(r"\b(Ser|Thr|Tyr|S|T|Y)\s*(\d+)\b", flags=re.IGNORECASE)
_RESIDUE_NAMES = {"ser": "S", "thr": "T", "tyr": "Y", "s": "S", "t": "T", "y": "Y"}


def _uniprot_accession(value):
    value = str(value).strip()
    if not value or value.lower() in {"-", "nan", "none"}:
        return None
    return value.split("-", 1)[0]


def _parse_residue(value):
    match = _RESIDUE.search(str(value))
    if match is None:
        return None, None
    return _RESIDUE_NAMES[match.group(1).lower()], int(match.group(2))


def filter_direct_signor_evidence(signor: pd.DataFrame, sites: pd.DataFrame):
    """Remove direct SIGNOR rows matching validation/test phosphosites.

    ``sites`` must contain a unique split assignment for each phosphosite. The
    returned retained frame preserves the original SIGNOR columns so callers
    can derive graph edges without losing provenance before the audit is saved.
    """
    required_signor = {"SIGNOR_ID", "IDA", "IDB", "RESIDUE", "DIRECT"}
    required_sites = {"site_id", "accession", "residue", "position", "split"}
    missing_signor = required_signor.difference(signor.columns)
    missing_sites = required_sites.difference(sites.columns)
    if missing_signor:
        raise ValueError(f"SIGNOR records missing columns: {sorted(missing_signor)}")
    if missing_sites:
        raise ValueError(f"site records missing columns: {sorted(missing_sites)}")

    frame = signor.copy().reset_index(drop=True)
    frame["_record_index"] = frame.index
    frame["_target_accession"] = frame["IDB"].map(_uniprot_accession)
    parsed = frame["RESIDUE"].map(_parse_residue)
    frame["_target_residue"] = [value[0] for value in parsed]
    frame["_target_position"] = [value[1] for value in parsed]
    frame["_is_direct"] = frame["DIRECT"].fillna("").astype(str).str.strip().str.lower().isin(DIRECT_TRUE)

    targets = sites.loc[sites["split"].isin(("validation", "test")), [
        "site_id", "accession", "residue", "position", "split"
    ]].copy()
    targets["accession"] = targets["accession"].astype(str)
    targets["residue"] = targets["residue"].astype(str).str.upper()
    targets["position"] = pd.to_numeric(targets["position"], errors="raise").astype(int)
    if targets.duplicated(["accession", "residue", "position"]).any():
        raise ValueError("validation/test target phosphosites are not unique")

    matched = frame.merge(
        targets,
        left_on=["_target_accession", "_target_residue", "_target_position"],
        right_on=["accession", "residue", "position"],
        how="left",
        validate="many_to_one",
    )
    remove = matched["_is_direct"] & matched["site_id"].notna()
    columns = [column for column in AUDIT_COLUMNS if column in matched.columns]
    removed = matched.loc[remove, columns].copy()
    removed["removal_reason"] = "direct_validation_or_test_site_evidence"
    removed = removed.sort_values(["split", "site_id", "SIGNOR_ID"], kind="mergesort").reset_index(drop=True)

    deleted_indices = set(matched.loc[remove, "_record_index"].astype(int))
    retained = signor.loc[~signor.index.isin(deleted_indices)].copy().reset_index(drop=True)
    return retained, removed
