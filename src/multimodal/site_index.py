import argparse
import gzip
import json
import re

import pandas as pd

from src.multimodal.config import MultimodalConfig


MOD_RSD_PATTERN = re.compile(r"^([STY])(\d+)-p$")
INDEX_COLUMNS = [
    "source_row",
    "site_id",
    "accession",
    "source_accession",
    "gene",
    "residue",
    "position",
    "sequence_length",
]
AUDIT_COLUMNS = [
    "source_row",
    "reason",
    "source_accession",
    "mod_rsd",
    "canonical_accession",
    "site_id",
]


def parse_mod_rsd(value):
    match = MOD_RSD_PATTERN.fullmatch(str(value).strip())
    if not match:
        raise ValueError(f"invalid MOD_RSD: {value}")
    return match.group(1), int(match.group(2))


def canonical_accession(accession, sequences):
    accession = str(accession).strip()
    if accession in sequences:
        return accession
    base = accession.split("-", 1)[0]
    return base if base in sequences else None


def _audit_record(source_row, row, reason, accession=None, site_id=None):
    return {
        "source_row": source_row,
        "reason": reason,
        "source_accession": str(row.get("ACC_ID", "")),
        "mod_rsd": str(row.get("MOD_RSD", "")),
        "canonical_accession": accession,
        "site_id": site_id,
    }


def build_site_index(frame, sequences):
    missing = {"ACC_ID", "MOD_RSD"} - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    valid = []
    rejected = []
    seen_site_ids = set()
    for source_row, row in frame.reset_index(drop=True).iterrows():
        accession = canonical_accession(row["ACC_ID"], sequences)
        if accession is None:
            rejected.append(_audit_record(source_row, row, "accession_not_found"))
            continue
        try:
            residue, position = parse_mod_rsd(row["MOD_RSD"])
        except ValueError:
            rejected.append(
                _audit_record(source_row, row, "invalid_mod_rsd", accession=accession)
            )
            continue

        sequence = str(sequences[accession]).upper()
        site_id = f"{accession}_{residue}{position}"
        if position < 1 or position > len(sequence):
            rejected.append(
                _audit_record(
                    source_row,
                    row,
                    "position_out_of_range",
                    accession=accession,
                    site_id=site_id,
                )
            )
            continue
        if sequence[position - 1] != residue:
            rejected.append(
                _audit_record(
                    source_row,
                    row,
                    "residue_mismatch",
                    accession=accession,
                    site_id=site_id,
                )
            )
            continue
        if site_id in seen_site_ids:
            rejected.append(
                _audit_record(
                    source_row,
                    row,
                    "duplicate_site_id",
                    accession=accession,
                    site_id=site_id,
                )
            )
            continue

        seen_site_ids.add(site_id)
        valid.append(
            {
                "source_row": source_row,
                "site_id": site_id,
                "accession": accession,
                "source_accession": str(row["ACC_ID"]).strip(),
                "gene": row.get("GENE"),
                "residue": residue,
                "position": position,
                "sequence_length": len(sequence),
            }
        )

    index = pd.DataFrame(valid, columns=INDEX_COLUMNS)
    audit = pd.DataFrame(rejected, columns=AUDIT_COLUMNS)
    return index.reset_index(drop=True), audit.reset_index(drop=True)


def _load_reviewed_fasta(path):
    from Bio import SeqIO

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        sequences = {}
        for record in SeqIO.parse(handle, "fasta"):
            fields = record.id.split("|")
            accession = fields[1] if len(fields) >= 3 else record.id
            sequences[accession] = str(record.seq)
        return sequences


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a validated phosphosite index")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = MultimodalConfig.load(args.config)
    config.ensure_output_directories()
    frame = pd.read_excel(
        config.source_data / "01_core" / "FuncPhos-SEQ_Phosphosite.xlsx",
        sheet_name="Sheet1",
    )
    sequences = _load_reviewed_fasta(
        config.source_data / "02_uniprot" / "uniprot_reviewed_human.fasta.gz"
    )
    index, audit = build_site_index(frame, sequences)
    index_path = config.processed_data / "site_index.parquet"
    audit_path = config.processed_data / "site_mapping_audit.csv"
    summary_path = config.processed_data / "site_mapping_audit.json"
    index.to_parquet(index_path, index=False)
    audit.to_csv(audit_path, index=False)
    summary = {
        "source_rows": int(len(frame)),
        "valid_unique_sites": int(len(index)),
        "rejected_rows": int(len(audit)),
        "rejection_reasons": {
            str(key): int(value)
            for key, value in audit["reason"].value_counts().sort_index().items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
