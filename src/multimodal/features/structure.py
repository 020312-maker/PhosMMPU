import argparse
import gzip
import io
import json
import math
import re
import tarfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.multimodal.config import MultimodalConfig


HYDROPHOBIC = set("AILMFWVY")
CHARGED = set("DEKR")
POLAR = set("STNQH")
ALPHAFOLD_MEMBER_PATTERN = re.compile(
    r"(?:^|/)AF-([^-]+)-F(\d+)-model_v6\.cif\.gz$"
)
STRUCTURE_FEATURE_COLUMNS = [
    "target_plddt",
    "local_mean_plddt",
    "local_min_plddt",
    "relative_sasa",
    "contacts_within_8a",
    "contacts_within_12a",
    "hydrophobic_fraction_8a",
    "charged_fraction_8a",
    "polar_fraction_8a",
    "secondary_helix",
    "secondary_strand",
    "secondary_coil",
    "normalized_position",
    "confident_plddt_ge_70",
    "accessible_sasa_ge_025",
    "isolated_within_8a",
]

# Maximum solvent-accessible areas from Tien et al. (A^2).
MAX_ACCESSIBILITY = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}


@dataclass(frozen=True)
class ResidueRecord:
    position: int
    residue: str
    ca: np.ndarray
    plddt: float
    relative_sasa: float
    secondary_structure: str


@dataclass(frozen=True)
class ParsedFragment:
    fragment: int
    begin: int
    end: int
    records: tuple
    context: object


@dataclass(frozen=True)
class PreparedStructure:
    positions: np.ndarray
    residues: np.ndarray
    coordinates: np.ndarray
    plddt: np.ndarray
    relative_sasa: np.ndarray
    secondary_structure: np.ndarray
    hydrophobic: np.ndarray
    charged: np.ndarray
    polar: np.ndarray
    index_by_position: dict


def _fraction(records, residue_set):
    return sum(record.residue in residue_set for record in records) / max(
        len(records), 1
    )


def prepare_structure(records):
    records = tuple(records)
    positions = np.asarray([record.position for record in records], dtype=np.int32)
    residues = np.asarray([record.residue for record in records], dtype="U1")
    coordinates = np.asarray([record.ca for record in records], dtype=np.float32)
    if coordinates.size == 0:
        coordinates = np.empty((0, 3), dtype=np.float32)
    return PreparedStructure(
        positions=positions,
        residues=residues,
        coordinates=coordinates,
        plddt=np.asarray([record.plddt for record in records], dtype=np.float32),
        relative_sasa=np.asarray(
            [record.relative_sasa for record in records], dtype=np.float32
        ),
        secondary_structure=np.asarray(
            [record.secondary_structure for record in records], dtype="U1"
        ),
        hydrophobic=np.isin(residues, list(HYDROPHOBIC)),
        charged=np.isin(residues, list(CHARGED)),
        polar=np.isin(residues, list(POLAR)),
        index_by_position={record.position: index for index, record in enumerate(records)},
    )


def vectorize_prepared(context, position, sequence_length=None):
    if position not in context.index_by_position:
        raise KeyError(f"position {position} is absent from structure")
    target_index = context.index_by_position[position]
    distances = np.linalg.norm(
        context.coordinates - context.coordinates[target_index], axis=1
    )
    near8 = (distances > 0) & (distances <= 8.0)
    near12 = (distances > 0) & (distances <= 12.0)
    local = np.abs(context.positions - position) <= 7
    near8_count = int(near8.sum())
    near12_count = int(near12.sum())
    secondary = context.secondary_structure[target_index]
    denominator = sequence_length or int(context.positions.max())

    def masked_fraction(values, mask):
        return float(values[mask].mean()) if mask.any() else 0.0

    vector = np.array(
        [
            context.plddt[target_index] / 100.0,
            context.plddt[local].mean() / 100.0,
            context.plddt[local].min() / 100.0,
            context.relative_sasa[target_index],
            min(near8_count, 64) / 64.0,
            min(near12_count, 128) / 128.0,
            masked_fraction(context.hydrophobic, near8),
            masked_fraction(context.charged, near8),
            masked_fraction(context.polar, near8),
            float(secondary == "H"),
            float(secondary == "E"),
            float(secondary not in {"H", "E"}),
            position / denominator,
            float(context.plddt[target_index] >= 70.0),
            float(context.relative_sasa[target_index] >= 0.25),
            float(near8_count == 0),
        ],
        dtype=np.float32,
    )
    quality = {
        "mapping_status": "matched",
        "target_plddt": float(context.plddt[target_index]),
        "low_confidence": bool(context.plddt[target_index] < 70.0),
    }
    return vector, quality


def vectorize_structure(records, position, sequence_length=None):
    return vectorize_prepared(
        prepare_structure(records), position, sequence_length=sequence_length
    )


def index_alphafold_members(archive):
    members = {}
    for member in archive.getmembers():
        match = ALPHAFOLD_MEMBER_PATTERN.fullmatch(member.name)
        if not match:
            continue
        accession, fragment = match.group(1), int(match.group(2))
        members.setdefault(accession, []).append((fragment, member))
    for accession in members:
        members[accession].sort(key=lambda item: item[0])
    return members


def _scalar(value):
    if isinstance(value, list):
        return value[0]
    return value


def _secondary_structure(phi, psi):
    if phi is None or psi is None:
        return "C"
    phi = math.degrees(phi)
    psi = math.degrees(psi)
    if -160 <= phi <= -30 and -90 <= psi <= 45:
        return "H"
    if -180 <= phi <= -40 and 90 <= psi <= 180:
        return "E"
    return "C"


def parse_alphafold_member(archive, member, fragment, compute_sasa=True):
    from Bio.PDB import MMCIFParser, PPBuilder
    from Bio.PDB.SASA import ShrakeRupley
    from Bio.SeqUtils import seq1

    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"unable to extract {member.name}")
    with extracted:
        compressed = extracted.read()
    text = gzip.decompress(compressed).decode("utf-8")
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(member.name, io.StringIO(text))
    details = parser._mmcif_dict
    begin = int(_scalar(details["_ma_target_ref_db_details.seq_db_align_begin"]))
    end = int(_scalar(details["_ma_target_ref_db_details.seq_db_align_end"]))
    if compute_sasa:
        ShrakeRupley().compute(structure, level="R")

    angles = {}
    for peptide in PPBuilder().build_peptides(structure):
        for residue, (phi, psi) in zip(peptide, peptide.get_phi_psi_list()):
            angles[id(residue)] = (phi, psi)

    raw_records = []
    for residue in structure.get_residues():
        if residue.id[0] != " " or "CA" not in residue:
            continue
        local_position = int(residue.id[1])
        global_position = begin + local_position - 1
        residue_name = residue.get_resname().upper()
        amino_acid = seq1(residue_name, undef_code="X")
        maximum = MAX_ACCESSIBILITY.get(residue_name)
        relative_sasa = 0.0
        if compute_sasa and maximum:
            relative_sasa = float(np.clip(getattr(residue, "sasa", 0.0) / maximum, 0, 1))
        phi, psi = angles.get(id(residue), (None, None))
        raw_records.append(
            (
                global_position,
                amino_acid,
                np.asarray(residue["CA"].coord, dtype=np.float32),
                float(residue["CA"].get_bfactor()),
                relative_sasa,
                _secondary_structure(phi, psi),
            )
        )
    if not raw_records:
        raise ValueError(f"no C-alpha residue records in {member.name}")
    if not compute_sasa:
        # Exact SASA is prohibitively slow for all 3,141 structures. The local
        # graph therefore uses a bounded C-alpha contact-density proxy instead.
        from scipy.spatial import cKDTree

        coordinates = np.stack([record[2] for record in raw_records])
        contact_counts = np.asarray(
            [len(neighbors) - 1 for neighbors in cKDTree(coordinates).query_ball_point(coordinates, 10.0)],
            dtype=np.float32,
        )
        accessibility = np.exp(-contact_counts / 8.0)
    else:
        accessibility = np.asarray([record[4] for record in raw_records], dtype=np.float32)
    records = tuple(
        ResidueRecord(
            position=position,
            residue=residue,
            ca=coordinate,
            plddt=plddt,
            relative_sasa=float(accessibility[index]),
            secondary_structure=secondary_structure,
        )
        for index, (position, residue, coordinate, plddt, _, secondary_structure) in enumerate(raw_records)
    )
    return ParsedFragment(fragment, begin, end, records, prepare_structure(records))


def _select_fragment(fragments, position):
    candidates = [item for item in fragments if item.begin <= position <= item.end]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (min(position - item.begin, item.end - position), -item.fragment),
    )


def _select_fragment_number(member_entries, position, sequence_length):
    if len(member_entries) == 1:
        fragment = member_entries[0][0]
        return fragment if 1 <= position <= sequence_length else None
    candidates = []
    for fragment, _ in member_entries:
        begin = 1 + 200 * (fragment - 1)
        end = min(begin + 1399, sequence_length)
        if begin <= position <= end:
            margin = min(position - begin, end - position)
            candidates.append((margin, -fragment, fragment))
    return max(candidates)[2] if candidates else None


def _missing_row(site, status, detail=""):
    row = {
        "source_order": int(site.source_order),
        "site_id": site.site_id,
        "accession": site.accession,
        "structure_present": np.int8(0),
        "mapping_status": status,
        "fragment": np.nan,
        "target_plddt_raw": np.nan,
        "low_confidence": False,
        "audit_detail": detail,
    }
    row.update({column: np.float32(0.0) for column in STRUCTURE_FEATURE_COLUMNS})
    return row


def _extract_accession_rows(accession, sites, archive, member_index):
    rows = []
    member_entries = member_index.get(accession)
    if not member_entries:
        return [
            _missing_row(site, "missing_accession")
            for site in sites.itertuples(index=False)
        ]
    member_by_number = dict(member_entries)
    selection = {}
    needed_fragments = set()
    for site in sites.itertuples(index=False):
        fragment_number = _select_fragment_number(
            member_entries,
            int(site.position),
            int(site.sequence_length),
        )
        selection[int(site.source_order)] = fragment_number
        if fragment_number is not None:
            needed_fragments.add(fragment_number)

    fragments = {}
    parse_errors = {}
    for fragment_number in sorted(needed_fragments):
        try:
            fragments[fragment_number] = parse_alphafold_member(
                archive, member_by_number[fragment_number], fragment_number
            )
        except Exception as error:
            parse_errors[fragment_number] = (
                f"F{fragment_number}:{type(error).__name__}:{error}"
            )
    for site in sites.itertuples(index=False):
        fragment_number = selection[int(site.source_order)]
        if fragment_number is None:
            rows.append(_missing_row(site, "position_absent"))
            continue
        selected = fragments.get(fragment_number)
        if selected is None:
            rows.append(
                _missing_row(
                    site,
                    "fragment_parse_error",
                    parse_errors.get(fragment_number, "fragment was not parsed"),
                )
            )
            continue
        target_index = selected.context.index_by_position.get(int(site.position))
        if target_index is None:
            rows.append(_missing_row(site, "position_absent"))
            continue
        observed_residue = selected.context.residues[target_index]
        if observed_residue != site.residue:
            rows.append(
                _missing_row(
                    site,
                    "residue_mismatch",
                    f"expected={site.residue};observed={observed_residue}",
                )
            )
            continue
        vector, quality = vectorize_prepared(
            selected.context,
            int(site.position),
            sequence_length=int(site.sequence_length),
        )
        row = {
            "source_order": int(site.source_order),
            "site_id": site.site_id,
            "accession": site.accession,
            "structure_present": np.int8(1),
            "mapping_status": quality["mapping_status"],
            "fragment": int(selected.fragment),
            "target_plddt_raw": quality["target_plddt"],
            "low_confidence": quality["low_confidence"],
            "audit_detail": "",
        }
        row.update(
            {
                column: value
                for column, value in zip(STRUCTURE_FEATURE_COLUMNS, vector)
            }
        )
        rows.append(row)
    return rows


_WORKER_ARCHIVE = None
_WORKER_MEMBER_INDEX = None


def _initialize_structure_worker(archive_path):
    global _WORKER_ARCHIVE, _WORKER_MEMBER_INDEX
    _WORKER_ARCHIVE = tarfile.open(archive_path, mode="r:")
    _WORKER_MEMBER_INDEX = index_alphafold_members(_WORKER_ARCHIVE)


def _extract_accession_worker(payload):
    accession, sites = payload
    return _extract_accession_rows(
        accession, sites, _WORKER_ARCHIVE, _WORKER_MEMBER_INDEX
    )


def extract_structure_features(site_index, archive_path, limit=None, workers=1):
    requested = site_index.head(limit).copy() if limit else site_index.copy()
    requested = requested.reset_index(drop=True)
    requested.index.name = "source_order"
    requested = requested.reset_index()
    rows = []
    with tarfile.open(archive_path, mode="r:") as archive:
        member_index = index_alphafold_members(archive)
        accessions = set(requested["accession"].astype(str))
        accession_coverage = len(accessions & set(member_index)) / max(len(accessions), 1)
        grouped = [
            (str(accession), sites.copy())
            for accession, sites in requested.groupby("accession", sort=False)
        ]
        if workers <= 1 or len(grouped) <= 1:
            for group_number, (accession, sites) in enumerate(grouped, start=1):
                rows.extend(
                    _extract_accession_rows(accession, sites, archive, member_index)
                )
                if group_number % 25 == 0:
                    print(
                        f"processed {group_number}/{len(grouped)} AlphaFold accessions",
                        flush=True,
                    )
        else:
            from concurrent.futures import ProcessPoolExecutor

            worker_count = min(int(workers), len(grouped))
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_initialize_structure_worker,
                initargs=(str(archive_path),),
            ) as executor:
                results = executor.map(_extract_accession_worker, grouped, chunksize=1)
                for group_number, accession_rows in enumerate(results, start=1):
                    rows.extend(accession_rows)
                    if group_number % 25 == 0:
                        print(
                            f"processed {group_number}/{len(grouped)} AlphaFold accessions",
                            flush=True,
                        )

    output = pd.DataFrame(rows).sort_values("source_order").reset_index(drop=True)
    if len(output) != len(requested):
        raise AssertionError("structure extraction silently lost requested sites")
    if output["site_id"].duplicated().any():
        raise AssertionError("structure extraction produced duplicate site_id values")
    values = output[STRUCTURE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise AssertionError("structure feature matrix contains non-finite values")
    if not np.array_equal(
        output["site_id"].astype(str).to_numpy(),
        requested["site_id"].astype(str).to_numpy(),
    ):
        raise AssertionError("structure feature order differs from site index order")
    output = output.drop(columns="source_order")
    return output, accession_coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extract local AlphaFold features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    config = MultimodalConfig.load(args.config)
    config.ensure_output_directories()
    site_index = pd.read_parquet(config.processed_data / "site_index.parquet")
    archive_path = (
        config.source_data
        / "06_structures"
        / "UP000005640_9606_HUMAN_v6.tar"
    )
    features, accession_coverage = extract_structure_features(
        site_index, archive_path, args.limit, workers=args.workers
    )
    features.to_parquet(
        config.processed_data / "structure_features.parquet", index=False
    )
    features[
        [
            "site_id",
            "accession",
            "mapping_status",
            "fragment",
            "target_plddt_raw",
            "low_confidence",
            "audit_detail",
        ]
    ].to_csv(config.processed_data / "structure_mapping_audit.csv", index=False)
    status_counts = features["mapping_status"].value_counts().sort_index()
    audit = {
        "requested_sites": int(len(features)),
        "mapped_sites": int(features["structure_present"].sum()),
        "mapped_site_fraction": float(features["structure_present"].mean()),
        "requested_accession_archive_coverage": float(accession_coverage),
        "low_confidence_sites": int(
            (features["structure_present"].eq(1) & features["low_confidence"]).sum()
        ),
        "mapping_status_counts": {
            str(key): int(value) for key, value in status_counts.items()
        },
        "feature_columns": STRUCTURE_FEATURE_COLUMNS,
    }
    (config.processed_data / "structure_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
