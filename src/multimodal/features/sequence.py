import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.multimodal.config import MultimodalConfig
from src.multimodal.site_index import _load_reviewed_fasta


ALPHABET = "ACDEFGHIKLMNPQRSTVWY#"
INDEX = {residue: index for index, residue in enumerate(ALPHABET)}
STANDARD_AMINO_ACIDS = len(ALPHABET) - 1


def encode_window(sequence, position, window=31, expected_residue=None):
    sequence = str(sequence).upper()
    if window % 2 != 1:
        raise ValueError("window must be odd")
    if position < 1 or position > len(sequence):
        raise ValueError("position outside sequence")
    if expected_residue is not None and sequence[position - 1] != str(
        expected_residue
    ).upper():
        raise ValueError("center residue does not match expected residue")

    half = window // 2
    left = max(0, half - (position - 1))
    right = max(0, half - (len(sequence) - position))
    start = max(0, position - 1 - half)
    stop = min(len(sequence), position + half)
    text = "#" * left + sequence[start:stop] + "#" * right
    if len(text) != window:
        raise ValueError("window construction produced an invalid length")

    encoded = np.zeros((window, len(ALPHABET)), dtype=np.float32)
    mask = np.ones(window, dtype=np.float32)
    for column, residue in enumerate(text):
        if residue in INDEX:
            encoded[column, INDEX[residue]] = 1.0
        else:
            encoded[column, :STANDARD_AMINO_ACIDS] = 1.0 / STANDARD_AMINO_ACIDS
        if residue == "#":
            mask[column] = 0.0
    return encoded, mask, text


def build_sequence_arrays(site_index, sequences, window, feature_path, mask_path):
    feature_path = Path(feature_path)
    mask_path = Path(mask_path)
    feature_temp = feature_path.with_suffix(feature_path.suffix + ".tmp")
    mask_temp = mask_path.with_suffix(mask_path.suffix + ".tmp")
    shape = (len(site_index), window, len(ALPHABET))
    mask_shape = (len(site_index), window)
    features = np.lib.format.open_memmap(
        feature_temp, mode="w+", dtype=np.float32, shape=shape
    )
    masks = np.lib.format.open_memmap(
        mask_temp, mode="w+", dtype=np.float32, shape=mask_shape
    )
    unknown_residues = 0
    padding_positions = 0
    try:
        for output_row, row in enumerate(site_index.itertuples(index=False)):
            sequence = sequences.get(row.accession)
            if sequence is None:
                raise ValueError(f"sequence is unavailable for {row.accession}")
            encoded, mask, text = encode_window(
                sequence,
                position=int(row.position),
                window=window,
                expected_residue=row.residue,
            )
            features[output_row] = encoded
            masks[output_row] = mask
            unknown_residues += sum(
                residue not in INDEX and residue != "#" for residue in text
            )
            padding_positions += int((mask == 0).sum())
        features.flush()
        masks.flush()
    except Exception:
        del features
        del masks
        feature_temp.unlink(missing_ok=True)
        mask_temp.unlink(missing_ok=True)
        raise
    del features
    del masks
    feature_temp.replace(feature_path)
    mask_temp.replace(mask_path)
    return {
        "feature_shape": list(shape),
        "mask_shape": list(mask_shape),
        "unknown_residues": int(unknown_residues),
        "padding_positions": int(padding_positions),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Encode site-centered sequence windows")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = MultimodalConfig.load(args.config)
    config.ensure_output_directories()
    site_index = pd.read_parquet(config.processed_data / "site_index.parquet")
    if site_index["site_id"].duplicated().any():
        raise ValueError("site index contains duplicate site_id values")
    sequences = _load_reviewed_fasta(
        config.source_data / "02_uniprot" / "uniprot_reviewed_human.fasta.gz"
    )
    audit = build_sequence_arrays(
        site_index,
        sequences,
        config.window,
        config.processed_data / "sequence.npy",
        config.processed_data / "sequence_mask.npy",
    )
    site_ids_path = config.processed_data / "sequence_site_ids.npy"
    site_ids_temp = site_ids_path.with_suffix(site_ids_path.suffix + ".tmp")
    requested_ids = np.asarray(site_index["site_id"].astype(str).tolist(), dtype=np.str_)
    with site_ids_temp.open("wb") as handle:
        np.save(handle, requested_ids, allow_pickle=False)
    site_ids_temp.replace(site_ids_path)

    saved_ids = np.load(site_ids_path, allow_pickle=False)
    if not np.array_equal(saved_ids, requested_ids):
        raise AssertionError("saved sequence site order differs from the site index")
    audit.update(
        {
            "sites": int(len(site_index)),
            "window": int(config.window),
            "alphabet": ALPHABET,
            "center_residue_counts": {
                str(key): int(value)
                for key, value in site_index["residue"].value_counts().sort_index().items()
            },
        }
    )
    (config.processed_data / "sequence_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
