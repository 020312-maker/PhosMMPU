"""Human-only enhanced phosphosite sequence artifacts."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.multimodal.config import MultimodalConfig
from src.multimodal.features.sequence import encode_window
from src.multimodal.site_index import _load_reviewed_fasta


MOTIF_COLUMNS = ("proline_directed", "basophilic", "kinase_profile")


def motif_descriptors(window, center_index):
    """Return label-independent motif scores centered on one phosphoacceptor."""
    window = str(window).upper()
    center = window[center_index]
    downstream = window[center_index + 1 : center_index + 2]
    upstream = window[max(0, center_index - 3) : center_index]
    proline_directed = float(center in "ST" and downstream == "P")
    basophilic = float(sum(residue in "RK" for residue in upstream) >= 2)
    kinase_profile = np.float32(0.6 * proline_directed + 0.4 * basophilic)
    return np.asarray((proline_directed, basophilic, kinase_profile), dtype=np.float32)


def contextual_pool(hidden, center_index):
    """Concatenate global, center, +/-2, and +/-3 hidden-state summaries."""
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 2:
        raise ValueError("hidden must have shape (window, hidden_dim)")
    if center_index < 3 or center_index + 3 >= len(hidden):
        raise ValueError("center_index must have three residues of local context")
    return np.concatenate(
        (
            hidden.max(axis=0),
            hidden[center_index],
            hidden[center_index - 2 : center_index + 3].mean(axis=0),
            hidden[center_index - 3 : center_index + 4].mean(axis=0),
        )
    ).astype(np.float32)


def validate_cache_metadata(expected, actual):
    if dict(expected) != dict(actual):
        raise ValueError("ESM cache metadata does not match the requested source")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_strings(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _site_windows(site_index, sequences, window):
    texts = []
    motifs = []
    center = window // 2
    for row in site_index.itertuples(index=False):
        sequence = sequences.get(row.accession)
        if sequence is None:
            raise ValueError(f"sequence is unavailable for {row.accession}")
        _, _, text = encode_window(
            sequence,
            int(row.position),
            window=window,
            expected_residue=row.residue,
        )
        texts.append(text)
        motifs.append(motif_descriptors(text, center))
    return texts, np.asarray(motifs, dtype=np.float32)


def _esm_embeddings(texts, masks, model_name, batch_size, device, endpoint):
    import torch

    os.environ["HF_ENDPOINT"] = endpoint
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            stop = min(start + batch_size, len(texts))
            # ESM does not encode terminal padding; X is an explicit unknown residue.
            batch = [text.replace("#", "X") for text in texts[start:stop]]
            tokens = tokenizer(batch, return_tensors="pt", padding=True)
            tokens = {name: value.to(device) for name, value in tokens.items()}
            hidden = model(**tokens).last_hidden_state[:, 1:-1].detach().cpu().numpy()
            if hidden.shape[1] != len(texts[0]):
                raise ValueError("ESM residue output does not match the fixed sequence window")
            for representation, residue_mask in zip(hidden, masks[start:stop]):
                valid = residue_mask.astype(bool)
                center = len(valid) // 2
                center_value = representation[center]
                mean_value = representation[valid].mean(axis=0)
                values.append(np.concatenate((center_value, mean_value)).astype(np.float32))
    return np.asarray(values, dtype=np.float32)


def build_sequence_auxiliary(site_index, sequences, window, model_name, batch_size, device, endpoint):
    texts, motifs = _site_windows(site_index, sequences, window)
    masks = np.asarray([[character != "#" for character in text] for text in texts])
    esm = _esm_embeddings(texts, masks, model_name, batch_size, device, endpoint)
    return np.concatenate((esm, motifs), axis=1).astype(np.float32)


def _save_array(path, values):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build ESM and motif sequence features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--esm-model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = MultimodalConfig.load(args.config)
    site_index = pd.read_parquet(config.processed_data / "site_index.parquet")
    site_ids = np.load(config.processed_data / "dataset_site_ids.npy", allow_pickle=False)
    expected_ids = site_index["site_id"].astype(str).to_numpy()
    if not np.array_equal(site_ids, expected_ids):
        raise ValueError("dataset_site_ids does not align with the site index")
    fasta = config.source_data / "02_uniprot" / "uniprot_reviewed_human.fasta.gz"
    metadata = {
        "model": args.esm_model,
        "window": int(config.window),
        "fasta_sha256": _sha256_file(fasta),
        "site_ids_sha256": _sha256_strings(site_ids),
        "hf_endpoint": args.hf_endpoint,
    }
    output = config.processed_data / "sequence_aux.npy"
    metadata_path = config.processed_data / "sequence_esm_cache.json"
    reused = output.exists() and metadata_path.exists() and not args.force
    if reused:
        actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        validate_cache_metadata(metadata, actual)
        values = np.load(output, allow_pickle=False)
    else:
        sequences = _load_reviewed_fasta(fasta)
        values = build_sequence_auxiliary(
            site_index, sequences, config.window, args.esm_model, args.batch_size, args.device,
            args.hf_endpoint,
        )
        _save_array(output, values)
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if len(values) != len(site_index) or not np.isfinite(values).all():
        raise ValueError("sequence auxiliary features are invalid")
    audit = {
        "sites": int(len(values)),
        "feature_shape": list(values.shape),
        "motif_columns": list(MOTIF_COLUMNS),
        "esm_model": args.esm_model,
        "cache_reused": bool(reused),
    }
    (config.processed_data / "sequence_enhanced_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
