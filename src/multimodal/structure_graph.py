"""Convert local AlphaFold residue neighborhoods into fixed-size geometric graphs."""

import argparse
import gc
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.multimodal.config import MultimodalConfig
from src.multimodal.features.sequence import ALPHABET, INDEX
from src.multimodal.features.structure import (
    _select_fragment_number,
    index_alphafold_members,
    parse_alphafold_member,
)


SCALAR_DIM = 28


def build_local_graph(
    coordinates,
    residues,
    positions,
    plddt,
    relative_sasa,
    secondary_structure,
    center,
    sequence_length,
    k=24,
):
    """Return fixed-K scalar nodes, direction vectors, position ids, and a node mask."""
    coordinates = np.asarray(coordinates, dtype=np.float32)
    residues = np.asarray(residues).astype("U1")
    positions = np.asarray(positions, dtype=np.int32)
    plddt = np.asarray(plddt, dtype=np.float32)
    relative_sasa = np.asarray(relative_sasa, dtype=np.float32)
    secondary_structure = np.asarray(secondary_structure).astype("U1")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (residues, 3)")
    if not 0 <= int(center) < len(coordinates):
        raise ValueError("center is outside coordinates")
    if k <= 0:
        raise ValueError("k must be positive")
    scalar = np.zeros((k, SCALAR_DIM), dtype=np.float32)
    vectors = np.zeros((k, 3), dtype=np.float32)
    neighbors = np.full(k, -1, dtype=np.int64)
    mask = np.zeros(k, dtype=np.float32)
    distances = np.linalg.norm(coordinates - coordinates[int(center)], axis=1)
    selected = np.argsort(distances, kind="stable")[:k]
    count = len(selected)
    displacement = coordinates[selected] - coordinates[int(center)]
    # Keep both direction and radial distance for the GVP. A unit vector would
    # erase the local geometry that distinguishes close from distant residues.
    vectors[:count] = displacement / 20.0
    for output_index, source_index in enumerate(selected):
        residue = residues[source_index]
        if residue in INDEX:
            scalar[output_index, INDEX[residue]] = 1.0
        scalar[output_index, 21] = np.clip(plddt[source_index] / 100.0, 0.0, 1.0)
        scalar[output_index, 22] = (
            positions[source_index] - positions[int(center)]
        ) / max(int(sequence_length), 1)
        scalar[output_index, 23] = np.clip(relative_sasa[source_index], 0.0, 1.0)
        secondary = secondary_structure[source_index]
        scalar[output_index, 24:27] = (
            float(secondary == "H"),
            float(secondary == "E"),
            float(secondary not in {"H", "E"}),
        )
        scalar[output_index, 27] = float(source_index == int(center))
        neighbors[output_index] = int(source_index)
        mask[output_index] = 1.0
    return scalar, vectors, neighbors, mask


def _open_output(path, shape, dtype):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    return temporary, np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)


def build_structure_graph_artifacts(site_index, archive_path, output_dir, neighbors=24):
    """Write one local graph row per validated site without dropping missing structures."""
    output_dir = Path(output_dir)
    rows = len(site_index)
    scalar_path, scalar_values = _open_output(
        output_dir / "structure_graph_scalars.npy", (rows, neighbors, SCALAR_DIM), np.float32
    )
    vector_path, vector_values = _open_output(
        output_dir / "structure_graph_vectors.npy", (rows, neighbors, 3), np.float32
    )
    neighbor_path, neighbor_values = _open_output(
        output_dir / "structure_graph_neighbors.npy", (rows, neighbors), np.int64
    )
    mask_path, mask_values = _open_output(
        output_dir / "structure_graph_mask.npy", (rows, neighbors), np.float32
    )
    neighbor_values[:] = -1
    mapped = 0
    missing = {}
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            member_index = index_alphafold_members(archive)
            for group_number, (accession, sites) in enumerate(
                site_index.groupby("accession", sort=False), start=1
            ):
                member_entries = member_index.get(str(accession), [])
                member_by_number = dict(member_entries)
                fragments = {}
                parse_errors = set()
                for row_index, site in sites.iterrows():
                    fragment_number = _select_fragment_number(
                        member_entries, int(site.position), int(site.sequence_length)
                    )
                    if fragment_number is None:
                        missing["position_absent"] = missing.get("position_absent", 0) + 1
                        continue
                    if fragment_number not in fragments and fragment_number not in parse_errors:
                        try:
                            fragments[fragment_number] = parse_alphafold_member(
                                archive,
                                member_by_number[fragment_number],
                                fragment_number,
                                compute_sasa=False,
                            )
                        except Exception:
                            parse_errors.add(fragment_number)
                    fragment = fragments.get(fragment_number)
                    if fragment is None:
                        missing["fragment_parse_error"] = missing.get("fragment_parse_error", 0) + 1
                        continue
                    context = fragment.context
                    target = context.index_by_position.get(int(site.position))
                    if target is None or context.residues[target] != str(site.residue):
                        reason = "residue_mismatch" if target is not None else "position_absent"
                        missing[reason] = missing.get(reason, 0) + 1
                        continue
                    scalar, vectors, graph_neighbors, graph_mask = build_local_graph(
                        context.coordinates,
                        context.residues,
                        context.positions,
                        context.plddt,
                        context.relative_sasa,
                        context.secondary_structure,
                        target,
                        int(site.sequence_length),
                        neighbors,
                    )
                    scalar_values[row_index] = scalar
                    vector_values[row_index] = vectors
                    neighbor_values[row_index] = graph_neighbors
                    mask_values[row_index] = graph_mask
                    mapped += 1
                if group_number % 25 == 0:
                    print(f"processed {group_number} AlphaFold accessions", flush=True)
        for values in (scalar_values, vector_values, neighbor_values, mask_values):
            values.flush()
        del values
    except Exception:
        for path in (scalar_path, vector_path, neighbor_path, mask_path):
            path.unlink(missing_ok=True)
        raise
    finally:
        del scalar_values, vector_values, neighbor_values, mask_values
        gc.collect()
    destinations = (
        output_dir / "structure_graph_scalars.npy",
        output_dir / "structure_graph_vectors.npy",
        output_dir / "structure_graph_neighbors.npy",
        output_dir / "structure_graph_mask.npy",
    )
    for temporary, destination in zip(
        (scalar_path, vector_path, neighbor_path, mask_path), destinations
    ):
        temporary.replace(destination)
    return {
        "sites": int(rows),
        "mapped_sites": int(mapped),
        "neighbors": int(neighbors),
        "scalar_dim": SCALAR_DIM,
        "accessibility_feature": "ca_contact_density_proxy",
        "missing_reasons": missing,
    }


def finalize_existing_artifacts(output_dir, neighbors=24):
    """Recover a fully written graph if Windows interrupted only the final rename."""
    output_dir = Path(output_dir)
    destinations = (
        output_dir / "structure_graph_scalars.npy",
        output_dir / "structure_graph_vectors.npy",
        output_dir / "structure_graph_neighbors.npy",
        output_dir / "structure_graph_mask.npy",
    )
    mask_temporary = destinations[-1].with_suffix(".npy.tmp")
    if mask_temporary.is_file():
        mask_temporary.replace(destinations[-1])
    if not all(path.is_file() for path in destinations):
        raise FileNotFoundError("complete structure graph artifacts are not available")
    scalar = np.load(destinations[0], mmap_mode="r")
    mask = np.load(destinations[-1], mmap_mode="r")
    if scalar.ndim != 3 or mask.shape != scalar.shape[:2] or scalar.shape[1] != neighbors:
        raise ValueError("recovered structure graph arrays have inconsistent dimensions")
    return {
        "sites": int(len(mask)),
        "mapped_sites": int(mask.any(axis=1).sum()),
        "neighbors": int(neighbors),
        "scalar_dim": SCALAR_DIM,
        "accessibility_feature": "ca_contact_density_proxy",
        "recovered_after_windows_rename_lock": True,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build local AlphaFold geometric graphs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--neighbors", type=int, default=24)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args(argv)
    config = MultimodalConfig.load(args.config)
    if args.finalize_existing:
        audit = finalize_existing_artifacts(config.processed_data, args.neighbors)
        (config.processed_data / "structure_graph_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        print(json.dumps(audit, indent=2))
        return
    site_index = pd.read_parquet(config.processed_data / "site_index.parquet")
    site_ids = np.load(config.processed_data / "dataset_site_ids.npy", allow_pickle=False)
    if not np.array_equal(site_ids, site_index["site_id"].astype(str).to_numpy()):
        raise ValueError("dataset_site_ids does not align with the site index")
    if args.limit:
        site_index = site_index.head(args.limit).copy()
    archive = config.source_data / "06_structures" / "UP000005640_9606_HUMAN_v6.tar"
    audit = build_structure_graph_artifacts(site_index, archive, config.processed_data, args.neighbors)
    (config.processed_data / "structure_graph_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
