import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.multimodal.config import MultimodalConfig
from src.multimodal.relation_graph import (
    RELATION_NAMES,
    build_relation_graph,
    sample_relation_neighbors,
)
from src.multimodal.network_evidence import filter_direct_signor_evidence


TOPOLOGY_COLUMNS = [
    "degree",
    "weighted_degree",
    "high_confidence_degree",
    "isolated",
]

NETWORK_VARIANTS = (
    "complete",
    "direct_evidence_masked",
    "without_signor",
    "ordinary_ppi",
    "without_network",
)


def _canonical_pairs(edges, index):
    selected = edges[
        edges["source"].isin(index)
        & edges["target"].isin(index)
        & edges["source"].ne(edges["target"])
    ].copy()
    if selected.empty:
        return selected
    source_index = selected["source"].map(index).to_numpy()
    target_index = selected["target"].map(index).to_numpy()
    swap = source_index > target_index
    selected.loc[swap, ["source", "target"]] = selected.loc[
        swap, ["target", "source"]
    ].to_numpy()
    return (
        selected.groupby(["source", "target"], as_index=False, sort=True)["weight"]
        .max()
        .reset_index(drop=True)
    )


def compute_graph_features(edges, accessions, embedding_dim=32, seed=0):
    if embedding_dim < 0:
        raise ValueError("embedding_dim must be non-negative")
    accessions = list(dict.fromkeys(str(value) for value in accessions))
    index = {accession: position for position, accession in enumerate(accessions)}
    selected = _canonical_pairs(edges, index)
    row = selected["source"].map(index).to_numpy(dtype=np.int64)
    col = selected["target"].map(index).to_numpy(dtype=np.int64)
    weight = selected["weight"].astype(float).to_numpy()
    adjacency = sparse.coo_matrix(
        (weight, (row, col)), shape=(len(accessions), len(accessions))
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    degree = np.asarray((adjacency > 0).sum(axis=1)).ravel()
    weighted_degree = np.asarray(adjacency.sum(axis=1)).ravel()
    high_confidence = np.asarray((adjacency >= 0.7).sum(axis=1)).ravel()

    if embedding_dim:
        components = min(embedding_dim, max(1, len(accessions) - 1))
        if adjacency.nnz == 0 or len(accessions) == 1:
            embedding = np.zeros((len(accessions), components), dtype=np.float32)
        else:
            embedding = TruncatedSVD(
                n_components=components, random_state=seed
            ).fit_transform(adjacency)
        if components < embedding_dim:
            embedding = np.pad(
                embedding, ((0, 0), (0, embedding_dim - components))
            )
    else:
        embedding = np.empty((len(accessions), 0), dtype=np.float32)

    result = pd.DataFrame(
        {
            "accession": accessions,
            "degree": degree,
            "weighted_degree": weighted_degree,
            "high_confidence_degree": high_confidence,
            "isolated": (degree == 0).astype(np.int8),
        }
    )
    for column in range(embedding_dim):
        result[f"embedding_{column:02d}"] = embedding[:, column]
    return result


def _uniprot_accession(value):
    value = str(value).strip()
    if not value or value in {"-", "nan", "None"}:
        return None
    return value.split("-", 1)[0]


def read_string_aliases(path, retained_accessions):
    retained = set(retained_accessions)
    aliases = pd.read_csv(path, sep="\t", compression="gzip", dtype=str)
    aliases = aliases[aliases["source"].str.contains("UniProt", na=False)].copy()
    aliases["accession"] = aliases["alias"].map(_uniprot_accession)
    aliases = aliases[aliases["accession"].isin(retained)]
    aliases = aliases.rename(columns={"#string_protein_id": "string_protein_id"})
    aliases = aliases.sort_values(
        ["string_protein_id", "source", "accession"], kind="mergesort"
    )
    return (
        aliases.drop_duplicates("string_protein_id", keep="first")
        .set_index("string_protein_id")["accession"]
        .to_dict()
    )


def read_string_edges(path, alias_map, chunksize=1_000_000):
    parts = []
    for chunk in pd.read_csv(
        path,
        sep=r"\s+",
        compression="gzip",
        dtype={"protein1": str, "protein2": str, "combined_score": np.int32},
        chunksize=chunksize,
    ):
        chunk["source"] = chunk["protein1"].map(alias_map)
        chunk["target"] = chunk["protein2"].map(alias_map)
        selected = chunk[
            chunk["source"].notna()
            & chunk["target"].notna()
            & chunk["source"].ne(chunk["target"])
        ]
        if selected.empty:
            continue
        parts.append(
            pd.DataFrame(
                {
                    "source": selected["source"],
                    "target": selected["target"],
                    "weight": selected["combined_score"].astype(float) / 1000.0,
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["source", "target", "weight"])
    return pd.concat(parts, ignore_index=True)


def _first_retained_accession(value, retained):
    for token in re.split(r"[|;,]", str(value)):
        accession = _uniprot_accession(token)
        if accession in retained:
            return accession
    return None


def read_biogrid_edges(path, retained_accessions, chunksize=100_000):
    retained = set(retained_accessions)
    parts = []
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith(".tab3.txt")]
        if len(members) != 1:
            raise ValueError("BioGRID physical ZIP must contain one tab3 file")
        with archive.open(members[0]) as binary:
            handle = io.TextIOWrapper(binary, encoding="utf-8", errors="replace")
            reader = pd.read_csv(
                handle,
                sep="\t",
                dtype=str,
                chunksize=chunksize,
                low_memory=False,
            )
            for chunk in reader:
                human = chunk[
                    chunk["Organism ID Interactor A"].eq("9606")
                    & chunk["Organism ID Interactor B"].eq("9606")
                    & chunk["Experimental System Type"].str.lower().eq("physical")
                ].copy()
                if human.empty:
                    continue
                human["source"] = human["SWISS-PROT Accessions Interactor A"].map(
                    lambda value: _first_retained_accession(value, retained)
                )
                human["target"] = human["SWISS-PROT Accessions Interactor B"].map(
                    lambda value: _first_retained_accession(value, retained)
                )
                human = human[
                    human["source"].notna()
                    & human["target"].notna()
                    & human["source"].ne(human["target"])
                ]
                if not human.empty:
                    parts.append(human[["source", "target"]])
    if not parts:
        return pd.DataFrame(columns=["source", "target", "weight"])
    edges = pd.concat(parts, ignore_index=True)
    edges["weight"] = 1.0
    return edges


def read_signor_frame(frame, retained_accessions):
    retained = set(retained_accessions)
    frame = frame.copy()
    frame["source"] = frame["IDA"].map(_uniprot_accession)
    frame["target"] = frame["IDB"].map(_uniprot_accession)
    frame = frame[
        frame["source"].isin(retained)
        & frame["target"].isin(retained)
        & frame["source"].ne(frame["target"])
        & frame["DATABASEA"].str.upper().eq("UNIPROT")
        & frame["DATABASEB"].str.upper().eq("UNIPROT")
    ].copy()
    effect = frame["EFFECT"].fillna("").str.lower()
    frame["activation"] = effect.str.contains("up-regulates").astype(np.int32)
    frame["inhibition"] = effect.str.contains("down-regulates").astype(np.int32)
    directed = pd.DataFrame({"accession": sorted(retained)})
    for endpoint, prefix in (("source", "out"), ("target", "in")):
        counts = frame.groupby(endpoint)[["activation", "inhibition"]].sum()
        counts = counts.rename(
            columns={
                "activation": f"signor_{prefix}_activation",
                "inhibition": f"signor_{prefix}_inhibition",
            }
        )
        directed = directed.merge(
            counts, left_on="accession", right_index=True, how="left"
        )
    directed = directed.fillna(0)
    count_columns = [column for column in directed if column != "accession"]
    directed[count_columns] = directed[count_columns].astype(np.int32)
    edges = frame[["source", "target"]].copy()
    edges["weight"] = 1.0
    typed_edges = {}
    for relation_name, column in (
        ("signor_activation", "activation"),
        ("signor_inhibition", "inhibition"),
    ):
        typed = frame.loc[frame[column].eq(1), ["source", "target"]].copy()
        typed["weight"] = 1.0
        typed_edges[relation_name] = typed
    return edges, directed, typed_edges


def read_signor(path, retained_accessions):
    frame = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
    return read_signor_frame(frame, retained_accessions)


def variant_relation_frames(string_edges, biogrid_edges, signor_relation_edges, variant):
    """Return relation frames for one network sensitivity condition."""
    if variant not in NETWORK_VARIANTS:
        raise ValueError(f"unknown network variant: {variant}")
    if variant == "without_network":
        return {}
    if variant == "ordinary_ppi":
        return {"string": pd.concat([string_edges, biogrid_edges], ignore_index=True)}
    frames = {"string": string_edges, "biogrid": biogrid_edges}
    if variant in {"complete", "direct_evidence_masked"}:
        frames.update(signor_relation_edges)
    return frames


def _site_splits(processed_root, split_path):
    sites = pd.read_parquet(processed_root / "site_index.parquet")
    splits = pd.read_parquet(split_path)
    required = {"site_id", "split"}
    if required.difference(splits.columns):
        raise ValueError("split file must contain site_id and split")
    merged = sites.merge(splits[["site_id", "split"]], on="site_id", how="inner", validate="one_to_one")
    if len(merged) != len(sites):
        raise ValueError("split file does not cover every site")
    return merged


def _empty_edge_frame():
    return pd.DataFrame(columns=["source", "target", "weight"])


def build_network_variant(config, split_path, variant, output_dir):
    """Build an isolated, auditable network sidecar without altering processed data."""
    if variant not in NETWORK_VARIANTS:
        raise ValueError(f"unknown network variant: {variant}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    sites = _site_splits(config.processed_data, Path(split_path))
    accessions = sorted(sites["accession"].astype(str).unique())
    network_root = config.source_data / "04_networks"
    alias_map = read_string_aliases(network_root / "9606.protein.aliases.v12.0.txt.gz", accessions)
    string_edges = read_string_edges(network_root / "9606.protein.links.v12.0.txt.gz", alias_map)
    biogrid_edges = read_biogrid_edges(network_root / "BIOGRID-MV-Physical-LATEST.tab3.zip", accessions)
    raw_signor = pd.read_csv(
        config.source_data / "03_regulation" / "SIGNOR_Jul2026_release.txt",
        sep="\t", dtype=str, low_memory=False,
    )
    filtered_signor, removed = filter_direct_signor_evidence(raw_signor, sites)
    selected_signor = filtered_signor if variant == "direct_evidence_masked" else raw_signor
    if variant in {"without_signor", "ordinary_ppi", "without_network"}:
        selected_signor = raw_signor.iloc[0:0].copy()
    signor_edges, signor_directed, signor_relation_edges = read_signor_frame(selected_signor, accessions)

    if variant == "ordinary_ppi":
        graph_string = pd.concat([string_edges, biogrid_edges], ignore_index=True)
        graph_biogrid = _empty_edge_frame()
    else:
        graph_string, graph_biogrid = string_edges, biogrid_edges
    relation_frames = variant_relation_frames(graph_string, graph_biogrid, signor_relation_edges, variant)

    if variant == "without_network":
        node_dim = np.load(config.processed_data / "relation_node_features.npy", mmap_mode="r").shape[1]
        global_dim = np.load(config.processed_data / "relation_global_context.npy", mmap_mode="r").shape[1]
        relation_graph = build_relation_graph({}, accessions)
        protein_features = pd.DataFrame({"accession": accessions})
        node_features = np.zeros((len(accessions), node_dim), dtype=np.float32)
        global_context = np.zeros((len(accessions), global_dim), dtype=np.float32)
        all_edges = _empty_edge_frame()
    else:
        all_edges = pd.concat([graph_string, graph_biogrid, signor_edges], ignore_index=True)
        combined = compute_graph_features(all_edges, accessions, embedding_dim=config.embedding_dim, seed=config.seed)
        string_topology = _prefixed_topology(graph_string, accessions, "string")
        biogrid_topology = _prefixed_topology(graph_biogrid, accessions, "biogrid")
        protein_features = combined.merge(string_topology, on="accession", validate="one_to_one")
        protein_features = protein_features.merge(biogrid_topology, on="accession", validate="one_to_one")
        protein_features = protein_features.merge(signor_directed, on="accession", validate="one_to_one")
        embedding_columns = [column for column in protein_features if column.startswith("embedding_")]
        node_columns = [
            column for column in protein_features.select_dtypes(include=[np.number]).columns
            if column not in embedding_columns
        ]
        node_features = protein_features[node_columns].to_numpy(dtype=np.float32)
        global_context = protein_features[embedding_columns].to_numpy(dtype=np.float32)
        relation_graph = build_relation_graph(relation_frames, accessions)

    neighbors = sample_relation_neighbors(relation_graph, limit=64)
    np.savez_compressed(
        output_dir / "relation_graph.npz",
        node_accessions=relation_graph["node_accessions"], edge_index=relation_graph["edge_index"],
        edge_weight=relation_graph["edge_weight"], relation=relation_graph["relation"],
        sampled_neighbor_index=neighbors["index"], sampled_neighbor_weight=neighbors["weight"],
        sampled_neighbor_relation=neighbors["relation"], sampled_neighbor_mask=neighbors["mask"],
    )
    np.save(output_dir / "relation_node_features.npy", node_features)
    np.save(output_dir / "relation_global_context.npy", global_context)
    if variant == "direct_evidence_masked":
        removed.to_csv(output_dir / "网络直接证据删除清单.csv", index=False, encoding="utf-8-sig")
    relation_counts = {
        name: int((relation_graph["relation"] == index).sum())
        for index, name in enumerate(RELATION_NAMES)
    }
    audit = {
        "variant": variant,
        "site_count": int(len(sites)),
        "accession_count": int(len(accessions)),
        "removed_direct_evidence_rows": int(len(removed) if variant == "direct_evidence_masked" else 0),
        "source_edge_rows": {"string": int(len(graph_string)), "biogrid": int(len(graph_biogrid)), "signor": int(len(signor_edges))},
        "relation_directed_edges": relation_counts,
        "node_feature_dim": int(node_features.shape[1]),
        "global_context_dim": int(global_context.shape[1]),
    }
    (output_dir / "network_variant_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def _prefixed_topology(edges, accessions, prefix):
    topology = compute_graph_features(edges, accessions, embedding_dim=0)
    return topology.rename(
        columns={column: f"{prefix}_{column}" for column in TOPOLOGY_COLUMNS}
    )


def _biogrid_release(path):
    with zipfile.ZipFile(path) as archive:
        member = next(name for name in archive.namelist() if name.endswith(".tab3.txt"))
    match = re.search(r"-(\d+\.\d+\.\d+)\.tab3", member)
    return match.group(1) if match else member


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build external interaction-network features")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config = MultimodalConfig.load(args.config)
    config.ensure_output_directories()
    site_index = pd.read_parquet(config.processed_data / "site_index.parquet")
    accessions = sorted(site_index["accession"].astype(str).unique())
    network_root = config.source_data / "04_networks"
    string_alias_path = network_root / "9606.protein.aliases.v12.0.txt.gz"
    string_links_path = network_root / "9606.protein.links.v12.0.txt.gz"
    biogrid_path = network_root / "BIOGRID-MV-Physical-LATEST.tab3.zip"
    signor_path = config.source_data / "03_regulation" / "SIGNOR_Jul2026_release.txt"

    alias_map = read_string_aliases(string_alias_path, accessions)
    string_edges = read_string_edges(string_links_path, alias_map)
    biogrid_edges = read_biogrid_edges(biogrid_path, accessions)
    signor_edges, signor_directed, signor_relation_edges = read_signor(
        signor_path, accessions
    )
    all_edges = pd.concat(
        [string_edges, biogrid_edges, signor_edges], ignore_index=True
    )
    combined = compute_graph_features(
        all_edges, accessions, embedding_dim=config.embedding_dim, seed=config.seed
    )
    string_topology = _prefixed_topology(string_edges, accessions, "string")
    biogrid_topology = _prefixed_topology(biogrid_edges, accessions, "biogrid")
    protein_features = combined.merge(
        string_topology, on="accession", validate="one_to_one"
    ).merge(biogrid_topology, on="accession", validate="one_to_one")
    protein_features = protein_features.merge(
        signor_directed, on="accession", validate="one_to_one"
    )
    protein_features["network_present"] = (1 - protein_features["isolated"]).astype(
        np.int8
    )
    numeric = protein_features.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise AssertionError("network feature matrix contains non-finite values")
    direct_coverage = float(protein_features["network_present"].mean())
    if direct_coverage < 0.90:
        raise ValueError(f"direct external-network coverage is only {direct_coverage:.3f}")

    protein_features.to_parquet(
        config.processed_data / "protein_network_features.parquet", index=False
    )

    relation_graph = build_relation_graph(
        {
            "string": string_edges,
            "biogrid": biogrid_edges,
            **signor_relation_edges,
        },
        accessions,
    )
    sampled_neighbors = sample_relation_neighbors(relation_graph, limit=64)
    np.savez_compressed(
        config.processed_data / "relation_graph.npz",
        node_accessions=relation_graph["node_accessions"],
        edge_index=relation_graph["edge_index"],
        edge_weight=relation_graph["edge_weight"],
        relation=relation_graph["relation"],
        sampled_neighbor_index=sampled_neighbors["index"],
        sampled_neighbor_weight=sampled_neighbors["weight"],
        sampled_neighbor_relation=sampled_neighbors["relation"],
        sampled_neighbor_mask=sampled_neighbors["mask"],
    )
    embedding_columns = [
        column for column in protein_features.columns if column.startswith("embedding_")
    ]
    node_feature_columns = [
        column
        for column in protein_features.select_dtypes(include=[np.number]).columns
        if column not in embedding_columns and column != "network_present"
    ]
    np.save(
        config.processed_data / "relation_node_features.npy",
        protein_features[node_feature_columns].to_numpy(dtype=np.float32),
    )
    np.save(
        config.processed_data / "relation_global_context.npy",
        protein_features[embedding_columns].to_numpy(dtype=np.float32),
    )
    expanded = site_index[["site_id", "accession"]].merge(
        protein_features, on="accession", how="left", validate="many_to_one"
    )
    if len(expanded) != len(site_index) or expanded["site_id"].duplicated().any():
        raise AssertionError("network expansion changed the unique site index")
    expanded.to_parquet(config.processed_data / "network_features.parquet", index=False)

    accession_index = {accession: index for index, accession in enumerate(accessions)}
    metadata = {
        "string_release": "12.0",
        "biogrid_release": _biogrid_release(biogrid_path),
        "signor_release": "Jul2026",
        "retained_accessions": int(len(accessions)),
        "string_aliases_mapped": int(len(alias_map)),
        "string_edges": int(len(_canonical_pairs(string_edges, accession_index))),
        "biogrid_edges": int(len(_canonical_pairs(biogrid_edges, accession_index))),
        "signor_edges": int(len(_canonical_pairs(signor_edges, accession_index))),
        "relation_names": list(RELATION_NAMES),
        "relation_directed_edge_counts": {
            relation_name: int((relation_graph["relation"] == relation_id).sum())
            for relation_id, relation_name in enumerate(RELATION_NAMES)
        },
        "relation_neighbor_limit": 64,
        "relation_node_feature_columns": node_feature_columns,
        "direct_network_coverage": direct_coverage,
        "isolated_accessions": int(protein_features["isolated"].sum()),
        "label_derived_edges": 0,
    }
    (config.processed_data / "network_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
