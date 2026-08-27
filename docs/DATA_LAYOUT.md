# External data layout

The source repository expects a separately distributed processed dataset at `data/processed`. The following files are required by the final training pipeline:

```text
data/processed/
  site_index.parquet
  dataset_site_ids.npy
  labels.npy
  split_names.npy
  sequence.npy
  sequence_aux.npy
  structure_graph_scalars.npy
  structure_graph_vectors.npy
  structure_graph_mask.npy
  relation_graph.npz
  relation_node_features.npy
  relation_global_context.npy
  proteomics_processed.npy
  modality_masks.npy
```

Recommended provenance and audit files include `labels.parquet`, `splits.parquet`, `homology_clusters.csv`, `homology_cluster_audit.json`, `label_audit.json`, `split_audit.json`, sequence and structure audit JSON files, `network_metadata.json` and `proteomics_audit.json`.

## Array contract

| Artifact | Expected shape |
|---|---:|
| `dataset_site_ids.npy` | `(130554,)` |
| `labels.npy` | `(130554, 3)` |
| `split_names.npy` | `(130554,)` |
| `sequence.npy` | `(130554, 31, 21)` |
| `sequence_aux.npy` | `(130554, 643)` |
| `structure_graph_scalars.npy` | `(130554, 24, 28)` |
| `structure_graph_vectors.npy` | `(130554, 24, 3)` |
| `structure_graph_mask.npy` | `(130554, 24)` |
| `relation_node_features.npy` | `(3141, 16)` |
| `relation_global_context.npy` | `(3141, 64)` |
| `proteomics_processed.npy` | `(130554, 28)` |
| `modality_masks.npy` | `(130554, 4)` |

The relation graph contains 3,141 protein nodes and sampled relation-specific neighborhoods of shape `(3141, 64)`. All site-level arrays must follow the ordering in `dataset_site_ids.npy`.

The three label columns represent activity regulation, interaction regulation and proteostasis regulation. Zeros denote unlabeled examples, not experimentally confirmed negative sites.

## Leakage-safe split

The final split is grouped by MMseqs2 homology clusters. It uses a minimum sequence identity of 0.40, minimum coverage of 0.70 and `cov-mode=0`, ensuring that each homology cluster occurs in only one of the training, validation or test partitions.

Dataset files and model weights are excluded from this GitHub directory by design. Keep them in the separate `PhosMMPU` data package or attach them locally after cloning.
