# PhosMMPU

Positive-unlabeled multimodal learning for functional phosphosite prioritization.

PhosMMPU prioritizes phosphorylation sites whose functional roles have not yet been annotated. Instead of treating every unannotated site as a confirmed negative example, it formulates functional phosphosite prediction as a positive-unlabeled learning problem. The framework integrates sequence, local structure, multi-relation protein-network and proteomics representations, and produces task-specific scores for three regulatory effects:

- regulation of protein activity;
- regulation of molecular interactions;
- regulation of protein stability or degradation.

This repository contains source code, configuration files and documentation. Processed datasets, prediction tables and trained model weights are distributed separately and are not included in the GitHub repository.

## Method overview

PhosMMPU represents each phosphosite with four complementary modalities:

| Modality | Representation |
|---|---|
| Sequence | A site-centered amino-acid window and an auxiliary sequence embedding |
| Structure | A local residue graph encoded with geometric vector perceptron layers |
| Protein network | Typed neighborhoods from a multi-relation protein interaction graph |
| Proteomics | Site-aligned quantitative proteomics features |

Each modality is projected into a shared latent space. A gated cross-modal Transformer integrates the available modality tokens while respecting missing-modality masks. Three output heads are optimized jointly with a non-negative positive-unlabeled objective.

## Repository structure

```text
PhosMMPU/
|-- configs/multimodal/      experiment configuration files
|-- docs/                    external data layout and array contracts
|-- scripts/                 training, evaluation and export utilities
|-- src/multimodal/          feature construction, model and PU-learning code
|-- .gitignore               data, weights and generated-output exclusions
|-- environment.yml          Conda environment definition
|-- pyproject.toml           package metadata and command-line entry points
|-- requirements.txt         Python dependencies
`-- LICENSE                  GNU General Public License
```

## Requirements

- Python 3.11
- PyTorch 2.2 or later
- A CUDA-capable GPU is recommended for training
- MMseqs2 is required only when rebuilding homology clusters

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate phosmmpu
pip install -e .
```

### Python virtual environment

```bash
python -m venv .venv
```

Activate the environment and install the project:

```bash
pip install -e .
```

For GPU training, install the PyTorch build compatible with the local CUDA runtime before running `pip install -e .`.

## External data

Copy the separately distributed processed dataset into the repository using this layout:

```text
data/
└── processed/
    ├── site_index.parquet
    ├── dataset_site_ids.npy
    ├── labels.npy
    ├── split_names.npy
    ├── sequence.npy
    ├── sequence_aux.npy
    ├── structure_graph_scalars.npy
    ├── structure_graph_vectors.npy
    ├── structure_graph_mask.npy
    ├── relation_graph.npz
    ├── relation_node_features.npy
    ├── relation_global_context.npy
    ├── proteomics_processed.npy
    └── modality_masks.npy
```

The complete filename list and expected array shapes are documented in [`docs/DATA_LAYOUT.md`](docs/DATA_LAYOUT.md). All site-level arrays must follow the ordering stored in `dataset_site_ids.npy`.

The three columns of `labels.npy` correspond to activity regulation, interaction regulation and proteostasis regulation. A value of zero denotes an unlabeled site, not an experimentally confirmed negative site.

## Training

The packaged pipeline trains five nnPU models with seeds 11, 23, 37, 51 and 73. It uses two cross-modal Transformer layers, the complete relation graph and no modality dropout.

Linux or macOS:

```bash
bash scripts/run_pipeline.sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_pipeline.ps1 -Device cuda
```

The same experiment can be launched directly:

```bash
phosmmpu-train \
  --config configs/multimodal/deployment.json \
  --seeds 11 23 37 51 73 \
  --epochs 200 \
  --patience 30 \
  --batch-size 512 \
  --objective nnpu \
  --transformer-layers 2 \
  --modality-dropout 0 \
  --network-dir data/processed \
  --network-variant complete \
  --device auto
```

Training outputs are written to `artifacts/multimodal/training_runs`. Each seed directory contains:

```text
nnpu_seed<seed>/
|-- history.csv
|-- model.pt
|-- run.json
|-- test_predictions.parquet
|-- test_probabilities.npy
`-- validation_probabilities.npy
```

Generated artifacts and model weights are ignored by Git.

## Candidate ranking

Export ranked candidates from one completed seed run:

```bash
phosmmpu-export \
  --run-dir artifacts/multimodal/training_runs/nnpu_seed11 \
  --top-k 10
```

Summarize the five runs and generate ensemble candidate scores:

```bash
phosmmpu-summarize \
  --run-root artifacts/multimodal/training_runs \
  --expected-seeds 11 23 37 51 73
```

Candidate scores are intended for experimental prioritization and should not be interpreted as direct experimental evidence of function.

## Evaluation

The ranking evaluation utilities report:

- PR-AUC;
- Hits@K;
- Recall@K;
- NDCG@K.

Additional modules support modality ablation, positive-prior sensitivity, network-evidence sensitivity, hidden-positive recovery, subgroup analysis and sequence-based comparisons.

## Reproducibility

- Homology-aware splits keep each MMseqs2 cluster within a single data partition.
- Network feature scaling is fitted using training-split proteins only.
- Site identifiers and split hashes are written to each run manifest.
- Random seeds, class-prior estimates, model dimensions and training settings are stored in `run.json`.
- Source data and trained weights remain outside this repository to prevent accidental inclusion in commits.

## Citation

The citation for PhosMMPU will be added after publication. Until then, please cite this repository and record the commit identifier used in the analysis.

## License

PhosMMPU is released under the GNU General Public License v3.0. See [`LICENSE`](LICENSE).
