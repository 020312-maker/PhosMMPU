"""Run a label-hidden known-positive recovery experiment for the final PhosMMPU."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.multimodal.config import MultimodalConfig
from src.multimodal.hidden_positive_recovery import (
    TASKS,
    candidate_positions_after_mask,
    evaluate_hidden_recovery,
    mask_hidden_positives,
    sample_hidden_positives,
)
from src.multimodal.torch_losses import estimate_class_priors
from src.multimodal.phosmmpu_model import PhosMMPUModel
from src.multimodal.train import (
    SPLIT_NAMES,
    _GraphSiteDataset,
    _device,
    _fit,
    _load_processed,
    _predict_loader,
)


TASK_NAMES = {"activity": "活性调控", "interaction": "分子关联", "proteostasis": "稳定性/降解"}
RECOVERY_COLUMNS = (
    "hidden_positive_count", "candidate_count", "hits_at_50", "recall_at_50", "random_recall_at_50",
    "ndcg_at_50", "hits_at_100", "recall_at_100", "random_recall_at_100",
    "ndcg_at_100", "hits_at_200", "recall_at_200", "random_recall_at_200", "ndcg_at_200",
    "hits_at_500", "recall_at_500", "random_recall_at_500", "ndcg_at_500", "mean_hidden_rank", "mrr",
)


def _model(arrays, embedding_dim, modality_dropout, transformer_layers, device):
    return PhosMMPUModel(
        alphabet_size=arrays["sequence"].shape[2], sequence_aux_dim=arrays["sequence_aux"].shape[1],
        structure_scalar_dim=arrays["structure_scalars"].shape[2], network_node_dim=arrays["node_features"].shape[1],
        network_global_dim=arrays["global_context"].shape[1], proteomics_dim=arrays["proteomics"].shape[1],
        embedding_dim=embedding_dim, transformer_layers=transformer_layers, modality_dropout=modality_dropout,
    ).to(device)


def _metric_rows(run_kind, repeat, seed, hidden, candidate_pools, probabilities, train_lookup):
    rows = []
    for task_index, task in enumerate(TASKS):
        candidates = candidate_pools[task_index]
        metrics = evaluate_hidden_recovery(
            hidden[task_index], candidates, probabilities[train_lookup[candidates], task_index],
        )
        rows.append({"模型": "主模型 PhosMMPU（无 modality dropout）", "run_kind": run_kind, "隐藏重复": int(repeat), "随机种子": seed, "task": task, "任务": TASK_NAMES[task], **metrics})
    return rows


def _summary(ensemble_rows):
    table = pd.DataFrame(ensemble_rows)
    long = table.melt(
        id_vars=["模型", "隐藏重复", "task", "任务"], value_vars=list(RECOVERY_COLUMNS),
        var_name="metric", value_name="value",
    )
    result = long.groupby(["模型", "task", "任务", "metric"], as_index=False).agg(
        五次隐藏均值=("value", "mean"), 五次隐藏标准差=("value", "std"), 重复数=("value", "count"),
    )
    return result.sort_values(["task", "metric"], kind="mergesort").reset_index(drop=True)


def run(
    config, processed_dir, network_dir, output_root, seeds, repeats, hidden_count, epochs, patience,
    batch_size, device, sampling_seed=20260814,
):
    output_root = Path(output_root)
    if output_root.exists():
        allowed_start_logs = {"training.log", "training.err.log"}
        existing = {path.name for path in output_root.iterdir()}
        if not existing.issubset(allowed_start_logs):
            raise FileExistsError(f"output root already contains experiment artifacts: {output_root}")
    else:
        output_root.mkdir(parents=True)
    config = replace(config, processed_data=Path(processed_dir).resolve(), modality_dropout=0.0)
    labels = np.load(config.processed_data / "labels.npy").astype(np.int8)
    splits = np.load(config.processed_data / "split_names.npy", allow_pickle=False)
    if len(labels) != len(splits) or set(np.unique(splits)).difference(SPLIT_NAMES):
        raise ValueError("processed labels or split names are invalid")
    groups = {name: np.flatnonzero(splits == name) for name in SPLIT_NAMES}
    if not all(len(rows) for rows in groups.values()):
        raise ValueError("all group-aware splits must be non-empty")
    arrays, site_ids = _load_processed(config, splits, Path(network_dir))
    site_index = pd.read_parquet(config.processed_data / "site_index.parquet")
    if not np.array_equal(site_index["site_id"].astype(str).to_numpy(), site_ids):
        raise ValueError("site index is not aligned with processed features")
    residues = site_index["residue"].astype(str).to_numpy()
    resolved = _device(device)
    train_lookup = np.full(len(labels), -1, dtype=np.int64)
    train_lookup[groups["train"]] = np.arange(len(groups["train"]), dtype=np.int64)
    seed_rows, ensemble_rows, hidden_rows = [], [], []

    for repeat in range(int(repeats)):
        hidden = sample_hidden_positives(
            labels, groups["train"], count_per_task=hidden_count, seed=int(sampling_seed) + repeat, strata=residues,
        )
        masked_labels = mask_hidden_positives(labels, hidden)
        candidate_pools = candidate_positions_after_mask(masked_labels, groups["train"])
        priors = estimate_class_priors(masked_labels[groups["train"]], "train", config.pu_prior_multiplier)
        repeat_root = output_root / f"隐藏重复_{repeat + 1:02d}"
        repeat_root.mkdir()
        for task_index, positions in hidden.items():
            rows = site_index.iloc[positions][["site_id", "accession", "gene", "residue", "position"]].copy()
            rows["隐藏重复"] = repeat + 1
            rows["task"] = TASKS[task_index]
            rows["任务"] = TASK_NAMES[TASKS[task_index]]
            hidden_rows.append(rows)

        seed_probabilities = []
        for seed in seeds:
            run_root = repeat_root / f"nnpu_seed{int(seed)}"
            run_root.mkdir()
            random.seed(int(seed))
            np.random.seed(int(seed))
            torch.manual_seed(int(seed))
            model = _model(arrays, config.embedding_dim, 0.0, 2, resolved)
            generator = torch.Generator().manual_seed(int(seed))
            train_loader = DataLoader(
                _GraphSiteDataset(arrays, groups["train"], masked_labels), batch_size=batch_size, shuffle=True,
                generator=generator, pin_memory=resolved.type == "cuda",
            )
            validation_loader = DataLoader(
                _GraphSiteDataset(arrays, groups["validation"]), batch_size=batch_size, shuffle=False,
                pin_memory=resolved.type == "cuda",
            )
            _, history, best_score = _fit(
                model, train_loader, masked_labels[groups["train"]], validation_loader,
                labels[groups["validation"]], run_root, epochs, int(seed), priors, "nnpu", resolved, patience,
            )
            train_loader_for_score = DataLoader(
                _GraphSiteDataset(arrays, groups["train"]), batch_size=batch_size, shuffle=False,
                pin_memory=resolved.type == "cuda",
            )
            train_probabilities = _predict_loader(model, train_loader_for_score, resolved)
            np.save(run_root / "training_candidate_probabilities.npy", train_probabilities)
            (run_root / "run.json").write_text(json.dumps({
                "model": "主模型 PhosMMPU（无 modality dropout）", "objective": "nnpu", "seed": int(seed),
                "hidden_repeat": repeat + 1, "hidden_count_per_task": int(hidden_count),
                "modality_dropout": 0.0, "transformer_layers": 2, "network_variant": "complete",
                "device": str(resolved), "epochs_requested": int(epochs), "epochs_completed": len(history),
                "early_stopping_patience": int(patience), "best_validation_macro_auprc": float(best_score),
                "class_priors_after_hiding": priors,
                "candidate_pool": "training split U sites after masking; hidden labels excluded from training and ranking",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            seed_rows.extend(_metric_rows("single_seed", repeat + 1, int(seed), hidden, candidate_pools, train_probabilities, train_lookup))
            seed_probabilities.append(train_probabilities)

        ensemble = np.mean(np.stack(seed_probabilities), axis=0)
        np.save(repeat_root / "五种子集成_训练候选预测分数.npy", ensemble)
        ensemble_rows.extend(_metric_rows("five_seed_ensemble", repeat + 1, "五种子集成", hidden, candidate_pools, ensemble, train_lookup))

    pd.concat(hidden_rows, ignore_index=True).to_csv(output_root / "人为隐藏正例清单.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(seed_rows).to_csv(output_root / "隐藏正例恢复_逐种子结果.csv", index=False, encoding="utf-8-sig")
    ensemble = pd.DataFrame(ensemble_rows)
    ensemble.to_csv(output_root / "隐藏正例恢复_五种子集成_每次重复.csv", index=False, encoding="utf-8-sig")
    _summary(ensemble_rows).to_csv(output_root / "隐藏正例恢复_五次重复汇总.csv", index=False, encoding="utf-8-sig")
    (output_root / "实验数据清单.json").write_text(json.dumps({
        "protocol": "训练集 P 标签屏蔽为 U；标签不传入训练、早停或候选排序；只在最终恢复指标读取 H。",
        "processed_dir": str(config.processed_data), "network_dir": str(Path(network_dir).resolve()),
        "model": "完整 PhosMMPU，无 modality dropout，nnPU", "hidden_count_per_task": int(hidden_count),
        "hidden_repeats": int(repeats), "training_seeds": [int(seed) for seed in seeds],
        "split": "homology_cluster", "candidate_pool": "masked training U only", "final_test_used": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output_root": str(output_root), "hidden_repeats": int(repeats), "seeds": [int(seed) for seed in seeds]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run hidden-positive recovery for final nnPU PhosMMPU")
    parser.add_argument("--config", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--network-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 51, 73])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hidden-count", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    config = MultimodalConfig.load(args.config)
    print(json.dumps(run(
        config, args.processed_dir, args.network_dir, args.output_root, args.seeds, args.repeats,
        args.hidden_count, args.epochs, args.patience, args.batch_size, args.device,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
