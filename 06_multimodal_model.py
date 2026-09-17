from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from multimodal_core import (
    MultimodalLSTMAttention,
    SequenceArrays,
    SequencePreprocessor,
    build_optimizer,
    load_sequences,
    make_loader,
    take,
)
from release_utils import choose_device, load_yaml, regression_metrics, resolve_repo_path, set_reproducible_seed, sha256_file, write_json


BRANCHES = ("MS", "LiDAR", "Fusion")


@torch.no_grad()
def evaluate(model, loader, device, collect=False):
    model.eval()
    losses = {branch: 0.0 for branch in BRANCHES}
    count = 0
    rows, attention_rows = [], []
    for ms, lidar, target, plant_uid in loader:
        ms, lidar, target = ms.to(device), lidar.to(device), target.to(device)
        predictions, attention = model(ms, lidar)
        count += len(target)
        for branch in BRANCHES:
            losses[branch] += float(torch.sum((predictions[branch] - target) ** 2).cpu())
            if collect:
                for uid, observed, predicted in zip(plant_uid, target.cpu().numpy().ravel(), predictions[branch].cpu().numpy().ravel()):
                    rows.append({"plant_uid": uid, "branch": branch, "observed": observed, "predicted": predicted})
        if collect:
            for branch in ("MS", "LiDAR"):
                for uid, values in zip(plant_uid, attention[branch].cpu().numpy()):
                    for time_index, value in enumerate(values):
                        attention_rows.append({"plant_uid": uid, "branch": branch, "time_index": time_index + 1, "attention_weight": value})
    return {key: value / count for key, value in losses.items()}, pd.DataFrame(rows), pd.DataFrame(attention_rows)


def inverse_ogr_weights(train_losses: dict, validation_losses: dict, epsilon: float) -> np.ndarray:
    ogr = np.array([validation_losses[branch] / (train_losses[branch] + epsilon) for branch in BRANCHES])
    inverse = 1.0 / (ogr + epsilon)
    return inverse / inverse.sum()


def run_arrays(
    config: dict,
    strategy: str,
    seed: int,
    all_data: SequenceArrays,
    dat_values: list[float],
    output: Path,
    model_label: str,
) -> pd.DataFrame:
    """Train one manuscript-protocol dual-branch model on supplied arrays."""
    if strategy not in {"ogr", "uniform"}:
        raise ValueError("strategy must be ogr or uniform")
    set_reproducible_seed(seed)
    raw = {split: take(all_data, all_data.split == split) for split in np.unique(all_data.split)}
    preprocessor = SequencePreprocessor.fit(raw["train"].ms, raw["train"].lidar)
    data = {split: preprocessor.transform(values) for split, values in raw.items()}
    training = config["training"]
    loaders = {split: make_loader(values, int(training["batch_size"]), split == "train", seed) for split, values in data.items()}
    evaluation_loaders = dict(loaders)
    evaluation_loaders["train"] = make_loader(data["train"], int(training["batch_size"]), False, seed)
    device = choose_device(training.get("device", "auto"))
    model = MultimodalLSTMAttention(
        data["train"].ms.shape[-1],
        data["train"].lidar.shape[-1],
        hidden_size=int(training["hidden_size"]),
        representation_size=int(training.get("representation_size", 16)),
        fusion_hidden=tuple(int(value) for value in training.get("fusion_hidden", [32, 16])),
        dropout=float(training.get("dropout", 0.15)),
    ).to(device)
    optimizer = build_optimizer(model.parameters(), training)
    weights = np.repeat(1 / 3, 3)
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    history = []
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        for ms, lidar, target, _ in loaders["train"]:
            ms, lidar, target = ms.to(device), lidar.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(ms, lidar)
            losses = torch.stack([F.mse_loss(predictions[branch], target) for branch in BRANCHES])
            torch.sum(losses * torch.tensor(weights, device=device, dtype=losses.dtype)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
        train_losses, _, _ = evaluate(model, evaluation_loaders["train"], device)
        validation_losses, _, _ = evaluate(model, loaders["internal_validation"], device)
        epsilon = float(training["ogr_epsilon"])
        ogr_values = np.asarray([validation_losses[branch] / (train_losses[branch] + epsilon) for branch in BRANCHES])
        inverse_scores = 1.0 / (ogr_values + epsilon)
        next_weights = inverse_scores / inverse_scores.sum() if strategy == "ogr" else np.repeat(1 / 3, 3)
        history.append({"model": model_label, "seed": seed, "strategy": strategy, "epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], **{f"train_loss_{b}": train_losses[b] for b in BRANCHES}, **{f"validation_loss_{b}": validation_losses[b] for b in BRANCHES}, **{f"ogr_value_{b}": ogr_values[i] for i, b in enumerate(BRANCHES)}, **{f"inverse_ogr_score_{b}": inverse_scores[i] for i, b in enumerate(BRANCHES)}, **{f"weight_used_{b}": weights[i] for i, b in enumerate(BRANCHES)}, **{f"weight_next_{b}": next_weights[i] for i, b in enumerate(BRANCHES)}})
        weights = next_weights
        if validation_losses["Fusion"] < best_loss:
            best_loss, best_epoch, stale, best_state = validation_losses["Fusion"], epoch, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch >= int(training["min_epochs"]) and stale >= int(training["patience"]):
            break
    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    output.mkdir(parents=True, exist_ok=True)
    metrics, predictions, attention = [], [], []
    for split, loader in evaluation_loaders.items():
        _, split_predictions, split_attention = evaluate(model, loader, device, collect=True)
        split_predictions["model"], split_predictions["split"], split_predictions["seed"], split_predictions["strategy"] = model_label, split, seed, strategy
        predictions.append(split_predictions)
        for branch, group in split_predictions.groupby("branch"):
            metrics.append({"model": model_label, "seed": seed, "strategy": strategy, "split": split, "branch": branch, "sample_count": len(group), **regression_metrics(group["observed"], group["predicted"])})
        if split == "held_out_replicate_test":
            split_attention["split"], split_attention["seed"], split_attention["strategy"] = split, seed, strategy
            split_attention["DAT"] = split_attention["time_index"].map(dict(enumerate(dat_values, start=1)))
            attention.append(split_attention)
    checkpoint_path = output / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model": model_label,
            "seed": seed,
            "strategy": strategy,
            "best_epoch": best_epoch,
            "config": config,
            "preprocessor": preprocessor.state_dict(),
            "ms_input_size": data["train"].ms.shape[-1],
            "lidar_input_size": data["train"].lidar.shape[-1],
        },
        checkpoint_path,
    )
    pd.DataFrame(history).to_csv(output / "ogr_weights.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    pd.concat(attention, ignore_index=True).to_csv(output / "attention_weights.csv", index=False)
    result = pd.DataFrame(metrics)
    result.to_csv(output / "metrics.csv", index=False)
    write_json(
        output / "run_metadata.json",
        {
            "model": model_label,
            "seed": seed,
            "strategy": strategy,
            "best_epoch": best_epoch,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "optimizer": str(training.get("optimizer", "sgd")).lower(),
            "weight_decay": float(training["weight_decay"]),
            "checkpoint_selection_split": "internal_validation",
            "final_evaluation_split": "held_out_replicate_test",
        },
    )
    return result


def run_one(config: dict, strategy: str, seed: int) -> pd.DataFrame:
    all_data, dat_values = load_sequences(config)
    output = resolve_repo_path(config.get("output_dir", "outputs/multimodal")) / strategy / f"seed_{seed}"
    return run_arrays(config, strategy, seed, all_data, dat_values, output, f"dual_branch_{strategy}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the dual-branch LSTM-attention model")
    parser.add_argument("--config", default="config/multimodal.yaml")
    parser.add_argument("--strategy", choices=["ogr", "uniform"], default="ogr")
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--all-seeds", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)
    seeds = config["training"]["seeds"] if args.all_seeds else (args.seed or [config["training"]["seeds"][0]])
    results = pd.concat([run_one(config, args.strategy, int(seed)) for seed in seeds], ignore_index=True)
    output = resolve_repo_path(config.get("output_dir", "outputs/multimodal"))
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / f"metrics_{args.strategy}_by_seed.csv", index=False)
    attention = pd.concat(
        [pd.read_csv(output / args.strategy / f"seed_{int(seed)}" / "attention_weights.csv") for seed in seeds],
        ignore_index=True,
    )
    histories = pd.concat(
        [pd.read_csv(output / args.strategy / f"seed_{int(seed)}" / "ogr_weights.csv") for seed in seeds],
        ignore_index=True,
    )
    attention.to_csv(output / f"attention_{args.strategy}_by_seed.csv", index=False)
    histories.to_csv(output / f"training_weights_{args.strategy}_by_seed.csv", index=False)
    print(results.query("split == 'held_out_replicate_test' and branch == 'Fusion'").to_string(index=False))


if __name__ == "__main__":
    main()
