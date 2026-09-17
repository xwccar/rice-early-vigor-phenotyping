from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from release_utils import (
    REPO_ROOT,
    choose_device,
    load_yaml,
    regression_metrics,
    resolve_repo_path,
    set_reproducible_seed,
    sha256_file,
    sha256_rows,
    validate_three_way_splits,
    write_json,
)


models = importlib.import_module("02_pointcloud_models")


TARGET_COLUMNS = {"tiller": "manual_tiller", "leaf_age": "manual_leaf_age"}


@dataclass
class PointCloudRecord:
    sample_id: str
    xyz: np.ndarray
    dat: float
    date: str
    target: float
    split: str


class PointCloudDataset(Dataset):
    def __init__(self, records: list[PointCloudRecord], target_mean: float, target_std: float):
        self.records = records
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        return {
            "sample_id": record.sample_id,
            "points": torch.from_numpy(record.xyz.T.copy()).float(),
            "DAT": torch.tensor(record.dat, dtype=torch.float32),
            "date": record.date,
            "target": torch.tensor((record.target - self.target_mean) / self.target_std, dtype=torch.float32),
            "target_raw": torch.tensor(record.target, dtype=torch.float32),
        }


def load_fixed_pointclouds(pointcloud_dir: Path) -> dict[str, tuple[np.ndarray, str, float, str]]:
    result: dict[str, tuple[np.ndarray, str, float, str]] = {}
    archives = sorted(pointcloud_dir.glob("*.npz"))
    if not archives:
        raise FileNotFoundError(f"No NPZ archives found in {pointcloud_dir}")
    for path in archives:
        if path.name == "archive_hashes.csv":
            continue
        with np.load(path, allow_pickle=False) as data:
            required = {"sample_id", "xyz", "date", "DAT", "source_area"}
            if not required.issubset(data.files):
                raise ValueError(f"{path.name} does not contain {sorted(required)}")
            if data["xyz"].ndim != 3 or data["xyz"].shape[1:] != (1024, 3):
                raise ValueError(f"{path.name} xyz must have shape [N, 1024, 3]")
            for sample_id, xyz, date, dat, source_area in zip(
                data["sample_id"], data["xyz"], data["date"], data["DAT"], data["source_area"]
            ):
                key = str(sample_id)
                if key in result:
                    raise ValueError(f"Duplicate point-cloud sample_id: {key}")
                result[key] = (np.asarray(xyz, dtype=np.float32), str(date), float(dat), str(source_area))
    return result


def load_records(config: dict, target: str) -> tuple[dict[str, list[PointCloudRecord]], dict]:
    if target not in TARGET_COLUMNS:
        raise ValueError(f"Unknown target: {target}")
    data_config = config["data"]
    traits_path = resolve_repo_path(data_config["manual_traits"])
    split_path = resolve_repo_path(data_config["splits"])
    traits = pd.read_csv(traits_path, dtype={"sample_id": str, "date": str})
    splits = pd.read_csv(split_path, dtype={"sample_id": str})
    validate_three_way_splits(splits)
    # data/splits.csv is the single authoritative split table.  The release
    # trait table also carries a convenience copy, which must not create
    # split_x/split_y columns during the validated join.
    traits = traits.drop(columns=["split"], errors="ignore")
    label_column = TARGET_COLUMNS[target]
    if label_column not in traits:
        raise ValueError(f"{traits_path.name} does not contain {label_column}")
    joined = traits.merge(splits[["sample_id", "split"]], on="sample_id", how="inner", validate="one_to_one")
    joined[label_column] = pd.to_numeric(joined[label_column], errors="coerce")
    joined = joined.loc[np.isfinite(joined[label_column])].copy()
    clouds = load_fixed_pointclouds(resolve_repo_path(data_config["pointcloud_dir"]))
    expected_dates = [str(value) for value in data_config.get("dates", [])]
    expected_dats = [float(value) for value in data_config.get("dat_values", [])]
    if expected_dates or expected_dats:
        if len(expected_dates) != len(expected_dats) or not expected_dates:
            raise ValueError("Configured dates and dat_values must be nonempty lists of equal length")
        expected_schedule = set(zip(expected_dates, expected_dats))
        observed_schedule = {(date, dat) for _, date, dat, _ in clouds.values()}
        if observed_schedule != expected_schedule:
            raise ValueError(
                f"Point-cloud date/DAT schedule differs from config: expected {sorted(expected_schedule)}, "
                f"observed {sorted(observed_schedule)}"
            )
    records = []
    for row in joined.itertuples(index=False):
        if row.sample_id not in clouds:
            continue
        xyz, date, dat, source_area = clouds[row.sample_id]
        if str(row.date) != date or float(row.DAT) != dat or str(row.source_area) != source_area:
            raise ValueError(f"Point-cloud metadata and manual_traits.csv disagree for {row.sample_id}")
        records.append(PointCloudRecord(row.sample_id, xyz, dat, date, float(getattr(row, label_column)), row.split))
    grouped = {name: [record for record in records if record.split == name] for name in ("train", "internal_validation", "held_out_replicate_test")}
    for name, values in grouped.items():
        if not values:
            raise ValueError(f"No usable {target} records for split {name}")
    train_targets = np.array([record.target for record in grouped["train"]], dtype=np.float64)
    train_dat = np.array([record.dat for record in grouped["train"]], dtype=np.float64)
    statistics = {
        "target_mean": float(train_targets.mean()),
        "target_std": float(train_targets.std()) if train_targets.std() > 1e-12 else 1.0,
        "dat_mean": float(train_dat.mean()),
        "dat_std": float(train_dat.std()) if train_dat.std() > 1e-12 else 1.0,
        "split_sha256": sha256_rows(splits, ["sample_id", "split"]),
        "manual_traits_sha256": sha256_file(traits_path),
        "splits_file_sha256": sha256_file(split_path),
    }
    return grouped, statistics


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=shuffle and len(dataset) % batch_size == 1,
    )


def forward_model(model: nn.Module, batch: dict, use_dat: bool) -> torch.Tensor:
    return model(batch["points"], batch["DAT"]) if use_dat else model(batch["points"])


def forward_pointnet_with_transform(model: nn.Module, points: torch.Tensor, dat: torch.Tensor, use_dat: bool):
    if use_dat:
        return model.forward_with_feature_transform(points, dat)
    return model.forward_with_feature_transform(points)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_dat: bool, mean: float, std: float):
    model.eval()
    rows = []
    standardized_losses = []
    for batch in loader:
        points = batch["points"].to(device)
        dat = batch["DAT"].to(device)
        predictions_z = forward_model(model, {"points": points, "DAT": dat}, use_dat).squeeze(1)
        target_z = batch["target"].to(device)
        standardized_losses.extend(((predictions_z - target_z) ** 2).cpu().numpy().tolist())
        predictions = predictions_z.cpu().numpy() * std + mean
        for sample_id, date, dat_value, observed, predicted in zip(
            batch["sample_id"], batch["date"], batch["DAT"].numpy(), batch["target_raw"].numpy(), predictions
        ):
            rows.append({"sample_id": sample_id, "date": date, "DAT": float(dat_value), "observed": float(observed), "predicted": float(predicted)})
    return float(np.mean(standardized_losses)), pd.DataFrame(rows)


def train_one(config: dict, target: str, model_name: str, seed: int, use_dat: bool = False, output_root: Path | None = None) -> dict:
    set_reproducible_seed(seed)
    grouped, statistics = load_records(config, target)
    training = config["training"]
    device = choose_device(training.get("device", "auto"))
    datasets = {
        split: PointCloudDataset(records, statistics["target_mean"], statistics["target_std"])
        for split, records in grouped.items()
    }
    loaders = {
        "train": make_loader(datasets["train"], int(training["batch_size"]), True, seed, int(training.get("num_workers", 0))),
        "internal_validation": make_loader(datasets["internal_validation"], int(training["batch_size"]), False, seed, int(training.get("num_workers", 0))),
        "held_out_replicate_test": make_loader(datasets["held_out_replicate_test"], int(training["batch_size"]), False, seed, int(training.get("num_workers", 0))),
    }
    model = models.build_model(
        model_name,
        dgcnn_k=int(config["models"].get("dgcnn_k", 20)),
        transformer_k=int(config["models"].get("transformer_k", 16)),
        use_dat=use_dat,
        dat_mean=statistics["dat_mean"],
        dat_std=statistics["dat_std"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(training["max_epochs"]), eta_min=float(training["min_learning_rate"])
    )
    criterion = nn.MSELoss()
    output_root = output_root or resolve_repo_path(config.get("output_dir", "outputs/pointcloud"))
    run_dir = output_root / target / (f"{model_name}_dat" if use_dat else model_name) / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_model.pth"
    model_definition_sha256 = sha256_file(REPO_ROOT / "02_pointcloud_models" / "pointnet.py") if model_name == "pointnet" else None
    training_script_sha256 = sha256_file(Path(__file__).resolve())
    best_loss, best_epoch, stale = float("inf"), 0, 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        losses = []
        for batch in loaders["train"]:
            points, dat = batch["points"].to(device), batch["DAT"].to(device)
            target_z = batch["target"].to(device)
            optimizer.zero_grad(set_to_none=True)
            if model_name == "pointnet":
                raw_prediction, feature_transform = forward_pointnet_with_transform(model, points, dat, use_dat)
                prediction_z = raw_prediction.squeeze(1)
            else:
                prediction_z = forward_model(model, {"points": points, "DAT": dat}, use_dat).squeeze(1)
                feature_transform = None
            loss = criterion(prediction_z, target_z)
            if feature_transform is not None:
                loss = loss + float(training.get("feature_transform_weight", 0.001)) * models.feature_transform_regularizer(feature_transform)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        validation_loss, _ = evaluate(
            model, loaders["internal_validation"], device, use_dat, statistics["target_mean"], statistics["target_std"]
        )
        history.append({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"], "train_mse_standardized": np.mean(losses), "internal_validation_mse_standardized": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(), "model": model_name, "target": target,
                    "seed": seed, "use_dat": use_dat, "best_epoch": best_epoch,
                    "training_statistics": statistics, "resolved_config": config,
                    "model_definition_sha256": model_definition_sha256,
                    "training_script_sha256": training_script_sha256,
                },
                best_path,
            )
        else:
            stale += 1
        scheduler.step()
        if epoch >= int(training["min_epochs"]) and stale >= int(training["patience"]):
            break
    training_seconds = time.perf_counter() - start
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    prediction_tables, metric_rows = [], []
    for split in ("train", "internal_validation", "held_out_replicate_test"):
        final_loader = make_loader(loaders[split].dataset, int(config["training"]["batch_size"]), False, seed)
        _, predictions = evaluate(model, final_loader, device, use_dat, statistics["target_mean"], statistics["target_std"])
        predictions.insert(1, "split", split)
        predictions.insert(0, "seed", seed)
        prediction_tables.append(predictions)
        metric_rows.append({"seed": seed, "target": target, "model": model_name, "use_dat": use_dat, "split": split, **regression_metrics(predictions["observed"], predictions["predicted"])})
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_tables, ignore_index=True)
    date_metric_rows = []
    for (split, date, dat), group in predictions.groupby(["split", "date", "DAT"], sort=True):
        date_metric_rows.append(
            {
                "seed": seed,
                "target": target,
                "model": model_name,
                "use_dat": use_dat,
                "split": split,
                "date": date,
                "DAT": dat,
                "sample_count": len(group),
                **regression_metrics(group["observed"], group["predicted"]),
            }
        )
    date_metrics = pd.DataFrame(date_metric_rows)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    date_metrics.to_csv(run_dir / "metrics_by_date.csv", index=False)
    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    metadata = {
        "seed": seed, "target": target, "model": model_name, "use_dat": use_dat,
        "protocol": "three_way_spatial_split_internal_validation_checkpoint_selection",
        "best_epoch": best_epoch, "training_seconds": training_seconds,
        "checkpoint_sha256": sha256_file(best_path),
        "model_definition_sha256": model_definition_sha256,
        "training_script_sha256": training_script_sha256,
        **statistics,
    }
    write_json(run_dir / "run_metadata.json", metadata)
    return {**metadata, **metrics.loc[metrics["split"] == "held_out_replicate_test"].iloc[0].to_dict(), "run_dir": str(run_dir.relative_to(REPO_ROOT))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train point-cloud regression backbones with a three-way split")
    parser.add_argument("--config", default="config/pointcloud.yaml")
    parser.add_argument("--target", required=True, choices=sorted(TARGET_COLUMNS))
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--use-dat", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    model_names = list(config["models"]["names"]) if args.all_models else models.parse_model_names(args.model or ["pointnet"])
    seeds = list(config["training"]["seeds"]) if args.all_seeds else (args.seed or [int(config["training"]["seeds"][0])])
    summaries = [train_one(config, args.target, model_name, int(seed), args.use_dat) for seed in seeds for model_name in model_names]
    output = resolve_repo_path(config.get("output_dir", "outputs/pointcloud"))
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"summary_{args.target}_{'dat' if args.use_dat else 'xyz'}.csv"
    pd.DataFrame(summaries).to_csv(path, index=False)
    print(f"Saved {len(summaries)} completed runs to {path}")


if __name__ == "__main__":
    main()
