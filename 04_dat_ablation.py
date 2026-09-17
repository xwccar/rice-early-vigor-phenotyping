from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from release_utils import REPO_ROOT, load_yaml, regression_metrics, resolve_repo_path, validate_three_way_splits


def load_training_module():
    path = REPO_ROOT / "03_train_pointcloud.py"
    spec = importlib.util.spec_from_file_location("pointcloud_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def dat_only(config: dict, target: str, seed: int, output_dir: Path) -> dict:
    target_column = {"tiller": "manual_tiller", "leaf_age": "manual_leaf_age"}[target]
    traits = pd.read_csv(resolve_repo_path(config["data"]["manual_traits"]), dtype={"sample_id": str})
    splits = pd.read_csv(resolve_repo_path(config["data"]["splits"]), dtype={"sample_id": str})
    validate_three_way_splits(splits)
    traits = traits.drop(columns=["split"], errors="ignore")
    frame = traits.merge(splits[["sample_id", "split"]], on="sample_id", validate="one_to_one")
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    frame["DAT"] = pd.to_numeric(frame["DAT"], errors="coerce")
    frame = frame.dropna(subset=[target_column, "DAT"])
    train = frame[frame["split"] == "train"]
    model = LinearRegression().fit(train[["DAT"]], train[target_column])
    rows, metrics = [], []
    for split, part in frame.groupby("split", sort=False):
        predicted = model.predict(part[["DAT"]])
        prediction = part[["sample_id", "date", "DAT"]].copy()
        prediction["observed"], prediction["predicted"] = part[target_column].to_numpy(), predicted
        prediction["split"], prediction["seed"], prediction["method"] = split, seed, "DAT-only"
        rows.append(prediction)
        metrics.append({"seed": seed, "target": target, "method": "DAT-only", "split": split, **regression_metrics(part[target_column], predicted)})
    run_dir = output_dir / target / "DAT-only" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(rows, ignore_index=True)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(metrics).to_csv(run_dir / "metrics.csv", index=False)
    by_date = []
    for (split, date, dat), group in predictions.groupby(["split", "date", "DAT"], sort=True):
        by_date.append(
            {
                "seed": seed,
                "target": target,
                "method": "DAT-only",
                "split": split,
                "date": date,
                "DAT": dat,
                "sample_count": len(group),
                **regression_metrics(group["observed"], group["predicted"]),
            }
        )
    pd.DataFrame(by_date).to_csv(run_dir / "metrics_by_date.csv", index=False)
    heldout = next(row for row in metrics if row["split"] == "held_out_replicate_test")
    return {**heldout, "coefficient": float(model.coef_[0]), "intercept": float(model.intercept_)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DAT-only, PointNet-only, and PointNet+DAT")
    parser.add_argument("--config", default="config/pointcloud.yaml")
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--all-seeds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seeds = list(config["training"]["seeds"]) if args.all_seeds else (args.seed or [int(config["training"]["seeds"][0])])
    output_dir = resolve_repo_path("outputs/dat_ablation")
    training = load_training_module()
    results = []
    for seed in seeds:
        for target in ("tiller", "leaf_age"):
            results.append(dat_only(config, target, int(seed), output_dir))
            for use_dat, method in ((False, "PointNet-only"), (True, "PointNet+DAT")):
                row = training.train_one(config, target, "pointnet", int(seed), use_dat=use_dat, output_root=output_dir)
                row["method"] = method
                results.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(results)
    result.to_csv(output_dir / "results_by_seed.csv", index=False)
    numeric = ["r2", "rmse", "rrmse_percent", "me_pred_minus_true"]
    summary = result.groupby(["target", "method"])[numeric].agg(["mean", "std"]).reset_index()
    summary.to_csv(output_dir / "mean_results.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
