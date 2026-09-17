"""Collect paired held-out metrics into the input table used by 09_statistics.py."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRICS = ["r2", "rmse", "rrmse_percent", "me_pred_minus_true"]


def read_required(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required completed-run table is missing: {path}")
    return pd.read_csv(path)


def collect(output_root: Path) -> pd.DataFrame:
    tables = []
    for target in ("tiller", "leaf_age"):
        frame = read_required(output_root / "pointcloud" / f"summary_{target}_xyz.csv")
        frame = frame.loc[frame["split"].eq("held_out_replicate_test")].copy()
        frame["target"] = target
        frame["statistical_family"] = "pointcloud_backbones"
        tables.append(frame)

    temporal = []
    for strategy, label in (("ogr", "Proposed OGR-weighted dual-branch LSTM"), ("uniform", "Uniform-weighted dual-branch LSTM")):
        frame = read_required(output_root / "multimodal" / f"metrics_{strategy}_by_seed.csv")
        frame = frame.loc[
            frame["split"].eq("held_out_replicate_test") & frame["branch"].eq("Fusion")
        ].copy()
        frame["model"] = label
        temporal.append(frame)
    baselines = read_required(output_root / "multimodal" / "baselines" / "results_by_seed.csv")
    temporal.append(baselines.loc[baselines["split"].eq("held_out_replicate_test")].copy())
    temporal = pd.concat(temporal, ignore_index=True)
    temporal["target"] = "vigor_score"
    temporal["statistical_family"] = "temporal_and_fusion_strategies"
    tables.append(temporal)

    controls = read_required(output_root / "multimodal" / "target_controls" / "results_by_seed.csv")
    controls = controls.loc[controls["split"].eq("held_out_replicate_test")].copy()
    controls["model"] = controls["control"]
    controls["target"] = "vigor_score"
    controls["statistical_family"] = "target_reconstruction_controls"
    tables.append(controls)

    combined = pd.concat(tables, ignore_index=True, sort=False)
    required = {"seed", "model", "target", "statistical_family", *METRICS}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"Combined results are missing columns: {sorted(missing)}")
    columns = ["statistical_family", "target", "model", "seed", "split", *METRICS]
    return combined.loc[:, columns].sort_values(["statistical_family", "target", "model", "seed"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect manuscript held-out metrics for paired statistics")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/combined_results_by_seed.csv"))
    args = parser.parse_args()
    result = collect(args.output_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Saved {len(result)} held-out metric rows to {args.output}")


if __name__ == "__main__":
    main()
