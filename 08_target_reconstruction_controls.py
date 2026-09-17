from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from multimodal_core import SequenceArrays, load_sequences
from release_utils import (
    REPO_ROOT,
    choose_device,
    load_yaml,
    regression_metrics,
    resolve_repo_path,
    sha256_file,
    sha256_rows,
    write_json,
)


CONTROL_ORDER = ("full_fusion", "deterministic_reconstruction", "independent_feature", "learned_structure")


def load_numbered_script(module_name: str, filename: str):
    path = REPO_ROOT / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def identifier_piece(value: object) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def sequence_pointcloud_ids(config: dict, arrays: SequenceArrays, dat_values: list[float]) -> tuple[np.ndarray, pd.DataFrame]:
    """Map every temporal-model row to the source-qualified point-cloud ID."""
    frame = pd.read_csv(
        resolve_repo_path(config["data"]["model_inputs"]),
        dtype={"plant_uid": str, "pointcloud_sample_id": str, "source_area": str, "date": str, "plot_id": str, "plant_id": str},
    )
    frame["DAT"] = pd.to_numeric(frame["DAT"], errors="coerce")
    identities, rows = [], []
    for plant_uid in arrays.plant_uid:
        group = frame.loc[frame["plant_uid"].eq(str(plant_uid))].copy()
        if group.empty or group["DAT"].duplicated().any():
            raise ValueError(f"Cannot make an unambiguous temporal-to-point-cloud mapping for {plant_uid}")
        group = group.set_index("DAT").reindex(dat_values)
        if group[["source_area", "date", "plot_id", "plant_id"]].isna().any().any():
            raise ValueError(f"Incomplete identity fields for {plant_uid}")
        current = []
        for _, row in group.iterrows():
            if "pointcloud_sample_id" in group.columns and pd.notna(row.get("pointcloud_sample_id")):
                sample_id = str(row["pointcloud_sample_id"])
            else:
                sample_id = ":".join(
                    (
                        identifier_piece(row["source_area"]),
                        identifier_piece(row["date"]).zfill(4),
                        identifier_piece(row["plot_id"]),
                        identifier_piece(row["plant_id"]),
                    )
                )
            current.append(sample_id)
            rows.append({"plant_uid": plant_uid, "DAT": float(row.name), "pointcloud_sample_id": sample_id})
        identities.append(current)
    identity = (
        frame.sort_values(["plant_uid", "DAT"])
        .drop_duplicates("plant_uid")
        .set_index("plant_uid")
        .reindex(arrays.plant_uid)
        .reset_index()[["plant_uid", "variety_id", "source_area", "plot_id", "plant_id", "split"]]
    )
    return np.asarray(identities, dtype=str), identity


def validate_pointnet_checkpoint(checkpoint: dict, target: str, seed: int, pointcloud_config: dict) -> None:
    expected_model_hash = sha256_file(REPO_ROOT / "02_pointcloud_models" / "pointnet.py")
    if checkpoint.get("model") != "pointnet" or checkpoint.get("target") != target or int(checkpoint.get("seed", -1)) != seed:
        raise ValueError(f"Checkpoint metadata does not match PointNet/{target}/seed {seed}")
    if checkpoint.get("use_dat") is not False:
        raise ValueError("Controls require the XYZ-only PointNet checkpoint, not PointNet+DAT")
    if checkpoint.get("model_definition_sha256") != expected_model_hash:
        raise RuntimeError(
            "PointNet checkpoint predates or differs from the current implementation. "
            "Rerun 03_train_pointcloud.py with the current code before running controls."
        )
    splits = pd.read_csv(resolve_repo_path(pointcloud_config["data"]["splits"]), dtype={"sample_id": str})
    expected_split_hash = sha256_rows(splits, ["sample_id", "split"])
    actual_split_hash = checkpoint.get("training_statistics", {}).get("split_sha256")
    if actual_split_hash != expected_split_hash:
        raise RuntimeError("PointNet checkpoint and current data/splits.csv do not have the same split hash")


@torch.no_grad()
def pointnet_predictions_and_descriptors(
    release_config: dict,
    pointcloud_config: dict,
    target: str,
    seed: int,
    sample_ids: np.ndarray,
    include_descriptor: bool,
) -> tuple[np.ndarray, np.ndarray | None, Path]:
    """Apply a seed-specific XYZ-only PointNet checkpoint to temporal samples."""
    training_script = load_numbered_script("pointcloud_training_for_controls", "03_train_pointcloud.py")
    models = importlib.import_module("02_pointcloud_models")
    checkpoint_root = resolve_repo_path(release_config.get("controls", {}).get("pointnet_checkpoint_root", "outputs/pointcloud"))
    checkpoint_path = checkpoint_root / target / "pointnet" / f"seed_{seed}" / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Train XYZ-only PointNet for {target}, seed {seed}, before controls."
        )
    device = choose_device(release_config["training"].get("device", "auto"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    validate_pointnet_checkpoint(checkpoint, target, seed, pointcloud_config)
    model = models.build_model("pointnet").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    clouds = training_script.load_fixed_pointclouds(resolve_repo_path(pointcloud_config["data"]["pointcloud_dir"]))
    flat_ids = sample_ids.reshape(-1)
    missing = sorted(set(flat_ids) - set(clouds))
    if missing:
        raise KeyError(f"{len(missing)} temporal point-cloud IDs are unavailable; first missing ID: {missing[0]}")
    statistics = checkpoint["training_statistics"]
    prediction_lookup, descriptor_lookup = {}, {}
    unique_ids = list(dict.fromkeys(flat_ids.tolist()))
    batch_size = int(pointcloud_config["training"]["batch_size"])
    for start in range(0, len(unique_ids), batch_size):
        identifiers = unique_ids[start : start + batch_size]
        values = np.stack([clouds[sample_id][0].T for sample_id in identifiers]).astype(np.float32)
        points = torch.from_numpy(values).to(device)
        prediction_z = model(points).squeeze(1).cpu().numpy()
        prediction = prediction_z * float(statistics["target_std"]) + float(statistics["target_mean"])
        for sample_id, value in zip(identifiers, prediction):
            prediction_lookup[sample_id] = float(value)
        if include_descriptor:
            descriptor = model.encoder(points)[0].cpu().numpy().astype(np.float32)
            descriptor_lookup.update(dict(zip(identifiers, descriptor)))
    predictions = np.asarray([prediction_lookup[value] for value in flat_ids], dtype=np.float32).reshape(sample_ids.shape)
    descriptors = None
    if include_descriptor:
        descriptors = np.stack([descriptor_lookup[value] for value in flat_ids]).reshape(*sample_ids.shape, 1024)
    return predictions, descriptors, checkpoint_path


def replace_lidar(arrays: SequenceArrays, lidar: np.ndarray) -> SequenceArrays:
    return SequenceArrays(
        arrays.plant_uid.copy(),
        arrays.variety_id.copy(),
        arrays.split.copy(),
        arrays.ms.copy(),
        np.asarray(lidar, dtype=np.float32),
        arrays.target.copy(),
    )


def keep_complete_pointcloud_sequences(
    arrays: SequenceArrays,
    sample_ids: np.ndarray,
    identity: pd.DataFrame,
    pointcloud_config: dict,
) -> tuple[SequenceArrays, np.ndarray, pd.DataFrame, pd.DataFrame]:
    index_path = resolve_repo_path(pointcloud_config["data"]["pointcloud_dir"]) / "pointcloud_index.csv"
    available = set(pd.read_csv(index_path, dtype={"sample_id": str})["sample_id"])
    complete = np.asarray([[sample_id in available for sample_id in row] for row in sample_ids], dtype=bool).all(axis=1)
    if not complete.any():
        raise ValueError("No temporal sequence has all ten required point clouds")
    exclusions = identity.loc[~complete].copy()
    exclusions["exclusion_reason"] = "one_or_more_of_ten_pointcloud_timepoints_unavailable"
    kept = SequenceArrays(
        arrays.plant_uid[complete], arrays.variety_id[complete], arrays.split[complete],
        arrays.ms[complete], arrays.lidar[complete], arrays.target[complete],
    )
    return kept, sample_ids[complete], identity.loc[complete].reset_index(drop=True), exclusions.reset_index(drop=True)


def maturity_thresholds(config: dict, identity: pd.DataFrame) -> np.ndarray:
    source = str(config.get("controls", {}).get("maturity_threshold_source", "plant_specific_manual"))
    if source == "reference_variety":
        reference = pd.read_csv(resolve_repo_path(config["data"]["target_table"]), dtype={"variety_id": str})
        if "productive_tiller_number_max" not in reference:
            raise ValueError("target table is missing productive_tiller_number_max")
        lookup = reference.drop_duplicates("variety_id").set_index("variety_id")["productive_tiller_number_max"]
        return np.asarray(
            [pd.to_numeric(lookup.get(identifier_piece(variety), np.nan), errors="coerce") for variety in identity["variety_id"]],
            dtype=float,
        )
    if source != "plant_specific_manual":
        raise ValueError("controls.maturity_threshold_source must be reference_variety or plant_specific_manual")
    manual = pd.read_csv(
        resolve_repo_path(config["data"].get("manual_traits", "data/manual_traits.csv")),
        dtype={"source_area": str, "plot_id": str, "plant_id": str},
    )
    manual["maturity_productive_tiller"] = pd.to_numeric(manual["maturity_productive_tiller"], errors="coerce")
    lookup = (
        manual.dropna(subset=["maturity_productive_tiller"])
        .groupby(["source_area", "plot_id", "plant_id"], dropna=False)["maturity_productive_tiller"]
        .max()
    )
    values = []
    for row in identity.itertuples(index=False):
        key = (identifier_piece(row.source_area), identifier_piece(row.plot_id), identifier_piece(row.plant_id))
        values.append(float(lookup.get(key, np.nan)))
    return np.asarray(values, dtype=float)


def deterministic_reconstruction(
    config: dict,
    arrays: SequenceArrays,
    identity: pd.DataFrame,
    dat_values: list[float],
    predicted_tiller: np.ndarray,
    predicted_leaf_age: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    vigor_module = load_numbered_script("vigor_score_for_controls", "05_vigor_score.py")
    reference = pd.read_csv(resolve_repo_path(config["data"]["target_table"]))
    a_values = pd.to_numeric(reference["a_days"], errors="coerce").dropna()
    b_values = pd.to_numeric(reference["b_days"], errors="coerce").dropna()
    if a_values.nunique() < 2 or b_values.nunique() < 2:
        raise ValueError("Reference a_days or b_days has a degenerate normalization range")
    a_min, a_max = float(a_values.min()), float(a_values.max())
    b_min, b_max = float(b_values.min()), float(b_values.max())
    no_attainment = float(config.get("controls", {}).get("no_attainment_time", 40.0))
    times = np.asarray(dat_values, dtype=float)
    thresholds = maturity_thresholds(config, identity)
    scores = np.full(len(arrays.target), np.nan, dtype=float)
    component_rows = []
    for index, (plant_uid, threshold) in enumerate(zip(arrays.plant_uid, thresholds)):
        if not np.isfinite(threshold):
            component_rows.append({"plant_uid": plant_uid, "exclusion_reason": "missing plant-specific maturity productive-tiller count"})
            continue
        a_days, attained = vigor_module.first_attainment_time(predicted_tiller[index], threshold, times, no_attainment)
        leaf_rate = vigor_module.derive_leaf_acceleration(predicted_leaf_age[index], times)
        b_days = float(times[int(np.nanargmax(leaf_rate))])
        normalized_a = (a_days - a_min) / (a_max - a_min)
        normalized_b = (b_days - b_min) / (b_max - b_min)
        score = 1.0 - 0.5 * (normalized_a + normalized_b)
        scores[index] = score
        component_rows.append(
            {
                "plant_uid": plant_uid,
                "maturity_productive_tiller": threshold,
                "predicted_a_days": a_days,
                "threshold_attained": attained,
                "predicted_b_days": b_days,
                "normalized_a": normalized_a,
                "normalized_b": normalized_b,
                "predicted_vigor_score": score,
                "exclusion_reason": "",
            }
        )
    return scores, pd.DataFrame(component_rows)


def deterministic_tables(arrays: SequenceArrays, predicted: np.ndarray, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows, metric_rows = [], []
    for split in ("train", "internal_validation", "held_out_replicate_test"):
        mask = (arrays.split == split) & np.isfinite(predicted) & np.isfinite(arrays.target)
        observed, estimate = arrays.target[mask], predicted[mask]
        if len(observed) < 2:
            raise ValueError(f"Too few deterministic-control rows for {split}")
        metric_rows.append(
            {
                "control": "deterministic_reconstruction",
                "model": "deterministic_reconstruction",
                "seed": seed,
                "split": split,
                "branch": "deterministic",
                "sample_count": len(observed),
                **regression_metrics(observed, estimate),
            }
        )
        prediction_rows.extend(
            {
                "control": "deterministic_reconstruction",
                "seed": seed,
                "split": split,
                "plant_uid": uid,
                "observed": float(y),
                "predicted": float(y_hat),
            }
            for uid, y, y_hat in zip(arrays.plant_uid[mask], observed, estimate)
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)


def run_seed(config: dict, pointcloud_config: dict, seed: int) -> pd.DataFrame:
    multimodal_training = load_numbered_script("multimodal_training_for_controls", "06_multimodal_model.py")
    arrays, dat_values = load_sequences(config)
    sample_ids, identity = sequence_pointcloud_ids(config, arrays, dat_values)
    arrays, sample_ids, identity, exclusions = keep_complete_pointcloud_sequences(
        arrays, sample_ids, identity, pointcloud_config
    )
    descriptor_target = str(config.get("controls", {}).get("descriptor_target", "tiller"))
    if descriptor_target not in {"tiller", "leaf_age"}:
        raise ValueError("controls.descriptor_target must be tiller or leaf_age")
    predicted_tiller, tiller_descriptor, tiller_checkpoint = pointnet_predictions_and_descriptors(
        config, pointcloud_config, "tiller", seed, sample_ids, descriptor_target == "tiller"
    )
    predicted_leaf, leaf_descriptor, leaf_checkpoint = pointnet_predictions_and_descriptors(
        config, pointcloud_config, "leaf_age", seed, sample_ids, descriptor_target == "leaf_age"
    )
    descriptor = tiller_descriptor if descriptor_target == "tiller" else leaf_descriptor
    if descriptor is None:
        raise RuntimeError("The configured PointNet descriptor was not extracted")

    plant_height = arrays.lidar[..., :1]
    variants = {
        "full_fusion": replace_lidar(arrays, np.concatenate((plant_height, predicted_tiller[..., None], predicted_leaf[..., None]), axis=-1)),
        "independent_feature": replace_lidar(arrays, plant_height),
        "learned_structure": replace_lidar(arrays, np.concatenate((plant_height, descriptor), axis=-1)),
    }
    output_root = resolve_repo_path(config.get("output_dir", "outputs/multimodal")) / "target_controls" / f"seed_{seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    exclusions.to_csv(output_root / "excluded_incomplete_pointcloud_sequences.csv", index=False)
    metric_tables = []
    for control, control_arrays in variants.items():
        metrics = multimodal_training.run_arrays(
            config,
            "ogr",
            seed,
            control_arrays,
            dat_values,
            output_root / control,
            control,
        )
        metrics = metrics.loc[metrics["branch"].eq("Fusion")].copy()
        metrics["control"] = control
        metric_tables.append(metrics)

    deterministic, components = deterministic_reconstruction(
        config, arrays, identity, dat_values, predicted_tiller, predicted_leaf
    )
    deterministic_metrics, deterministic_predictions = deterministic_tables(arrays, deterministic, seed)
    deterministic_dir = output_root / "deterministic_reconstruction"
    deterministic_dir.mkdir(parents=True, exist_ok=True)
    deterministic_predictions.to_csv(deterministic_dir / "predictions.csv", index=False)
    deterministic_metrics.to_csv(deterministic_dir / "metrics.csv", index=False)
    components.to_csv(deterministic_dir / "reconstructed_components.csv", index=False)
    metric_tables.append(deterministic_metrics)

    mapping = pd.DataFrame(
        {
            "plant_uid": np.repeat(arrays.plant_uid, len(dat_values)),
            "DAT": np.tile(dat_values, len(arrays.plant_uid)),
            "pointcloud_sample_id": sample_ids.reshape(-1),
            "predicted_tiller": predicted_tiller.reshape(-1),
            "predicted_leaf_age": predicted_leaf.reshape(-1),
        }
    )
    mapping.to_csv(output_root / "pointnet_derived_inputs.csv", index=False)
    np.savez_compressed(
        output_root / "pointnet_descriptors.npz",
        plant_uid=arrays.plant_uid.astype(str),
        DAT=np.asarray(dat_values, dtype=np.float32),
        descriptor=descriptor.astype(np.float32),
        descriptor_target=np.asarray(descriptor_target),
    )
    write_json(
        output_root / "control_metadata.json",
        {
            "seed": seed,
            "descriptor_target": descriptor_target,
            "tiller_checkpoint": str(tiller_checkpoint.relative_to(REPO_ROOT)),
            "tiller_checkpoint_sha256": sha256_file(tiller_checkpoint),
            "leaf_age_checkpoint": str(leaf_checkpoint.relative_to(REPO_ROOT)),
            "leaf_age_checkpoint_sha256": sha256_file(leaf_checkpoint),
            "deterministic_uses_multispectral": False,
            "deterministic_uses_measured_maturity_threshold": True,
            "maturity_threshold_source": str(config.get("controls", {}).get("maturity_threshold_source", "plant_specific_manual")),
            "independent_feature_lidar_columns": ["plant_height"],
            "learned_structure_lidar_columns": ["plant_height", f"{descriptor_target}_pointnet_descriptor_1024d"],
            "included_sequence_count": int(len(arrays.target)),
            "excluded_incomplete_pointcloud_sequence_count": int(len(exclusions)),
        },
    )
    result = pd.concat(metric_tables, ignore_index=True)
    result.to_csv(output_root / "metrics.csv", index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the manuscript target-reconstruction and input-dependence controls")
    parser.add_argument("--config", default="config/multimodal.yaml")
    parser.add_argument("--pointcloud-config", default="config/pointcloud.yaml")
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--all-seeds", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    pointcloud_config = load_yaml(args.pointcloud_config)
    config.setdefault("data", {})["manual_traits"] = pointcloud_config["data"]["manual_traits"]
    seeds = config["training"]["seeds"] if args.all_seeds else (args.seed or [config["training"]["seeds"][0]])
    result = pd.concat([run_seed(config, pointcloud_config, int(seed)) for seed in seeds], ignore_index=True)
    output = resolve_repo_path(config.get("output_dir", "outputs/multimodal")) / "target_controls"
    result.to_csv(output / "results_by_seed.csv", index=False)
    numeric = ["r2", "rmse", "rrmse_percent", "me_pred_minus_true"]
    summary = (
        result.loc[result["split"].eq("held_out_replicate_test")]
        .groupby("control")[numeric]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(output / "mean_results.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
