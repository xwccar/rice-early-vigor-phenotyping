from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import r2_score


REPO_ROOT = Path(__file__).resolve().parent


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_yaml(path: str | Path) -> dict:
    with resolve_repo_path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_rows(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    ordered = frame.loc[:, list(columns)].sort_values(list(columns), kind="stable")
    payload = ordered.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def regression_metrics(observed: Iterable[float], predicted: Iterable[float]) -> dict:
    y = np.asarray(list(observed), dtype=np.float64)
    p = np.asarray(list(predicted), dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if y.size < 2:
        raise ValueError("At least two finite observation-prediction pairs are required")
    error = p - y
    rmse = float(np.sqrt(np.mean(error**2)))
    denominator = abs(float(np.mean(y)))
    return {
        "sample_count": int(y.size),
        "r2": float(r2_score(y, p)),
        "rmse": rmse,
        "rrmse_percent": float(100.0 * rmse / denominator) if denominator > 0 else np.nan,
        "me_pred_minus_true": float(np.mean(error)),
    }


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, axis: int | tuple[int, ...] = 0) -> "Standardizer":
        mean = np.nanmean(values, axis=axis, keepdims=False)
        scale = np.nanstd(values, axis=axis, keepdims=False)
        scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
        return cls(np.asarray(mean), np.asarray(scale))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean

    def to_dict(self) -> dict:
        return {"mean": np.asarray(self.mean).tolist(), "scale": np.asarray(self.scale).tolist()}


def write_json(path: str | Path, value: Mapping | Sequence) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_three_way_splits(frame: pd.DataFrame) -> None:
    required = {"sample_id", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split file is missing columns: {sorted(missing)}")
    allowed = {"train", "internal_validation", "held_out_replicate_test"}
    unexpected = set(frame["split"].dropna().unique()) - allowed
    if unexpected:
        raise ValueError(f"Unexpected split labels: {sorted(unexpected)}")
    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[frame["sample_id"].duplicated(False), "sample_id"].head().tolist()
        raise ValueError(f"Duplicate sample_id values in split table: {duplicates}")
    missing_splits = allowed - set(frame["split"].unique())
    if missing_splits:
        raise ValueError(f"Split file does not contain: {sorted(missing_splits)}")
