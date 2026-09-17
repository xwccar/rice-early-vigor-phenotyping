from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from release_utils import resolve_repo_path, validate_three_way_splits


@dataclass
class SequenceArrays:
    plant_uid: np.ndarray
    variety_id: np.ndarray
    split: np.ndarray
    ms: np.ndarray
    lidar: np.ndarray
    target: np.ndarray


@dataclass
class SequencePreprocessor:
    ms_impute: np.ndarray
    lidar_impute: np.ndarray
    ms_mean: np.ndarray
    ms_std: np.ndarray
    lidar_mean: np.ndarray
    lidar_std: np.ndarray

    @classmethod
    def fit(cls, ms: np.ndarray, lidar: np.ndarray) -> "SequencePreprocessor":
        ms_impute = np.nanmean(ms, axis=(0, 1))
        lidar_impute = np.nanmean(lidar, axis=(0, 1))
        if not np.isfinite(ms_impute).all() or not np.isfinite(lidar_impute).all():
            raise ValueError("At least one configured feature is entirely missing in the training split")
        ms_filled = np.where(np.isfinite(ms), ms, ms_impute.reshape(1, 1, -1))
        lidar_filled = np.where(np.isfinite(lidar), lidar, lidar_impute.reshape(1, 1, -1))
        ms_mean, ms_std = ms_filled.mean((0, 1)), ms_filled.std((0, 1))
        lidar_mean, lidar_std = lidar_filled.mean((0, 1)), lidar_filled.std((0, 1))
        return cls(ms_impute, lidar_impute, ms_mean, np.where(ms_std > 1e-12, ms_std, 1.0), lidar_mean, np.where(lidar_std > 1e-12, lidar_std, 1.0))

    def transform(self, arrays: SequenceArrays) -> SequenceArrays:
        ms = np.where(np.isfinite(arrays.ms), arrays.ms, self.ms_impute.reshape(1, 1, -1))
        lidar = np.where(np.isfinite(arrays.lidar), arrays.lidar, self.lidar_impute.reshape(1, 1, -1))
        return SequenceArrays(
            arrays.plant_uid, arrays.variety_id, arrays.split,
            ((ms - self.ms_mean) / self.ms_std).astype(np.float32),
            ((lidar - self.lidar_mean) / self.lidar_std).astype(np.float32),
            arrays.target.astype(np.float32),
        )

    def state_dict(self) -> dict[str, list[float]]:
        """Return preprocessing parameters in a checkpoint-safe form."""
        return {
            name: np.asarray(getattr(self, name), dtype=float).tolist()
            for name in (
                "ms_impute", "lidar_impute", "ms_mean", "ms_std",
                "lidar_mean", "lidar_std",
            )
        }


def take(data: SequenceArrays, mask: np.ndarray) -> SequenceArrays:
    return SequenceArrays(data.plant_uid[mask], data.variety_id[mask], data.split[mask], data.ms[mask], data.lidar[mask], data.target[mask])


def load_sequences(config: dict) -> tuple[SequenceArrays, list[float]]:
    data_config = config["data"]
    frame = pd.read_csv(resolve_repo_path(data_config["model_inputs"]), dtype={"plant_uid": str})
    split_table = pd.read_csv(resolve_repo_path(data_config["splits"]), dtype={"sample_id": str})
    validate_three_way_splits(split_table)
    if "entity_type" not in split_table:
        raise ValueError("splits.csv is missing entity_type")
    temporal_splits = split_table.loc[split_table["entity_type"].eq("temporal_sequence"), ["sample_id", "split"]]
    if temporal_splits.empty:
        raise ValueError("splits.csv does not contain temporal_sequence rows")
    declared = frame[["plant_uid", "split"]].drop_duplicates()
    verified = declared.merge(temporal_splits, left_on="plant_uid", right_on="sample_id", how="left", suffixes=("_inputs", "_authoritative"), validate="one_to_one")
    mismatch = verified["split_authoritative"].isna() | verified["split_inputs"].ne(verified["split_authoritative"])
    if mismatch.any():
        raise ValueError("model_inputs.csv and the authoritative temporal splits in splits.csv disagree")
    target_column = data_config.get("target_column", "vigor_score")
    required = {"plant_uid", "variety_id", "split", "DAT", target_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"model_inputs.csv is missing: {sorted(missing)}")
    reference = pd.read_csv(resolve_repo_path(data_config["target_table"]), dtype={"plant_uid": str})
    if target_column not in reference:
        raise ValueError(f"target table is missing {target_column}")
    if reference["plant_uid"].duplicated().any():
        raise ValueError("Reference targets must be unique per plant, not per variety")
    reference = reference[["plant_uid", target_column]].dropna()
    target_check = frame[["plant_uid", target_column]].copy()
    target_check = target_check.merge(reference, on="plant_uid", how="left", suffixes=("_inputs", "_reference"), validate="many_to_one")
    left = pd.to_numeric(target_check[f"{target_column}_inputs"], errors="coerce")
    right = pd.to_numeric(target_check[f"{target_column}_reference"], errors="coerce")
    mismatch = left.notna() & (right.isna() | ~np.isclose(left, right, rtol=0.0, atol=1e-10))
    if mismatch.any():
        raise ValueError("model_inputs.csv target values disagree with the authoritative target table")
    ms_features, lidar_features = list(data_config["multispectral_features"]), list(data_config["lidar_features"])
    missing_features = set(ms_features + lidar_features) - set(frame.columns)
    if missing_features:
        raise ValueError(f"Configured features are missing: {sorted(missing_features)}")
    dat_values = [float(value) for value in data_config["dat_values"]]
    rows = []
    for plant_uid, group in frame.groupby("plant_uid", sort=True):
        group = group.copy()
        group["DAT"] = pd.to_numeric(group["DAT"], errors="coerce")
        if group["DAT"].duplicated().any():
            raise ValueError(f"Duplicate DAT rows for {plant_uid}")
        group = group.set_index("DAT").reindex(dat_values)
        identity = frame.loc[frame["plant_uid"].eq(plant_uid)].iloc[0]
        target = pd.to_numeric(frame.loc[frame["plant_uid"].eq(plant_uid), target_column], errors="coerce").dropna().unique()
        if len(target) != 1:
            continue
        rows.append((plant_uid, identity["variety_id"], identity["split"], group[ms_features].apply(pd.to_numeric, errors="coerce").to_numpy(float), group[lidar_features].apply(pd.to_numeric, errors="coerce").to_numpy(float), float(target[0])))
    if not rows:
        raise ValueError("No complete target sequences were found")
    data = SequenceArrays(
        np.array([row[0] for row in rows]), np.array([row[1] for row in rows]), np.array([row[2] for row in rows]),
        np.stack([row[3] for row in rows]), np.stack([row[4] for row in rows]), np.array([row[5] for row in rows], dtype=np.float32),
    )
    allowed = {"train", "internal_validation", "held_out_replicate_test"}
    if set(data.split) != allowed:
        raise ValueError(f"Sequence splits must contain exactly {sorted(allowed)}; got {sorted(set(data.split))}")
    return data, dat_values


class SequenceDataset(Dataset):
    def __init__(self, data: SequenceArrays):
        self.data = data

    def __len__(self):
        return len(self.data.target)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.data.ms[index]), torch.from_numpy(self.data.lidar[index]),
            torch.tensor(self.data.target[index]).reshape(1), self.data.plant_uid[index],
        )


def make_loader(data: SequenceArrays, batch_size: int, shuffle: bool, seed: int):
    return DataLoader(SequenceDataset(data), batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=torch.Generator().manual_seed(seed))


def build_optimizer(parameters, training: dict) -> torch.optim.Optimizer:
    """Build the optimizer declared in the manuscript-level configuration.

    The multimodal framework is described as SGD with L2 regularization.
    AdamW remains available only when an explicitly different protocol is
    declared in a separate configuration file.
    """
    name = str(training.get("optimizer", "sgd")).lower()
    learning_rate = float(training["learning_rate"])
    weight_decay = float(training["weight_decay"])
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(training.get("momentum", 0.0)),
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.energy = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, sequence: torch.Tensor):
        weights = torch.softmax(self.score(torch.tanh(self.energy(sequence))).squeeze(-1), dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1), weights


class LSTMBranch(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, representation_size: int = 16, dropout: float = 0.15):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = TemporalAttention(hidden_size)
        self.project = nn.Sequential(nn.Linear(hidden_size, representation_size), nn.ReLU(), nn.Dropout(dropout))
        self.regressor = nn.Linear(representation_size, 1)

    def forward(self, values: torch.Tensor):
        sequence, _ = self.lstm(values)
        context, attention = self.attention(sequence)
        representation = self.project(context)
        return representation, torch.sigmoid(self.regressor(representation)), attention


class MultimodalLSTMAttention(nn.Module):
    def __init__(
        self,
        ms_size: int,
        lidar_size: int,
        hidden_size: int = 16,
        representation_size: int = 16,
        fusion_hidden: tuple[int, int] = (32, 16),
        dropout: float = 0.15,
    ):
        super().__init__()
        self.ms_branch = LSTMBranch(ms_size, hidden_size, representation_size, dropout)
        self.lidar_branch = LSTMBranch(lidar_size, hidden_size, representation_size, dropout)
        self.fusion = nn.Sequential(
            nn.Linear(representation_size * 2, fusion_hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden[0], fusion_hidden[1]),
            nn.ReLU(),
            nn.Linear(fusion_hidden[1], 1),
        )

    def forward(self, ms: torch.Tensor, lidar: torch.Tensor):
        ms_repr, ms_prediction, ms_attention = self.ms_branch(ms)
        lidar_repr, lidar_prediction, lidar_attention = self.lidar_branch(lidar)
        fusion_prediction = torch.sigmoid(self.fusion(torch.cat((ms_repr, lidar_repr), dim=1)))
        return {"MS": ms_prediction, "LiDAR": lidar_prediction, "Fusion": fusion_prediction}, {"MS": ms_attention, "LiDAR": lidar_attention}
