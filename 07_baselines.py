from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from xgboost import XGBRegressor

from multimodal_core import SequencePreprocessor, build_optimizer, load_sequences, make_loader, take
from release_utils import choose_device, load_yaml, regression_metrics, resolve_repo_path, set_reproducible_seed, sha256_file, write_json


MODEL_ORDER = ("XGBoost", "GRU", "TCN", "Transformer", "Early-fusion LSTM")


class GRURegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 48):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 24),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(24, 1),
        )

    def forward(self, values):
        sequence, _ = self.gru(values)
        return torch.sigmoid(self.head(sequence[:, -1]))


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation), nn.ReLU(), nn.Dropout(0.15),
            nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation), nn.ReLU(),
        )

    def forward(self, values):
        return values + self.block(values)


class TCNRegressor(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.input = nn.Conv1d(input_size, 48, 1)
        self.blocks = nn.Sequential(*(ResidualTCNBlock(48, dilation) for dilation in (1, 2, 4)))
        self.head = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, values):
        features = self.blocks(self.input(values.transpose(1, 2))).mean(dim=2)
        return torch.sigmoid(self.head(features))


class TransformerRegressor(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.project = nn.Linear(input_size, 32)
        self.position = nn.Parameter(torch.zeros(1, 10, 32))
        layer = nn.TransformerEncoderLayer(32, 4, 128, dropout=0.15, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, values):
        encoded = self.encoder(self.project(values) + self.position[:, : values.shape[1]])
        return torch.sigmoid(self.head(encoded.mean(1)))


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.energy = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, values):
        weights = torch.softmax(self.score(torch.tanh(self.energy(values))).squeeze(-1), dim=1)
        return torch.sum(values * weights.unsqueeze(-1), dim=1)


class EarlyFusionLSTM(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 48, batch_first=True)
        self.pool = AttentionPooling(48)
        self.head = nn.Sequential(nn.Linear(48, 24), nn.ReLU(), nn.Dropout(0.15), nn.Linear(24, 1))

    def forward(self, values):
        sequence, _ = self.lstm(values)
        return torch.sigmoid(self.head(self.pool(sequence)))


def build_model(name: str, input_size: int):
    return {
        "GRU": GRURegressor, "TCN": TCNRegressor,
        "Transformer": TransformerRegressor, "Early-fusion LSTM": EarlyFusionLSTM,
    }[name](input_size)


def combined_batches(loader, device):
    for ms, lidar, target, plant_uid in loader:
        yield torch.cat((ms, lidar), dim=-1).to(device), target.to(device), plant_uid


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    observed, predicted, identifiers = [], [], []
    for values, target, plant_uid in combined_batches(loader, device):
        identifiers.extend(plant_uid)
        observed.extend(target.cpu().numpy().ravel())
        predicted.extend(model(values).cpu().numpy().ravel())
    return np.asarray(identifiers), np.asarray(observed), np.asarray(predicted)


def train_neural(name, input_size, train_loader, validation_loader, device, config, seed):
    set_reproducible_seed(seed)
    model = build_model(name, input_size).to(device)
    optimizer = build_optimizer(model.parameters(), config)
    best_loss, stale, best_state = float("inf"), 0, None
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        for values, target, _ in combined_batches(train_loader, device):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(values), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
        _, observed, predicted = predict(model, validation_loader, device)
        validation_loss = float(np.mean((predicted - observed) ** 2))
        if validation_loss < best_loss:
            best_loss, stale, best_state = validation_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch >= int(config["min_epochs"]) and stale >= int(config["patience"]):
            break
    model.load_state_dict(best_state)
    return model


def run_one(config: dict, name: str, seed: int) -> pd.DataFrame:
    all_data, _ = load_sequences(config)
    raw = {split: take(all_data, all_data.split == split) for split in np.unique(all_data.split)}
    scaler = SequencePreprocessor.fit(raw["train"].ms, raw["train"].lidar)
    data = {split: scaler.transform(value) for split, value in raw.items()}
    training = config["training"]
    loaders = {split: make_loader(value, int(training["batch_size"]), split == "train", seed) for split, value in data.items()}
    device = choose_device(training.get("device", "auto"))
    set_reproducible_seed(seed)
    if name == "XGBoost":
        train_x = np.concatenate((data["train"].ms, data["train"].lidar), axis=-1).reshape(len(data["train"].target), -1)
        validation_x = np.concatenate((data["internal_validation"].ms, data["internal_validation"].lidar), axis=-1).reshape(len(data["internal_validation"].target), -1)
        model = XGBRegressor(
            objective="reg:squarederror", n_estimators=1500, learning_rate=0.02, max_depth=3,
            min_child_weight=2, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
            random_state=seed, n_jobs=4, early_stopping_rounds=100,
        )
        model.fit(train_x, data["train"].target, eval_set=[(validation_x, data["internal_validation"].target)], verbose=False)
    else:
        input_size = data["train"].ms.shape[-1] + data["train"].lidar.shape[-1]
        model = train_neural(name, input_size, loaders["train"], loaders["internal_validation"], device, training, seed)
    rows, metrics = [], []
    for split, values in data.items():
        if name == "XGBoost":
            x = np.concatenate((values.ms, values.lidar), axis=-1).reshape(len(values.target), -1)
            identifiers, observed, predicted = values.plant_uid, values.target, np.clip(model.predict(x), 0, 1)
        else:
            identifiers, observed, predicted = predict(model, loaders[split], device)
        rows.append(pd.DataFrame({"model": name, "seed": seed, "split": split, "plant_uid": identifiers, "observed": observed, "predicted": predicted}))
        metrics.append({"model": name, "seed": seed, "split": split, "sample_count": len(observed), **regression_metrics(observed, predicted)})
    output = resolve_repo_path(config.get("output_dir", "outputs/multimodal")) / "baselines" / name.replace(" ", "_") / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(output / "predictions.csv", index=False)
    result = pd.DataFrame(metrics)
    result.to_csv(output / "metrics.csv", index=False)
    if name == "XGBoost":
        checkpoint_path = output / "best_model.json"
        model.save_model(checkpoint_path)
    else:
        checkpoint_path = output / "best_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model": name,
                "seed": seed,
                "config": config,
                "preprocessor": scaler.state_dict(),
            },
            checkpoint_path,
        )
    write_json(
        output / "run_metadata.json",
        {
            "model": name,
            "seed": seed,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "optimizer": "xgboost" if name == "XGBoost" else str(training.get("optimizer", "sgd")).lower(),
            "checkpoint_selection_split": "internal_validation",
            "final_evaluation_split": "held_out_replicate_test",
        },
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Train temporal and tree baselines")
    parser.add_argument("--config", default="config/multimodal.yaml")
    parser.add_argument("--model", action="append", choices=MODEL_ORDER, default=[])
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument("--all-seeds", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml(args.config)
    seeds = config["training"]["seeds"] if args.all_seeds else (args.seed or [config["training"]["seeds"][0]])
    names = args.model or list(MODEL_ORDER)
    result = pd.concat([run_one(config, name, int(seed)) for seed in seeds for name in names], ignore_index=True)
    output = resolve_repo_path(config.get("output_dir", "outputs/multimodal")) / "baselines"
    result.to_csv(output / "results_by_seed.csv", index=False)
    print(result.query("split == 'held_out_replicate_test'").to_string(index=False))


if __name__ == "__main__":
    main()
