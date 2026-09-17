"""Fast interface checks that do not use or generate an experimental dataset."""
from __future__ import annotations

import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_numbered(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    torch.manual_seed(7)
    torch.set_num_threads(2)

    preprocessing = load_numbered("pointcloud_preprocessing", "01_pointcloud_preprocessing.py")
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_path = Path(temporary_directory)
        manifest_rows = []
        for index in range(2):
            source = temporary_path / f"plant_{index}.las"
            cloud = preprocessing.laspy.create(point_format=3, file_version="1.2")
            cloud.x = np.linspace(index, index + 1, 24)
            cloud.y = np.linspace(2 * index, 2 * index + 0.5, 24)
            cloud.z = np.linspace(0.0, 0.25, 24)
            cloud.write(source)
            manifest_rows.append(
                {
                    "sample_id": f"main:0806:1:{index + 1}",
                    "source_las": source,
                    "date": "0806",
                    "DAT": 0,
                    "source_area": "main",
                }
            )
        archive_dir = temporary_path / "pointcloud_1024"
        audit = preprocessing.build_archives(pd.DataFrame(manifest_rows), archive_dir, n_points=1024, seed=2024)
        assert audit["sample_count"].tolist() == [2]
        archives = list(archive_dir.glob("*.npz"))
        assert len(archives) == 1
        with np.load(archives[0], allow_pickle=False) as archive:
            assert set(archive.files) == {"sample_id", "xyz", "date", "DAT", "source_area"}
            assert archive["xyz"].shape == (2, 1024, 3)
            assert archive["xyz"].dtype == np.float32
            assert np.max(np.abs(archive["xyz"].mean(axis=1))) < 1e-6
        pointcloud_training = load_numbered("pointcloud_training", "03_train_pointcloud.py")
        loaded_clouds = pointcloud_training.load_fixed_pointclouds(archive_dir)
        assert set(loaded_clouds) == {"main:0806:1:1", "main:0806:1:2"}
        assert all(len(record) == 4 and record[3] == "main" for record in loaded_clouds.values())

    models = importlib.import_module("02_pointcloud_models")
    points = torch.randn(2, 3, 1024)
    for name in models.MODEL_ORDER:
        model = models.build_model(name).eval()
        with torch.inference_mode():
            prediction = model(points)
        assert prediction.shape == (2, 1) and torch.isfinite(prediction).all(), name

    dat_model = models.build_model("pointnet", use_dat=True, dat_mean=18.0, dat_std=11.49).eval()
    with torch.inference_mode():
        assert dat_model(points, torch.tensor([0.0, 36.0])).shape == (2, 1)

    regularized = models.build_model("pointnet").train()
    prediction, transform = regularized.forward_with_feature_transform(points)
    loss = prediction.square().mean() + 0.001 * models.feature_transform_regularizer(transform)
    loss.backward()
    assert any(parameter.grad is not None for parameter in regularized.parameters())

    vigor = load_numbered("vigor_score", "05_vigor_score.py")
    dates = np.arange(0, 40, 4, dtype=float)
    rows = []
    for plant in ("p1", "p2"):
        for index, dat in enumerate(dates):
            is_second = plant == "p2"
            rows.append(
                {
                    "plant_uid": plant,
                    "plant_id": plant,
                    "variety_id": plant,
                    "plot_id": plant,
                    "source_area": "main",
                    "group": "test",
                    "DAT": dat,
                    "manual_tiller": (1.2 if is_second else 1.0) * index,
                    "manual_leaf_age": ((index + 1.0) ** 2 / 10.0) if is_second else np.sqrt(index + 1.0),
                    "maturity_productive_tiller": 4 if is_second else 5,
                    "yield_g_per_plant": 21 if is_second else 20,
                }
            )
    score = vigor.build_reference(pd.DataFrame(rows))
    assert len(score) == 2 and score["vigor_score"].between(0, 1).all()
    assert set(score.columns) == {
        "plant_uid", "plant_id", "variety_id", "plot_id", "source_area", "group",
        "a_days", "b_days", "ax", "bx", "maturity_productive_tiller",
        "threshold_attained", "yield_g_per_plant", "normalized_a", "normalized_b", "vigor_score",
    }

    features = importlib.import_module("feature_engineering")
    indices = features.spectral_indices(0.2, 0.3, 0.4, 0.8)
    assert set(indices) == {"NDVI", "LCI", "GNDVI", "OSAVI", "NDRE"}
    assert features.canopy_coverage([0.2, 0.5, 0.7]) == 2 / 3
    aggregated = features.plot_spectral_features(
        red=np.array([0.2, 0.4]),
        green=np.array([0.3, 0.4]),
        red_edge=np.array([0.4, 0.5]),
        nir=np.array([0.8, 0.5]),
    )
    assert set(aggregated) == {"NDVI", "LCI", "GNDVI", "OSAVI", "NDRE", "canopy_coverage"}

    multimodal = importlib.import_module("multimodal_core")
    network = multimodal.MultimodalLSTMAttention(6, 3).eval()
    with torch.inference_mode():
        predictions, attention = network(torch.randn(2, 10, 6), torch.randn(2, 10, 3))
    assert all(value.shape == (2, 1) for value in predictions.values())
    assert all(torch.allclose(value.sum(1), torch.ones(2), atol=1e-6) for value in attention.values())

    trainer = load_numbered("multimodal_training", "06_multimodal_model.py")
    weights = trainer.inverse_ogr_weights(
        {"MS": 1.0, "LiDAR": 1.0, "Fusion": 1.0},
        {"MS": 2.0, "LiDAR": 1.0, "Fusion": 0.5},
        1e-8,
    )
    assert np.isclose(weights.sum(), 1.0) and weights[2] > weights[1] > weights[0]

    baselines = load_numbered("temporal_baselines", "07_baselines.py")
    sequence = torch.randn(2, 10, 9)
    for name in ("GRU", "TCN", "Transformer", "Early-fusion LSTM"):
        baseline = baselines.build_model(name, 9).eval()
        with torch.inference_mode():
            assert baseline(sequence).shape == (2, 1), name
    print("Smoke test passed: preprocessing, five point-cloud backbones, DAT fusion, vigor score, temporal models, and OGR weights.")


if __name__ == "__main__":
    main()
