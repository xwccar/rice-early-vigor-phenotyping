from __future__ import annotations

from collections.abc import Iterable

from .dgcnn import DGCNNRegressor
from .fusion import DATLateFusionRegressor
from .pct import PCTRegressor
from .point_transformer import PointTransformerRegressor
from .pointnet import PointNetRegressor, feature_transform_regularizer
from .pointnet2 import PointNet2Regressor


MODEL_ORDER = ("pointnet", "pointnet2", "dgcnn", "pct", "pointtransformer")
ALIASES = {
    "pointnet": "pointnet", "pointnet2": "pointnet2", "pointnet++": "pointnet2",
    "pointnetplusplus": "pointnet2", "dgcnn": "dgcnn", "pct": "pct",
    "pointtransformer": "pointtransformer", "point_transformer": "pointtransformer", "pt": "pointtransformer",
}


def canonical_model_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "").replace(" ", "")
    if normalized not in ALIASES:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_ORDER}")
    return ALIASES[normalized]


def parse_model_names(values: Iterable[str]) -> list[str]:
    result = []
    for value in values:
        for item in value.split(","):
            if item.strip():
                name = canonical_model_name(item)
                if name not in result:
                    result.append(name)
    return result


def build_model(name: str, dgcnn_k: int = 20, transformer_k: int = 16, use_dat: bool = False, dat_mean: float = 0.0, dat_std: float = 1.0):
    canonical = canonical_model_name(name)
    if canonical == "pointnet":
        backbone = PointNetRegressor()
    elif canonical == "pointnet2":
        backbone = PointNet2Regressor()
    elif canonical == "dgcnn":
        backbone = DGCNNRegressor(k=dgcnn_k)
    elif canonical == "pct":
        backbone = PCTRegressor(neighbors=transformer_k)
    else:
        backbone = PointTransformerRegressor(neighbors=transformer_k)
    return DATLateFusionRegressor(backbone, dat_mean, dat_std) if use_dat else backbone


def count_trainable_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
