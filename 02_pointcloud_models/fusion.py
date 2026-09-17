from __future__ import annotations

import torch
import torch.nn as nn


class DATLateFusionRegressor(nn.Module):
    fusion_version = "scalar_output_plus_standardized_DAT_linear_v1"

    def __init__(self, backbone: nn.Module, dat_mean: float, dat_std: float):
        super().__init__()
        if not torch.isfinite(torch.tensor(dat_mean)):
            raise ValueError("DAT mean must be finite")
        if not torch.isfinite(torch.tensor(dat_std)) or dat_std < 1e-12:
            raise ValueError("DAT standard deviation must be positive and finite")
        self.backbone = backbone
        self.register_buffer("dat_mean", torch.tensor(float(dat_mean), dtype=torch.float32))
        self.register_buffer("dat_std", torch.tensor(float(dat_std), dtype=torch.float32))
        self.fusion = nn.Linear(2, 1)
        with torch.no_grad():
            self.fusion.weight.copy_(torch.tensor([[1.0, 0.0]]))
            self.fusion.bias.zero_()

    def forward_with_components(self, points: torch.Tensor, dat_raw: torch.Tensor):
        cloud_scalar = self.backbone(points)
        dat_column = dat_raw.reshape(points.shape[0], 1).float()
        standardized_dat = ((dat_column - self.dat_mean) / self.dat_std).to(cloud_scalar.dtype)
        fused = self.fusion(torch.cat((cloud_scalar, standardized_dat), dim=1))
        return fused, cloud_scalar, standardized_dat

    def forward_with_feature_transform(self, points: torch.Tensor, dat_raw: torch.Tensor):
        if not hasattr(self.backbone, "forward_with_feature_transform"):
            return self(points, dat_raw), None
        cloud_scalar, feature_transform = self.backbone.forward_with_feature_transform(points)
        dat_column = dat_raw.reshape(points.shape[0], 1).float()
        standardized_dat = ((dat_column - self.dat_mean) / self.dat_std).to(cloud_scalar.dtype)
        fused = self.fusion(torch.cat((cloud_scalar, standardized_dat), dim=1))
        return fused, feature_transform

    def forward(self, points: torch.Tensor, dat_raw: torch.Tensor) -> torch.Tensor:
        return self.forward_with_components(points, dat_raw)[0]
