from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .common import RegressionHead, ball_query, index_points, sample_centroids


class SharedMLP2d(nn.Module):
    def __init__(self, channels: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:]):
            layers += [
                nn.Conv2d(input_channels, output_channels, 1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            ]
        self.layers = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class SetAbstractionMSG(nn.Module):
    def __init__(
        self,
        number: int,
        radii: Sequence[float],
        neighbors: Sequence[int],
        input_channels: int,
        mlps: Sequence[Sequence[int]],
    ):
        super().__init__()
        if not (len(radii) == len(neighbors) == len(mlps)):
            raise ValueError("radii, neighbors, and mlps must have the same length")
        self.number, self.radii, self.neighbors = number, tuple(radii), tuple(neighbors)
        self.mlps = nn.ModuleList(SharedMLP2d([input_channels + 3, *channels]) for channels in mlps)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor | None):
        new_xyz, _ = sample_centroids(xyz, self.number)
        outputs = []
        for radius, number, mlp in zip(self.radii, self.neighbors, self.mlps):
            indices = ball_query(new_xyz, xyz, radius, number)
            relative = index_points(xyz, indices) - new_xyz.unsqueeze(2)
            grouped = relative if features is None else torch.cat((relative, index_points(features, indices)), dim=-1)
            outputs.append(mlp(grouped.permute(0, 3, 2, 1).contiguous()).amax(dim=2))
        return new_xyz, torch.cat(outputs, dim=1).transpose(1, 2).contiguous()


class PointNet2Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = SetAbstractionMSG(
            128, (0.10, 0.20, 0.40), (16, 32, 64), 0,
            ((32, 32, 64), (64, 64, 128), (64, 96, 128)),
        )
        self.stage2 = SetAbstractionMSG(
            64, (0.20, 0.40, 0.80), (32, 64, 128), 320,
            ((64, 64, 128), (128, 128, 256), (128, 128, 256)),
        )
        self.global_mlp = SharedMLP2d((643, 256, 512, 1024))
        self.head = RegressionHead(1024, dropout=0.4)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points.transpose(1, 2).contiguous()
        xyz1, features1 = self.stage1(xyz, None)
        xyz2, features2 = self.stage2(xyz1, features1)
        values = torch.cat((xyz2, features2), dim=-1).transpose(1, 2).unsqueeze(2)
        return self.head(self.global_mlp(values).amax(dim=-1).squeeze(-1))
