from __future__ import annotations

import torch
import torch.nn as nn

from .common import RegressionHead, TransitionDown, index_points, knn_indices


class PointTransformerBlock(nn.Module):
    def __init__(self, channels: int, neighbors: int = 16):
        super().__init__()
        self.neighbors = neighbors
        self.pre = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, channels), nn.GELU())
        self.query, self.key, self.value = (
            nn.Linear(channels, channels, bias=False),
            nn.Linear(channels, channels, bias=False),
            nn.Linear(channels, channels, bias=False),
        )
        self.position = nn.Sequential(nn.Linear(3, channels), nn.GELU(), nn.Linear(channels, channels))
        self.attention = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))
        self.output = nn.Sequential(nn.Linear(channels, channels), nn.Dropout(0.1))
        self.norm = nn.LayerNorm(channels)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        residual, normalized = features, self.pre(features)
        _, indices = knn_indices(xyz, xyz, self.neighbors)
        relative = xyz.unsqueeze(2) - index_points(xyz, indices)
        position = self.position(relative)
        query = self.query(normalized).unsqueeze(2)
        keys = index_points(self.key(normalized), indices)
        values = index_points(self.value(normalized), indices)
        attention = torch.softmax(self.attention(query - keys + position), dim=2)
        return self.norm(residual + self.output(torch.sum(attention * (values + position), dim=2)))


class PointTransformerRegressor(nn.Module):
    def __init__(self, neighbors: int = 16):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(3, 64, bias=False), nn.LayerNorm(64), nn.GELU())
        self.block1 = PointTransformerBlock(64, neighbors)
        self.down1 = TransitionDown(256, neighbors, 64, 128)
        self.block2 = PointTransformerBlock(128, neighbors)
        self.down2 = TransitionDown(64, neighbors, 128, 256)
        self.block3 = PointTransformerBlock(256, neighbors)
        self.block4 = PointTransformerBlock(256, neighbors)
        self.head = RegressionHead(512, dropout=0.3)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points.transpose(1, 2).contiguous()
        features = self.block1(xyz, self.stem(xyz))
        xyz, features = self.down1(xyz, features)
        features = self.block2(xyz, features)
        xyz, features = self.down2(xyz, features)
        features = self.block4(xyz, self.block3(xyz, features))
        return self.head(torch.cat((features.amax(1), features.mean(1)), dim=1))
