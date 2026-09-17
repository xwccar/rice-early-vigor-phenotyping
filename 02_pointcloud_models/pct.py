from __future__ import annotations

import torch
import torch.nn as nn

from .common import RegressionHead, TransitionDown


class OffsetAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        reduced = max(16, channels // 4)
        self.query = nn.Linear(channels, reduced, bias=False)
        self.key = nn.Linear(channels, reduced, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.transform = nn.Sequential(nn.Linear(channels, channels, bias=False), nn.LayerNorm(channels), nn.GELU())
        self.scale = reduced**-0.5

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        energy = torch.matmul(self.query(features), self.key(features).transpose(1, 2)) * self.scale
        attended = torch.matmul(torch.softmax(energy, dim=-1), self.value(features))
        return features + self.transform(features - attended)


class PCTRegressor(nn.Module):
    def __init__(self, neighbors: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(3, 64, bias=False), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64, bias=False), nn.LayerNorm(64), nn.GELU(),
        )
        self.embedding1 = TransitionDown(256, neighbors, 64, 128)
        self.embedding2 = TransitionDown(128, neighbors, 128, 256)
        self.attention_layers = nn.ModuleList(OffsetAttention(256) for _ in range(4))
        self.fuse = nn.Sequential(nn.Linear(1024, 1024, bias=False), nn.LayerNorm(1024), nn.LeakyReLU(0.1, inplace=True))
        self.head = RegressionHead(2048, dropout=0.4)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points.transpose(1, 2).contiguous()
        xyz, features = self.embedding1(xyz, self.stem(xyz))
        xyz, features = self.embedding2(xyz, features)
        outputs = []
        for layer in self.attention_layers:
            features = layer(features)
            outputs.append(features)
        fused = self.fuse(torch.cat(outputs, dim=-1))
        return self.head(torch.cat((fused.amax(1), fused.mean(1)), dim=1))
