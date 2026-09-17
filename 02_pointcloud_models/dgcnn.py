from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import RegressionHead, index_points, knn_indices


def dynamic_graph_feature(features: torch.Tensor, k: int) -> torch.Tensor:
    values = features.transpose(1, 2).contiguous()
    _, indices = knn_indices(values, values, k)
    neighbors = index_points(values, indices)
    centers = values.unsqueeze(2).expand_as(neighbors)
    return torch.cat((neighbors - centers, centers), dim=-1).permute(0, 3, 1, 2).contiguous()


def edge_conv(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


class DGCNNRegressor(nn.Module):
    def __init__(self, k: int = 20, embedding_channels: int = 1024):
        super().__init__()
        self.k = k
        self.edge1, self.edge2 = edge_conv(6, 64), edge_conv(128, 64)
        self.edge3, self.edge4 = edge_conv(128, 128), edge_conv(256, 256)
        self.embedding = nn.Sequential(
            nn.Conv1d(512, embedding_channels, 1, bias=False),
            nn.BatchNorm1d(embedding_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = RegressionHead(embedding_channels * 2, dropout=0.5)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        f1 = self.edge1(dynamic_graph_feature(points, self.k)).amax(-1)
        f2 = self.edge2(dynamic_graph_feature(f1, self.k)).amax(-1)
        f3 = self.edge3(dynamic_graph_feature(f2, self.k)).amax(-1)
        f4 = self.edge4(dynamic_graph_feature(f3, self.k)).amax(-1)
        features = self.embedding(torch.cat((f1, f2, f3, f4), dim=1))
        global_features = torch.cat((F.adaptive_max_pool1d(features, 1), F.adaptive_avg_pool1d(features, 1)), dim=1).flatten(1)
        return self.head(global_features)
