from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = points.shape[0]
    view = [batch] + [1] * (indices.ndim - 1)
    batch_index = torch.arange(batch, device=points.device).view(view).expand_as(indices)
    return points[batch_index, indices]


def square_distance(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    source32, target32 = source.float(), target.float()
    distance = -2.0 * torch.matmul(source32, target32.transpose(1, 2))
    distance += torch.sum(source32 * source32, dim=-1, keepdim=True)
    distance += torch.sum(target32 * target32, dim=-1).unsqueeze(1)
    return distance.clamp_min_(0.0)


@torch.no_grad()
def knn_indices(query: torch.Tensor, support: torch.Tensor, k: int):
    k = min(max(1, int(k)), support.shape[1])
    return torch.topk(square_distance(query, support), k=k, dim=-1, largest=False, sorted=False)


@torch.no_grad()
def farthest_point_sample(xyz: torch.Tensor, number: int) -> torch.Tensor:
    batch, total, _ = xyz.shape
    number = min(max(1, int(number)), total)
    coordinates = xyz.float()
    centroids = torch.empty(batch, number, dtype=torch.long, device=xyz.device)
    minimum = torch.full((batch, total), float("inf"), device=xyz.device)
    center = coordinates.mean(dim=1, keepdim=True)
    farthest = ((coordinates - center) ** 2).sum(-1).argmax(-1)
    batch_index = torch.arange(batch, device=xyz.device)
    for position in range(number):
        centroids[:, position] = farthest
        centroid = coordinates[batch_index, farthest].unsqueeze(1)
        minimum = torch.minimum(minimum, ((coordinates - centroid) ** 2).sum(-1))
        farthest = minimum.argmax(-1)
    return centroids


def sample_centroids(xyz: torch.Tensor, number: int):
    indices = farthest_point_sample(xyz, number)
    return index_points(xyz, indices), indices


def ball_query(centroids: torch.Tensor, support: torch.Tensor, radius: float, number: int):
    distances, indices = knn_indices(centroids, support, number)
    invalid = distances > float(radius) ** 2
    nearest = indices[..., :1].expand_as(indices)
    return torch.where(invalid, nearest, indices)


class RegressionHead(nn.Module):
    def __init__(self, input_channels: int, hidden: Sequence[int] = (512, 256), dropout: float = 0.4):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_channels
        for width in hidden:
            layers += [nn.Linear(current, width), nn.BatchNorm1d(width), nn.LeakyReLU(0.2, inplace=True)]
            if dropout:
                layers.append(nn.Dropout(dropout))
            current = width
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class TransitionDown(nn.Module):
    def __init__(self, number: int, neighbors: int, input_channels: int, output_channels: int):
        super().__init__()
        self.number, self.neighbors = number, neighbors
        self.mlp = nn.Sequential(
            nn.Linear(input_channels + 3, output_channels, bias=False),
            nn.LayerNorm(output_channels), nn.GELU(),
            nn.Linear(output_channels, output_channels, bias=False),
            nn.LayerNorm(output_channels), nn.GELU(),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        new_xyz, _ = sample_centroids(xyz, self.number)
        _, indices = knn_indices(new_xyz, xyz, self.neighbors)
        relative = index_points(xyz, indices) - new_xyz.unsqueeze(2)
        grouped = torch.cat((relative, index_points(features, indices)), dim=-1)
        return new_xyz, self.mlp(grouped).amax(dim=2)
