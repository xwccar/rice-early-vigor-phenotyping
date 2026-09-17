from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d(nn.Module):
    def __init__(self, channel: int = 3):
        super().__init__()
        self.conv1, self.conv2, self.conv3 = nn.Conv1d(channel, 64, 1), nn.Conv1d(64, 128, 1), nn.Conv1d(128, 1024, 1)
        self.fc1, self.fc2, self.fc3 = nn.Linear(1024, 512), nn.Linear(512, 256), nn.Linear(256, 9)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(64), nn.BatchNorm1d(128), nn.BatchNorm1d(1024)
        self.bn4, self.bn5 = nn.BatchNorm1d(512), nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)).amax(dim=2)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        identity = torch.eye(3, device=x.device, dtype=x.dtype).reshape(1, 9).repeat(batch, 1)
        return (self.fc3(x) + identity).reshape(-1, 3, 3)


class STNkd(nn.Module):
    def __init__(self, k: int = 64):
        super().__init__()
        self.k = k
        self.conv1, self.conv2, self.conv3 = nn.Conv1d(k, 64, 1), nn.Conv1d(64, 128, 1), nn.Conv1d(128, 1024, 1)
        self.fc1, self.fc2, self.fc3 = nn.Linear(1024, 512), nn.Linear(512, 256), nn.Linear(256, k * k)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(64), nn.BatchNorm1d(128), nn.BatchNorm1d(1024)
        self.bn4, self.bn5 = nn.BatchNorm1d(512), nn.BatchNorm1d(256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)).amax(dim=2)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).reshape(1, -1).repeat(batch, 1)
        return (self.fc3(x) + identity).reshape(-1, self.k, self.k)


class PointNetEncoder(nn.Module):
    def __init__(self, feature_transform: bool = True):
        super().__init__()
        self.feature_transform = feature_transform
        self.input_transform = STN3d(3)
        self.conv1, self.conv2, self.conv3 = nn.Conv1d(3, 64, 1), nn.Conv1d(64, 128, 1), nn.Conv1d(128, 1024, 1)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(64), nn.BatchNorm1d(128), nn.BatchNorm1d(1024)
        self.feature_stn = STNkd(64) if feature_transform else None

    def forward(self, points: torch.Tensor):
        transform = self.input_transform(points)
        x = torch.bmm(points.transpose(1, 2), transform).transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        feature_transform = None
        if self.feature_stn is not None:
            feature_transform = self.feature_stn(x)
            x = torch.bmm(x.transpose(1, 2), feature_transform).transpose(1, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        return self.bn3(self.conv3(x)).amax(dim=2), transform, feature_transform


class PointNetRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = PointNetEncoder(feature_transform=True)
        self.fc1, self.fc2, self.fc3 = nn.Linear(1024, 512), nn.Linear(512, 256), nn.Linear(256, 1)
        self.bn1, self.bn2 = nn.BatchNorm1d(512), nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.4)

    def _regression_head(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc3(x)

    def forward_with_feature_transform(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, _, feature_transform = self.encoder(points)
        return self._regression_head(x), feature_transform

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.forward_with_feature_transform(points)[0]


def feature_transform_regularizer(transform: torch.Tensor) -> torch.Tensor:
    dimension = transform.shape[1]
    identity = torch.eye(dimension, device=transform.device, dtype=transform.dtype).unsqueeze(0)
    return torch.linalg.matrix_norm(torch.bmm(transform, transform.transpose(1, 2)) - identity, dim=(1, 2)).mean()
