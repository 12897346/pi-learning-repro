import torch
import torch.nn as nn


class PhysCNN(nn.Module):
    """物理先验 CNN: 输入 3D 结构，输出 7 个过电位点的 J 预测。"""

    def __init__(self, in_channels: int = 4, out_dim: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(1 * 2 * 2 * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = feat.flatten(start_dim=1)
        return self.head(feat)
