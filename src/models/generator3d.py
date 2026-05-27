import torch
import torch.nn as nn


class Generator3D(nn.Module):
    """论文风格 3D 生成器：z + 条件标签 -> 64^3 三相体素概率。"""

    def __init__(
        self,
        z_channels: int = 16,
        num_classes: int = 3,
        embed_dim: int = 320,
        channels: list[int] | None = None,
    ):
        super().__init__()
        self.z_channels = z_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        c = channels or [512, 256, 128, 64]

        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.cond_proj = nn.Linear(embed_dim, 4 * 4 * 4)

        in_channels = z_channels + 1
        self.net = nn.Sequential(
            nn.ConvTranspose3d(in_channels, c[0], kernel_size=4, stride=2, padding=2),
            nn.BatchNorm3d(c[0]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(c[0], c[1], kernel_size=4, stride=2, padding=2),
            nn.BatchNorm3d(c[1]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(c[1], c[2], kernel_size=4, stride=2, padding=2),
            nn.BatchNorm3d(c[2]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(c[2], c[3], kernel_size=4, stride=2, padding=2),
            nn.BatchNorm3d(c[3]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(c[3], 3, kernel_size=4, stride=2, padding=3),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # 标签嵌入后投影成 4x4x4 条件体，与 z 在通道维拼接
        cond = self.embedding(labels)
        cond = self.cond_proj(cond).view(-1, 1, 4, 4, 4)
        x = torch.cat([z, cond], dim=1)
        logits = self.net(x)
        return torch.softmax(logits, dim=1)
