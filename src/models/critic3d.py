import torch
import torch.nn as nn


class Critic3D(nn.Module):
    """3D 判别器：直接对 3D 体素判别，避免 2D 切片近似。"""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        embed_dim: int = 320,
        channels: list[int] | None = None,
    ):
        super().__init__()
        c = channels or [512, 256, 128, 64]
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.cond_proj = nn.Linear(embed_dim, 64 * 64 * 64)

        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels + 1, c[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(c[0], c[1], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(c[1], c[2], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(c[2], c[3], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(c[3], 1, kernel_size=4, stride=2, padding=0),
        )

    def forward(self, volume: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        volume: [B, C, D, H, W]
        输出: [B] critic score
        """
        b, _, d, h, w = volume.shape
        cond = self.embedding(labels)
        cond = self.cond_proj(cond).view(b, 1, d, h, w)
        x = torch.cat([volume, cond], dim=1)
        score = self.backbone(x)
        return score.view(b, -1).mean(dim=1)
