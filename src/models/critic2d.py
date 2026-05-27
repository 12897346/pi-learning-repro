import torch
import torch.nn as nn


class Critic2D(nn.Module):
    """论文思路的 2D critic：从三个方向逐片评分后聚合。"""

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
        self.cond_proj = nn.Linear(embed_dim, 64 * 64)

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels + 1, c[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c[0], c[1], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c[1], c[2], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c[2], c[3], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c[3], 1, kernel_size=4, stride=2, padding=0),
        )

    def forward(self, volume: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        volume: [B, C, D, H, W]
        输出: [B] critic score
        """
        b, _, d, h, w = volume.shape
        if d != h or h != w:
            raise ValueError(f"为与论文设定一致，当前要求立方体体素，收到 {volume.shape}")

        cond_base = self.embedding(labels)
        cond_base = self.cond_proj(cond_base).view(b, 1, h, w)

        def _score_along_axis(v: torch.Tensor, axis: int) -> torch.Tensor:
            # axis=2/3/4 对应 D/H/W 方向逐片
            if axis == 2:
                s = v.permute(0, 2, 1, 3, 4).reshape(b * d, -1, h, w)
                n = d
            elif axis == 3:
                s = v.permute(0, 3, 1, 2, 4).reshape(b * h, -1, d, w)
                n = h
            else:
                s = v.permute(0, 4, 1, 2, 3).reshape(b * w, -1, d, h)
                n = w

            cond = cond_base.unsqueeze(1).repeat(1, n, 1, 1, 1).reshape(b * n, 1, h, w)
            x = torch.cat([s, cond], dim=1)
            score = self.backbone(x).reshape(b, n, -1)
            return score.mean(dim=(1, 2))

        score_d = _score_along_axis(volume, axis=2)
        score_h = _score_along_axis(volume, axis=3)
        score_w = _score_along_axis(volume, axis=4)
        return (score_d + score_h + score_w) / 3.0
