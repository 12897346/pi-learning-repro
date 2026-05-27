import torch
import torch.nn as nn


class PhysDNN(nn.Module):
    """物理先验 DNN: 输入 active TPB 长度，输出 7 点 J。"""

    def __init__(self, input_size: int = 1, hidden_dim: int = 50, output_size: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
