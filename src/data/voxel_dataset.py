from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.physics.tpb_logic import active_union_mask_from_label_volume, fast_union_mask_from_label_volume


@dataclass
class DataBundle:
    volumes: np.ndarray
    j_labels: np.ndarray
    tpb: np.ndarray


def _load_required(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"缺少数据文件: {path}")
    return np.load(path)


def load_data_bundle(data_dir: str | Path) -> DataBundle:
    """
    约定数据文件:
    - volumes.npy: [N, D, H, W] 或 [N, C, D, H, W]
    - labels_j.npy: [N, 7]
    - tpb.npy: [N] 或 [N, 1]
    """
    root = Path(data_dir)
    volumes = _load_required(root / "volumes.npy")
    j_labels = _load_required(root / "labels_j.npy")
    tpb = _load_required(root / "tpb.npy")

    if volumes.ndim == 4:
        volumes = volumes[:, None, ...]
    if volumes.ndim != 5:
        raise ValueError(f"volumes 维度应为 4 或 5，当前为 {volumes.ndim}")

    if j_labels.ndim != 2 or j_labels.shape[1] != 7:
        raise ValueError(f"labels_j 期望形状 [N,7]，当前为 {j_labels.shape}")

    if tpb.ndim == 1:
        tpb = tpb[:, None]
    if tpb.ndim != 2 or tpb.shape[1] != 1:
        raise ValueError(f"tpb 期望形状 [N,1]，当前为 {tpb.shape}")

    n = volumes.shape[0]
    if j_labels.shape[0] != n or tpb.shape[0] != n:
        raise ValueError("volumes/labels_j/tpb 样本数不一致")

    return DataBundle(volumes=volumes, j_labels=j_labels, tpb=tpb)


def train_val_split(
    bundle: DataBundle,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[DataBundle, DataBundle]:
    n = bundle.volumes.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    val_n = max(1, int(n * val_ratio))
    val_idx = idx[:val_n]
    train_idx = idx[val_n:]

    def _pick(x: np.ndarray, i: np.ndarray) -> np.ndarray:
        return x[i]

    train_bundle = DataBundle(
        volumes=_pick(bundle.volumes, train_idx),
        j_labels=_pick(bundle.j_labels, train_idx),
        tpb=_pick(bundle.tpb, train_idx),
    )
    val_bundle = DataBundle(
        volumes=_pick(bundle.volumes, val_idx),
        j_labels=_pick(bundle.j_labels, val_idx),
        tpb=_pick(bundle.tpb, val_idx),
    )
    return train_bundle, val_bundle


class PhysCNNDataset(Dataset):
    """phys-CNN 数据集：输入 3D 体素，输出 7 点 J。"""

    def __init__(self, bundle: DataBundle, connectivity_mode: str = "fast"):
        self.volumes = bundle.volumes.astype(np.float32)
        self.j_labels = bundle.j_labels.astype(np.float32)
        self.connectivity_mode = str(connectivity_mode or "fast").lower()

    def __len__(self) -> int:
        return self.volumes.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # 正文写体素归一化到 (−1,1) 为 CNN 常见预处理；本实现按 Supporting Table S7 使用
        # one-hot 三相 + 连通性共 4 通道。
        v = self.volumes[idx]
        if v.ndim != 4 or v.shape[0] != 1:
            raise ValueError(f"phys-CNN 输入期望 [1,D,H,W]，当前 {v.shape}")
        vol = v[0]
        pore = (vol == 0).astype(np.float32)
        ni = (vol == 128).astype(np.float32)
        ysz = (vol == 255).astype(np.float32)
        label = np.zeros_like(vol, dtype=np.int8)
        label[vol == 0] = 0
        label[vol == 128] = 1
        label[vol == 255] = 2
        if self.connectivity_mode == "strict":
            conn = active_union_mask_from_label_volume(
                label, pore_value=0, ion_value=2, ele_value=1, min_connected_fraction=0.01
            ).astype(np.float32)
        else:
            conn = fast_union_mask_from_label_volume(
                label, pore_value=0, ion_value=2, ele_value=1
            ).astype(np.float32)
        x = torch.from_numpy(np.stack([pore, ni, ysz, conn], axis=0))
        y = torch.from_numpy(self.j_labels[idx])
        return x, y


class PhysDNNDataset(Dataset):
    """phys-DNN 数据集：输入 active TPB 标量，输出 7 点 J。"""

    def __init__(self, bundle: DataBundle):
        self.tpb = bundle.tpb.astype(np.float32)
        self.j_labels = bundle.j_labels.astype(np.float32)

    def __len__(self) -> int:
        return self.tpb.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.tpb[idx])
        y = torch.from_numpy(self.j_labels[idx])
        return x, y
