from __future__ import annotations

import numpy as np
from scipy import ndimage


def _conn18_structure() -> np.ndarray:
    """3x3x3 的 18 邻域连通结构元（含面邻+棱邻，不含角邻）。"""
    s = np.zeros((3, 3, 3), dtype=bool)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                di, dj, dk = abs(i - 1), abs(j - 1), abs(k - 1)
                if di == dj == dk == 0:
                    s[i, j, k] = True
                elif (di + dj + dk) in (1, 2):
                    s[i, j, k] = True
    return s


def _active_mask_periodic_xyz(binary_phase: np.ndarray, min_fraction: float = 0.01) -> np.ndarray:
    """
    近似 MATLAB 周期连通域判定：
    - 先在 xyz 三方向进行 3x3x3 平铺
    - 做 18 邻域连通域
    - 保留体素数 > 1% 原始体素总量的连通域
    - 回裁中心块作为 active mask
    """
    nx, ny, nz = binary_phase.shape
    tiled = np.tile(binary_phase.astype(bool), (3, 3, 3))
    labeled, n_comp = ndimage.label(tiled, structure=_conn18_structure())
    if n_comp == 0:
        return np.zeros_like(binary_phase, dtype=bool)

    min_size = int(max(1, round(min_fraction * nx * ny * nz)))
    comp_sizes = np.bincount(labeled.ravel())
    keep = np.zeros_like(comp_sizes, dtype=bool)
    keep[np.where(comp_sizes > min_size)] = True
    keep[0] = False
    active_tiled = keep[labeled]
    return active_tiled[nx : 2 * nx, ny : 2 * ny, nz : 2 * nz]


def active_tpb_density_from_label_volume(
    label_vol: np.ndarray,
    pore_value: int = 0,
    ion_value: int = 1,
    ele_value: int = 2,
    min_connected_fraction: float = 0.01,
) -> float:
    """
    基于 B2_IdentifyTPB 思路的 TPB 计数（近似）：
    1) 三相分别做周期连通域过滤，仅保留 active 区域；
    2) 按三种边方向遍历 2x2 邻域，若同时含三相则计为 TPB edge；
    3) 输出归一化密度 count/(Nx*Ny*Nz)。
    """
    f = np.asarray(label_vol)
    if f.ndim != 3:
        raise ValueError(f"label_vol 必须是 3D，当前 {f.shape}")

    pore_active = _active_mask_periodic_xyz(f == pore_value, min_fraction=min_connected_fraction)
    ion_active = _active_mask_periodic_xyz(f == ion_value, min_fraction=min_connected_fraction)
    ele_active = _active_mask_periodic_xyz(f == ele_value, min_fraction=min_connected_fraction)

    # 仅保留 active 相；非 active 位置记为 -1
    active_label = np.full_like(f, fill_value=-1, dtype=np.int8)
    active_label[pore_active] = 0
    active_label[ion_active] = 1
    active_label[ele_active] = 2

    # 周期索引通过 np.roll 实现
    c000 = active_label
    c0m0 = np.roll(active_label, shift=1, axis=1)
    c00m = np.roll(active_label, shift=1, axis=2)
    c0mm = np.roll(c0m0, shift=1, axis=2)
    # x 方向边：yz 平面的 2x2
    tpb_x = _contains_three_phases(c000, c0m0, c00m, c0mm)

    cm00 = np.roll(active_label, shift=1, axis=0)
    cm0m = np.roll(cm00, shift=1, axis=2)
    # y 方向边：xz 平面的 2x2
    tpb_y = _contains_three_phases(c000, cm00, c00m, cm0m)

    cmm0 = np.roll(cm00, shift=1, axis=1)
    # z 方向边：xy 平面的 2x2
    tpb_z = _contains_three_phases(c000, cm00, c0m0, cmm0)

    count_tpb = int(np.sum(tpb_x) + np.sum(tpb_y) + np.sum(tpb_z))
    denom = float(f.shape[0] * f.shape[1] * f.shape[2])
    return float(count_tpb / max(denom, 1.0))


def active_union_mask_from_label_volume(
    label_vol: np.ndarray,
    pore_value: int = 0,
    ion_value: int = 1,
    ele_value: int = 2,
    min_connected_fraction: float = 0.01,
) -> np.ndarray:
    """
    输出三相 active 连通域并集掩膜（用于 phys-CNN 的 connectivity 特征通道）。
    """
    f = np.asarray(label_vol)
    if f.ndim != 3:
        raise ValueError(f"label_vol 必须是 3D，当前 {f.shape}")
    pore_active = _active_mask_periodic_xyz(f == pore_value, min_fraction=min_connected_fraction)
    ion_active = _active_mask_periodic_xyz(f == ion_value, min_fraction=min_connected_fraction)
    ele_active = _active_mask_periodic_xyz(f == ele_value, min_fraction=min_connected_fraction)
    return pore_active | ion_active | ele_active


def fast_union_mask_from_label_volume(
    label_vol: np.ndarray,
    pore_value: int = 0,
    ion_value: int = 1,
    ele_value: int = 2,
) -> np.ndarray:
    """
    O(V) 快速连通性代理：6 邻域内存在不同相标签的体素（界面区），不做周期连通域过滤。
    用于 PSO / 训练加速；与 strict 的 active_union 数值不同但量级相近。
    """
    f = np.asarray(label_vol, dtype=np.int8)
    if f.ndim != 3:
        raise ValueError(f"label_vol 必须是 3D，当前 {f.shape}")
    diff = np.zeros_like(f, dtype=bool)
    for ax in range(3):
        diff |= np.roll(f, 1, axis=ax) != f
    valid = (f == pore_value) | (f == ion_value) | (f == ele_value)
    return diff & valid


def fast_union_mask_batch(
    labels: np.ndarray,
    pore_value: int = 0,
    ion_value: int = 1,
    ele_value: int = 2,
) -> np.ndarray:
    """labels: [B,D,H,W] int8；返回 [B,D,H,W] float32 连通性掩膜。"""
    f = np.asarray(labels, dtype=np.int8)
    if f.ndim != 4:
        raise ValueError(f"labels 期望 [B,D,H,W]，当前 {f.shape}")
    diff = np.zeros(f.shape, dtype=bool)
    for ax in range(1, 4):
        diff |= np.roll(f, 1, axis=ax) != f
    valid = (f == pore_value) | (f == ion_value) | (f == ele_value)
    return (diff & valid).astype(np.float32)


def _contains_three_phases(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """四个格点是否同时包含三相(0,1,2)。"""
    has0 = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    has1 = (a == 1) | (b == 1) | (c == 1) | (d == 1)
    has2 = (a == 2) | (b == 2) | (c == 2) | (d == 2)
    return has0 & has1 & has2
