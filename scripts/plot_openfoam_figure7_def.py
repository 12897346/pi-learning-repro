from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.physics.tpb_logic import active_union_mask_from_label_volume


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="基于 OpenFOAM 导出场重建论文 Figure 7 d/e/f")
    p.add_argument("--openfoam-dir", required=True, help="包含 low/intermediate/global 三组 .npy 的目录")
    p.add_argument("--out-path", default="outputs/paper_figures/figure7_def_openfoam.png")
    return p.parse_args()


def _load_group(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    # phase: 0/128/255，phi_ion: 与 phase 同尺寸
    phase = np.load(root / f"phase_{name}.npy").astype(np.float32)
    phi = np.load(root / f"phi_ion_{name}.npy").astype(np.float32)
    if phase.shape != phi.shape:
        raise ValueError(f"{name} 组 phase 与 phi 维度不一致: {phase.shape} vs {phi.shape}")
    if phase.ndim != 3:
        raise ValueError(f"{name} 组输入必须是 3D: {phase.shape}")
    return phase, phi


def _isolated_density_map(phase: np.ndarray) -> np.ndarray:
    label = np.zeros_like(phase, dtype=np.int8)
    label[phase == 0] = 0
    label[phase == 128] = 1
    label[phase == 255] = 2
    active = active_union_mask_from_label_volume(label, pore_value=0, ion_value=2, ele_value=1)
    iso = (~active).astype(np.float32)
    # 按论文 f 图思路做厚度方向累计
    return np.sum(iso, axis=0)


def main() -> int:
    args = parse_args()
    root = Path(args.openfoam_dir)
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    groups = [("low", "Low"), ("intermediate", "Intermediate"), ("global", "Global optimum")]
    data: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    for key, label in groups:
        phase, phi = _load_group(root, key)
        iso = _isolated_density_map(phase)
        data.append((phase, phi, iso, label))

    fig = plt.figure(figsize=(13, 7))
    for i, (phase, phi, iso, label) in enumerate(data, start=1):
        zmid = phase.shape[0] // 2
        ax1 = fig.add_subplot(2, 3, i)
        ax1.imshow(phi[zmid], cmap="plasma")
        ax1.set_title(f"(d/e) {label}  $\\phi_{{ion}}$")
        ax1.axis("off")

        ax2 = fig.add_subplot(2, 3, i + 3)
        ax2.imshow(iso, cmap="magma")
        ax2.set_title(f"(f) {label} isolated density")
        ax2.axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"已输出: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

