#!/usr/bin/env python3
"""构建可训练替代数据；需要 Python 3.8+。若 `python` 报语法错，请用 `python3` 或先 `conda activate zxh`。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

if sys.version_info < (3, 8):
    raise SystemExit(
        "本脚本需要 Python 3.8+。登录节点上 `python` 可能指向 Python 2。\n"
        "请执行其一：\n"
        "  bash scripts/build_fallback_training_data.sh --out-dir data/processed_fallback\n"
        "  python3 scripts/build_fallback_training_data.py --out-dir data/processed_fallback\n"
        "  conda activate zxh && python scripts/build_fallback_training_data.py ...\n"
        f"当前: {sys.executable}\n{sys.version}\n"
        "若仍出现「需要 Python 3.9+」字样，说明集群上该脚本未更新，请 git pull 或从本机同步 scripts/build_fallback_training_data.py。"
    )

import numpy as np
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.tpb_logic import active_tpb_density_from_label_volume
from src.utils.output_manifest import write_training_bundle_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建可训练的替代数据（非论文原始数据）")
    parser.add_argument("--out-dir", default="data/processed_fallback", help="输出目录")
    parser.add_argument("--num-samples", type=int, default=128, help="样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def make_three_phase_volume(rng: np.random.Generator, size: int = 64) -> np.ndarray:
    # 构造更像真实团簇形貌的三相随机场（避免纯白噪声导致结构过碎）
    g1 = gaussian_filter(rng.standard_normal((size, size, size)), sigma=float(rng.uniform(1.0, 2.4)))
    g2 = gaussian_filter(rng.standard_normal((size, size, size)), sigma=float(rng.uniform(1.0, 2.8)))
    field = 0.65 * g1 + 0.35 * g2 + 0.15 * gaussian_filter(g1 * g2, sigma=1.0)
    field = (field - field.mean()) / (field.std() + 1e-6)

    # 随机化相分数，扩大样本分布跨度
    f_pore = float(rng.uniform(0.18, 0.42))
    f_ni = float(rng.uniform(0.25, 0.48))
    f_ysz = max(0.08, 1.0 - f_pore - f_ni)
    s = f_pore + f_ni + f_ysz
    f_pore, f_ni, f_ysz = f_pore / s, f_ni / s, f_ysz / s

    t1 = np.quantile(field, f_pore)
    t2 = np.quantile(field, f_pore + f_ni)
    vol = np.zeros((size, size, size), dtype=np.float32)
    vol[(field > t1) & (field <= t2)] = 128.0
    vol[field > t2] = 255.0
    return vol


def approx_tpb_and_j(vol: np.ndarray, rng: np.random.Generator) -> Tuple[float, np.ndarray]:
    # TPB 采用与训练侧统一的严格口径
    label = np.zeros_like(vol, dtype=np.int8)
    label[vol == 0] = 0
    label[vol == 128] = 1
    label[vol == 255] = 2
    tpb = float(active_tpb_density_from_label_volume(label, pore_value=0, ion_value=2, ele_value=1))

    # 生成论文量级（mA cm^-2）的 7 点 J-η 曲线，并确保单调
    eta_mv = np.array([20, 40, 60, 80, 100, 120, 140], dtype=np.float32)
    eta_norm = eta_mv / 140.0
    f_pore = float(np.mean(vol == 0))
    f_ni = float(np.mean(vol == 128))
    f_ysz = float(np.mean(vol == 255))

    # 平衡度（接近常见阳极比例时性能更高）
    balance = np.exp(-((f_pore - 0.30) ** 2 + (f_ni - 0.35) ** 2 + (f_ysz - 0.35) ** 2) / 0.015)
    tpb_term = np.clip(tpb / 1.8, 0.0, 1.2)
    gain = 70.0 + 170.0 * balance + 120.0 * tpb_term + float(rng.normal(0, 12.0))
    curve = eta_norm ** float(rng.uniform(1.18, 1.45))
    j = gain * curve + 8.0 * eta_norm + rng.normal(0, 4.0, size=eta_norm.shape)
    j = np.maximum.accumulate(np.clip(j, 0.0, None)).astype(np.float32)
    return tpb, j


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    volumes = np.zeros((args.num_samples, 1, 64, 64, 64), dtype=np.float32)
    tpb = np.zeros((args.num_samples, 1), dtype=np.float32)
    labels_j = np.zeros((args.num_samples, 7), dtype=np.float32)

    for i in range(args.num_samples):
        vol = make_three_phase_volume(rng, size=64)
        tpb_i, j_i = approx_tpb_and_j(vol, rng)
        volumes[i, 0] = vol
        tpb[i, 0] = tpb_i
        labels_j[i] = j_i

    np.save(out_dir / "volumes.npy", volumes)
    np.save(out_dir / "tpb.npy", tpb)
    np.save(out_dir / "labels_j.npy", labels_j)

    write_training_bundle_manifest(
        out_dir,
        volumes_shape=volumes.shape,
        labels_j_shape=labels_j.shape,
        tpb_shape=tpb.shape,
        data_provenance="fallback：高斯随机场体素 + proxy_formula 合成 J 曲线；非 OpenFOAM 真值",
        extra={"script": "scripts/build_fallback_training_data.py", "seed": args.seed, "num_samples": args.num_samples},
    )

    print("=== 替代训练数据构建完成 ===")
    print(f"输出目录: {out_dir}")
    print(f"volumes: {volumes.shape}")
    print(f"tpb: {tpb.shape}")
    print(f"labels_j: {labels_j.shape}")
    print("注意: 该数据用于流程训练与调试，不代表论文真实标注。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
