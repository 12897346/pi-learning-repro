#!/usr/bin/env python3
"""从已训练 Generator3D 采样微结构，写出与训练 bundle 相同格式的 npy（供 phys-CNN 混合训练等）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.generator3d import Generator3D  # noqa: E402
from src.physics.tpb_logic import active_tpb_density_from_label_volume  # noqa: E402
from src.utils.output_manifest import write_training_bundle_manifest  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GAN 生成器批量采样 → volumes.npy / tpb.npy / labels_j.npy")
    p.add_argument("--config", type=Path, default=Path("configs/paper_params.yaml"))
    p.add_argument("--gen-ckpt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=3000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--labels-j-npy",
        type=Path,
        default=None,
        help="若提供：须为 [num_samples,7] 的 OpenFOAM/论文对齐标签；否则用 --label-mode",
    )
    p.add_argument(
        "--label-mode",
        choices=["proxy", "phys_dnn", "zeros"],
        default="proxy",
        help="无外部 labels 时：proxy 公式；phys_dnn 用已训 phys_dnn 由 tpb 推断；zeros 占位",
    )
    p.add_argument(
        "--phys-dnn-ckpt",
        type=Path,
        default=None,
        help="label-mode=phys_dnn 时必填",
    )
    return p.parse_args()


def _onehot_to_phase_volume(x3: torch.Tensor) -> np.ndarray:
    """[1,3,D,H,W] softmax → [D,H,W] 0/128/255 float32"""
    hard = torch.argmax(x3[0], dim=0).detach().cpu().numpy().astype(np.int8)
    out = np.zeros_like(hard, dtype=np.float32)
    out[hard == 1] = 128.0
    out[hard == 2] = 255.0
    return out


def _tpb_from_vol(vol: np.ndarray) -> float:
    lab = np.zeros_like(vol, dtype=np.int8)
    lab[vol == 0] = 0
    lab[vol == 128] = 1
    lab[vol == 255] = 2
    return float(
        active_tpb_density_from_label_volume(
            lab, pore_value=0, ion_value=2, ele_value=1, min_connected_fraction=0.01
        )
    )


def _labels_proxy(tpb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    eta = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14], dtype=np.float32)
    base = 0.03 + 0.8 * tpb[:, None]
    y = base * (1.0 + 1.8 * eta[None, :]) + rng.normal(0, 0.003, size=(tpb.shape[0], 7))
    return np.clip(y, 0.0, None).astype(np.float32)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gan = cfg["gan"]
    use_tiny = bool(gan.get("debug_tiny", True))
    g_channels = gan["tiny_channels"]["generator"] if use_tiny else gan["paper_channels"]["generator"]

    gen = Generator3D(
        z_channels=gan["latent_channels"],
        num_classes=gan["classes_inverse"],
        embed_dim=gan["embedding_size"],
        channels=g_channels,
    ).to(device)
    gen.load_state_dict(torch.load(args.gen_ckpt, map_location=device))
    gen.eval()

    n = int(args.num_samples)
    vols: list[np.ndarray] = []
    tpbs: list[float] = []

    with torch.no_grad():
        for _ in range(n):
            z = torch.randn(1, gan["latent_channels"], 4, 4, 4, device=device)
            lab = torch.zeros(1, dtype=torch.long, device=device)
            x3 = gen(z, lab)
            v = _onehot_to_phase_volume(x3)
            vols.append(v)
            tpbs.append(_tpb_from_vol(v))

    volumes = np.stack(vols, axis=0)[:, None, ...].astype(np.float32)
    tpb = np.asarray(tpbs, dtype=np.float32)[:, None]

    rng = np.random.default_rng(args.seed)
    if args.labels_j_npy is not None:
        lj = np.load(Path(args.labels_j_npy)).astype(np.float32)
        if lj.shape != (n, 7):
            raise ValueError(f"--labels-j-npy 形状须为 ({n},7)，当前 {lj.shape}")
        labels_j = lj
        note = f"外部标签文件: {args.labels_j_npy}"
    elif args.label_mode == "phys_dnn":
        if not args.phys_dnn_ckpt or not Path(args.phys_dnn_ckpt).exists():
            raise FileNotFoundError("label-mode=phys_dnn 时请提供有效 --phys-dnn-ckpt")
        from src.models.phys_dnn import PhysDNN  # noqa: E402

        phys = PhysDNN(input_size=1, hidden_dim=50, output_size=7).to(device)
        phys.load_state_dict(torch.load(args.phys_dnn_ckpt, map_location=device))
        phys.eval()
        with torch.no_grad():
            pred = phys(torch.from_numpy(tpb).to(device)).cpu().numpy()
        labels_j = pred.astype(np.float32)
        note = f"phys_dnn 推断: {args.phys_dnn_ckpt}"
    elif args.label_mode == "zeros":
        labels_j = np.zeros((n, 7), dtype=np.float32)
        note = "zeros 占位"
    else:
        labels_j = _labels_proxy(tpb, rng)
        note = "proxy_formula（非 OpenFOAM）"

    np.save(out / "volumes.npy", volumes)
    np.save(out / "tpb.npy", tpb)
    np.save(out / "labels_j.npy", labels_j)

    write_training_bundle_manifest(
        out,
        volumes_shape=volumes.shape,
        labels_j_shape=labels_j.shape,
        tpb_shape=tpb.shape,
        data_provenance=f"export_gan_microstructures: ckpt={args.gen_ckpt}; {note}",
        extra={
            "script": "scripts/export_gan_microstructures.py",
            "gen_ckpt": str(Path(args.gen_ckpt).resolve()),
            "num_samples": n,
            "label_mode": args.label_mode,
        },
    )
    print("=== GAN 采样导出完成 ===", f"out={out}", f"volumes={volumes.shape}", f"labels: {note}", sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
