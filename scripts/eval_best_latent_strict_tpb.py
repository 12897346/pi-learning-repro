"""对 PSO 最优潜向量生成体素后，离线计算一次 strict TPB（训练/搜索阶段用 fast）。"""
from __future__ import annotations

import argparse
import json
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
from src.utils.seed import set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="最优 latent → 体素 → strict active TPB（仅评估，不参与训练）")
    p.add_argument("--config", default="configs/paper_params.yaml")
    p.add_argument("--gen-ckpt", required=True)
    p.add_argument("--best-latent-npy", required=True)
    p.add_argument("--out-json", default="")
    p.add_argument("--device", default="cpu")
    p.add_argument("--force-tiny", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _phase_vol_from_onehot(x3: torch.Tensor) -> np.ndarray:
    idx = torch.argmax(x3, dim=1)
    out = torch.zeros_like(idx, dtype=torch.float32)
    out[idx == 1] = 128.0
    out[idx == 2] = 255.0
    return out[0].detach().cpu().numpy()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    latent_path = Path(args.best_latent_npy)
    z_flat = np.load(latent_path).astype(np.float32).reshape(1, -1)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gan = cfg["gan"]
    use_tiny = bool(getattr(args, "force_tiny", False)) or bool(gan.get("debug_tiny", True))
    g_channels = gan["tiny_channels"]["generator"] if use_tiny else gan["paper_channels"]["generator"]
    latent_dim = int(gan["latent_channels"] * 4 * 4 * 4)
    if z_flat.shape[1] != latent_dim:
        raise ValueError(f"latent 维度 {z_flat.shape[1]} != 期望 {latent_dim}")

    gen = Generator3D(
        z_channels=gan["latent_channels"],
        num_classes=gan["classes_inverse"],
        embed_dim=gan["embedding_size"],
        channels=g_channels,
    ).to(device)
    gen.load_state_dict(torch.load(args.gen_ckpt, map_location=device))
    gen.eval()

    z = torch.from_numpy(z_flat).to(device).view(-1, gan["latent_channels"], 4, 4, 4)
    labels = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        x3 = gen(z, labels)
    vol = _phase_vol_from_onehot(x3)
    label = np.zeros_like(vol, dtype=np.int8)
    label[vol == 0] = 0
    label[vol == 128] = 1
    label[vol == 255] = 2
    strict_tpb = float(
        active_tpb_density_from_label_volume(label, pore_value=0, ion_value=2, ele_value=1)
    )

    out = {
        "best_latent_npy": str(latent_path.resolve()),
        "gen_ckpt": str(Path(args.gen_ckpt).resolve()),
        "strict_active_tpb_density": strict_tpb,
        "connectivity_eval_mode": "strict",
    }
    out_json = Path(args.out_json) if args.out_json else latent_path.parent / "strict_tpb_eval.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[OK] strict TPB = {strict_tpb:.6f} -> {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
