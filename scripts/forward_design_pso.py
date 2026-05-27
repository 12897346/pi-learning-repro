from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.generator3d import Generator3D  # noqa: E402
from src.models.phys_cnn import PhysCNN  # noqa: E402
from src.models.phys_dnn import PhysDNN  # noqa: E402
from src.physics.tpb_logic import (  # noqa: E402
    active_tpb_density_from_label_volume,
    active_union_mask_from_label_volume,
    fast_union_mask_batch,
    fast_union_mask_from_label_volume,
)
from src.utils.seed import set_seed  # noqa: E402
from src.utils.output_manifest import (  # noqa: E402
    PAPER_ETA_MV_MILLIVOLT,
    output_spec_block,
    write_json,
)


def resolve_pso_plateau_patience(
    patience: int,
    iters: int,
    *,
    reference_iters: int = 300,
    paper_patience: int = 200,
) -> int:
    """将平台耐心限制在 [1, iters]。

    仅当 patience 达到论文量级（>= paper_patience）时按 reference_iters 比例缩放；
    用户显式设较小值时只做不超过 iters 的上限。
    """
    p = max(0, int(patience))
    n = max(1, int(iters))
    if p <= 0:
        return 0
    if p >= int(paper_patience):
        ref = max(1, int(reference_iters))
        scaled = max(1, int(round(p * n / ref)))
        return min(p, n, scaled)
    return min(p, n)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="论文 forward design：PSO 搜索潜空间全局最优 J")
    p.add_argument("--config", default="configs/paper_params.yaml")
    p.add_argument("--gen-ckpt", default="outputs/gan_fallback/generator_fallback.pth")
    p.add_argument("--phys-dnn-ckpt", default="outputs/phys_models/phys_dnn.pth")
    p.add_argument("--phys-cnn-ckpt", default="outputs/phys_models/phys_cnn.pth")
    p.add_argument("--surrogate", choices=["phys_dnn", "phys_cnn"], default="phys_cnn")
    p.add_argument("--out-dir", default="outputs/forward_design")
    p.add_argument("--particles", type=int, default=1000)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument(
        "--prior-samples",
        type=int,
        default=0,
        help="先验 J 采样数；0=沿用 min(10000, max(2000, particles*5)) 规则",
    )
    p.add_argument("--c1", type=float, default=2.0)
    p.add_argument("--c2", type=float, default=2.0)
    p.add_argument("--w", type=float, default=0.8)
    p.add_argument("--j-index", type=int, default=5, help="J-eta 第几个点作为目标（默认 0.12V）")
    p.add_argument(
        "--eval-microbatch",
        type=int,
        default=32,
        help="PSO 中 gen+surrogate 前向的微批大小；粒子数或 prior 很大时需减小以防 CUDA OOM（如 8/16）",
    )
    p.add_argument(
        "--plateau-patience",
        type=int,
        default=-1,
        help="gbest_J 连续若干迭代「提升 < plateau-tol」则早停；0=关闭；-1=读 configs/paper_params.yaml 的 pso.plateau_patience",
    )
    p.add_argument(
        "--plateau-tol",
        type=float,
        default=-1.0,
        help="判定 gbest_J 是否「有实质提升」的最小增量（mA cm^-2）；默认读 YAML（1e-3），非要求完全不变",
    )
    p.add_argument(
        "--connectivity-mode",
        choices=["fast", "strict"],
        default="fast",
        help="Phys-CNN 第 4 通道：fast=界面 O(V) 批处理；strict=周期连通域（慢）",
    )
    p.add_argument(
        "--force-tiny",
        action="store_true",
        help="强制 tiny_channels 加载 Generator（须与 train_gan --force-tiny 一致）",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def batch_phys_cnn_features(
    vol_0128_255: np.ndarray,
    connectivity_mode: str,
) -> np.ndarray:
    """vol: [B,D,H,W] float；返回 [B,4,D,H,W] float32。"""
    v = np.asarray(vol_0128_255, dtype=np.float32)
    pore = (v == 0).astype(np.float32)
    ni = (v == 128).astype(np.float32)
    ysz = (v == 255).astype(np.float32)
    label = np.zeros(v.shape, dtype=np.int8)
    label[v == 0] = 0
    label[v == 128] = 1
    label[v == 255] = 2
    mode = str(connectivity_mode or "fast").lower()
    if mode == "strict":
        conn = np.zeros(v.shape, dtype=np.float32)
        for i in range(v.shape[0]):
            conn[i] = active_union_mask_from_label_volume(
                label[i], pore_value=0, ion_value=2, ele_value=1
            ).astype(np.float32)
    else:
        conn = fast_union_mask_batch(label, pore_value=0, ion_value=2, ele_value=1)
    return np.stack([pore, ni, ysz, conn], axis=1)


def onehot_to_phase_value_torch(x3: torch.Tensor) -> torch.Tensor:
    idx = torch.argmax(x3, dim=1)
    out = torch.zeros_like(idx, dtype=torch.float32)
    out[idx == 1] = 128.0
    out[idx == 2] = 255.0
    return out


def strict_tpb_from_phase_vol(vol_0128_255: np.ndarray) -> float:
    label = np.zeros_like(vol_0128_255, dtype=np.int8)
    label[vol_0128_255 == 0] = 0
    label[vol_0128_255 == 128] = 1
    label[vol_0128_255 == 255] = 2
    return active_tpb_density_from_label_volume(label, pore_value=0, ion_value=2, ele_value=1)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gan = cfg["gan"]
    pso_cfg = cfg.get("pso", {})
    _paper_repro = cfg.get("paper_repro") or {}
    ref_iters = int(_paper_repro.get("pso_iterations", 300) or 300)
    particles = int(args.particles or pso_cfg.get("particles", 1000))
    c1 = float(args.c1 or pso_cfg.get("c1", 2.0))
    c2 = float(args.c2 or pso_cfg.get("c2", 2.0))
    w = float(args.w or pso_cfg.get("w", 0.8))
    use_tiny = bool(getattr(args, "force_tiny", False)) or bool(gan.get("debug_tiny", True))
    g_channels = gan["tiny_channels"]["generator"] if use_tiny else gan["paper_channels"]["generator"]
    latent_dim = int(gan["latent_channels"] * 4 * 4 * 4)

    gen = Generator3D(
        z_channels=gan["latent_channels"],
        num_classes=gan["classes_inverse"],
        embed_dim=gan["embedding_size"],
        channels=g_channels,
    ).to(device)
    gen.load_state_dict(torch.load(args.gen_ckpt, map_location=device))
    gen.eval()

    phys_dnn = PhysDNN(input_size=1, hidden_dim=50, output_size=7).to(device)
    if Path(args.phys_dnn_ckpt).exists():
        phys_dnn.load_state_dict(torch.load(args.phys_dnn_ckpt, map_location=device))
    phys_dnn.eval()

    phys_cnn = PhysCNN(in_channels=4, out_dim=7).to(device)
    if Path(args.phys_cnn_ckpt).exists():
        phys_cnn.load_state_dict(torch.load(args.phys_cnn_ckpt, map_location=device))
    phys_cnn.eval()

    rng = np.random.default_rng(args.seed)
    pos = rng.normal(0, 1, size=(particles, latent_dim)).astype(np.float32)
    vel = np.zeros_like(pos, dtype=np.float32)
    pbest = pos.copy()
    pbest_val = np.full((particles,), -np.inf, dtype=np.float32)
    gbest = pos[0].copy()
    gbest_val = -np.inf
    history: list[tuple[int, float, float]] = []

    micro = max(1, int(args.eval_microbatch))

    @torch.no_grad()
    def evaluate_swarm(z_flat: np.ndarray) -> np.ndarray:
        """按 micro-batch 前向，避免一次性 1000/10000 粒子撑爆显存。"""
        n = int(z_flat.shape[0])
        out_parts: list[np.ndarray] = []
        for s in range(0, n, micro):
            chunk = z_flat[s : s + micro]
            z = torch.from_numpy(chunk).to(device).view(-1, gan["latent_channels"], 4, 4, 4)
            labels = torch.zeros(z.shape[0], dtype=torch.long, device=device)
            x3 = gen(z, labels)
            if args.surrogate == "phys_cnn":
                v = onehot_to_phase_value_torch(x3).detach().cpu().numpy().astype(np.float32)
                feat = batch_phys_cnn_features(v, str(getattr(args, "connectivity_mode", "fast")))
                pred = phys_cnn(torch.from_numpy(feat).to(device))[:, args.j_index]
                out_parts.append(pred.detach().cpu().numpy().astype(np.float32))
            else:
                v = onehot_to_phase_value_torch(x3).detach().cpu().numpy().astype(np.float32)
                # phys_dnn 路径仍用标量 TPB；PSO 默认 phys_cnn，此处保留供对照
                tpb = np.asarray(
                    [strict_tpb_from_phase_vol(v[i]) for i in range(v.shape[0])],
                    dtype=np.float32,
                )
                pred = phys_dnn(torch.from_numpy(tpb[:, None]).to(device))[:, args.j_index]
                out_parts.append(pred.detach().cpu().numpy().astype(np.float32))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        return np.concatenate(out_parts, axis=0)

    # 先计算 prior 分布（论文 Figure 6a 绿色区域）
    prior_n = (
        int(args.prior_samples)
        if int(getattr(args, "prior_samples", 0) or 0) > 0
        else min(10000, max(2000, particles * 5))
    )
    prior_latent = rng.normal(0, 1, size=(prior_n, latent_dim)).astype(np.float32)
    prior_j = evaluate_swarm(prior_latent)
    np.save(out_dir / "prior_j.npy", prior_j)

    plateau_patience_raw = int(args.plateau_patience)
    if plateau_patience_raw < 0:
        plateau_patience_raw = int(pso_cfg.get("plateau_patience", 200) or 200)
    plateau_tol = float(args.plateau_tol)
    if plateau_tol < 0:
        plateau_tol = float(pso_cfg.get("plateau_tol", 1.0e-3) or 1.0e-3)
    plateau_patience = resolve_pso_plateau_patience(
        plateau_patience_raw,
        int(args.iters),
        reference_iters=ref_iters,
        paper_patience=int(pso_cfg.get("plateau_patience", 200) or 200),
    )
    stagnant = 0
    print(
        f"开始 PSO: particles={particles}, iters={args.iters}, surrogate={args.surrogate}, "
        f"plateau_patience={plateau_patience or 'off'}"
        + (
            f" (请求={plateau_patience_raw}, tol={plateau_tol:g} mA/cm²)"
            if plateau_patience > 0
            else ""
        ),
        flush=True,
    )
    stopped_plateau = False
    for it in range(1, args.iters + 1):
        gbest_before = float(gbest_val)
        val = evaluate_swarm(pos)
        better = val > pbest_val
        pbest[better] = pos[better]
        pbest_val[better] = val[better]
        idx = int(np.argmax(val))
        if float(val[idx]) > gbest_val:
            gbest_val = float(val[idx])
            gbest = pos[idx].copy()
        if gbest_val > gbest_before + plateau_tol:
            stagnant = 0
        else:
            stagnant += 1
        r1 = rng.random(size=pos.shape, dtype=np.float32)
        r2 = rng.random(size=pos.shape, dtype=np.float32)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest[None, :] - pos)
        pos = pos + vel
        mean_val = float(np.mean(val))
        history.append((it, gbest_val, mean_val))
        if it % 20 == 0 or it == 1:
            print(f"iter={it:03d} best_j={gbest_val:.4f} mean_j={mean_val:.4f}", flush=True)
        if plateau_patience > 0 and stagnant >= plateau_patience:
            print(
                f"[INFO] PSO 早停：gbest_J 已连续 {stagnant} 次迭代提升 < {plateau_tol:g} mA/cm² "
                f"（plateau_patience={plateau_patience}）",
                flush=True,
            )
            stopped_plateau = True
            break

    np.save(out_dir / "best_latent.npy", gbest)
    np.save(out_dir / "pso_history.npy", np.asarray(history, dtype=np.float32))
    with (out_dir / "pso_history.csv").open("w", newline="", encoding="utf-8") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["iter", "best_j", "mean_j"])
        wcsv.writerows(history)

    j_idx = int(args.j_index)
    eta_mv = PAPER_ETA_MV_MILLIVOLT[j_idx] if 0 <= j_idx < len(PAPER_ETA_MV_MILLIVOLT) else None
    write_json(
        out_dir / "forward_manifest.json",
        {
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "surrogate": args.surrogate,
            "j_index": j_idx,
            "eta_mV_at_j_index": eta_mv,
            "columns": {
                "prior_j.npy": "每个先验样本在给定 eta 下的 J（与训练 labels_j 单位一致，默认 mA cm^-2）",
                "pso_history.csv": "iter, best_j, mean_j",
            },
            "pso": {
                "particles": particles,
                "iters_requested": int(args.iters),
                "iters_run": len(history),
                "c1": c1,
                "c2": c2,
                "w": w,
                "plateau_patience": plateau_patience,
                "plateau_patience_requested": plateau_patience_raw,
                "plateau_tol": plateau_tol,
                "stopped_by_plateau": stopped_plateau,
            },
            "output_spec": output_spec_block(),
        },
    )

    print(
        f"PSO 完成，best_j={gbest_val:.4f}，实际迭代 {len(history)}/{args.iters}"
        f"{'（已 plateau 早停）' if stopped_plateau else ''}，输出目录: {out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

