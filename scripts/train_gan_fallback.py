from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.losses.pi_loss import (  # noqa: E402
    compute_gradient_penalty,
    critic_wgan_gp_loss,
    generator_loss_with_physics,
)
from src.models.critic2d import Critic2D  # noqa: E402
from src.models.generator3d import Generator3D  # noqa: E402
from src.models.phys_dnn import PhysDNN  # noqa: E402
from src.physics.tpb_logic import active_tpb_density_from_label_volume  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


class GanFallbackDataset(Dataset):
    def __init__(self, data_dir: str | Path):
        root = Path(data_dir)
        self.volumes = np.load(root / "volumes.npy").astype(np.float32)  # [N,1,64,64,64]
        self.labels_j = np.load(root / "labels_j.npy").astype(np.float32)  # [N,7]
        if self.volumes.ndim != 5:
            raise ValueError(f"volumes 期望 5 维，当前 {self.volumes.shape}")

    def __len__(self) -> int:
        return self.volumes.shape[0]

    def __getitem__(self, idx: int):
        vol = torch.from_numpy(self.volumes[idx])  # [1,D,H,W]
        # 将单通道 phase id 转 one-hot 三通道，与生成器输出对齐
        pore = (vol == 0).float()
        ni = (vol == 128).float()
        ysz = (vol == 255).float()
        x3 = torch.cat([pore, ni, ysz], dim=0)  # [3,D,H,W]
        y = torch.from_numpy(self.labels_j[idx])
        # fallback 数据无真实 class，统一用 0
        cls = torch.tensor(0, dtype=torch.long)
        return x3, y, cls


def strict_tpb_density_from_onehot(x3: torch.Tensor) -> torch.Tensor:
    """
    更严格 TPB 代理：基于 6 邻域三相接触统计。
    x3: [B,3,D,H,W]，通道顺序 [pore, ni, ysz]
    返回: [B,1]
    """
    if x3.ndim != 5 or x3.shape[1] != 3:
        raise ValueError(f"x3 期望 [B,3,D,H,W]，当前 {x3.shape}")
    hard = torch.argmax(x3, dim=1).detach().cpu().numpy().astype(np.int8)
    vals = []
    for i in range(hard.shape[0]):
        tpb_i = active_tpb_density_from_label_volume(
            hard[i],
            pore_value=0,
            ion_value=2,  # 当前 one-hot 通道: [pore, ni, ysz]
            ele_value=1,
            min_connected_fraction=0.01,
        )
        vals.append(tpb_i)
    out = torch.tensor(vals, device=x3.device, dtype=torch.float32)[:, None]
    return out


def fast_tpb_density_from_onehot(x3: torch.Tensor) -> torch.Tensor:
    """
    快速 TPB 代理（全 torch，可在 GPU 运行）：
    使用 6 邻域接触计数近似，速度显著快于严格连通域版本。
    x3: [B,3,D,H,W]，通道顺序 [pore, ni, ysz]
    返回: [B,1]
    """
    if x3.ndim != 5 or x3.shape[1] != 3:
        raise ValueError(f"x3 期望 [B,3,D,H,W]，当前 {x3.shape}")
    hard = torch.argmax(x3, dim=1)  # [B,D,H,W]
    pore = hard == 0
    ni = hard == 1
    ysz = hard == 2

    def _contact(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        c = (
            (a[:, :-1] & b[:, 1:]).sum(dim=(1, 2, 3))
            + (b[:, :-1] & a[:, 1:]).sum(dim=(1, 2, 3))
            + (a[:, :, :-1] & b[:, :, 1:]).sum(dim=(1, 2, 3))
            + (b[:, :, :-1] & a[:, :, 1:]).sum(dim=(1, 2, 3))
            + (a[:, :, :, :-1] & b[:, :, :, 1:]).sum(dim=(1, 2, 3))
            + (b[:, :, :, :-1] & a[:, :, :, 1:]).sum(dim=(1, 2, 3))
        )
        return c.float()

    contact_ni_ysz = _contact(ni, ysz)
    contact_pore_ni = _contact(pore, ni)
    denom = float(hard.shape[1] * hard.shape[2] * hard.shape[3])
    tpb = (contact_ni_ysz + 0.5 * contact_pore_ni) / max(denom, 1.0)
    return tpb[:, None]


def soft_tpb_density_from_probs(x3: torch.Tensor) -> torch.Tensor:
    """
    可导 TPB 近似：基于三相概率场的邻域接触项构造。
    目的：为生成器提供稳定梯度（训练），并可与 strict 前向值做 STE 融合。
    x3: [B,3,D,H,W]，通道顺序 [pore, ni, ysz]
    返回: [B,1]
    """
    if x3.ndim != 5 or x3.shape[1] != 3:
        raise ValueError(f"x3 期望 [B,3,D,H,W]，当前 {x3.shape}")
    pore = x3[:, 0]
    ni = x3[:, 1]
    ysz = x3[:, 2]

    def _soft_contact(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        c = (
            (a[:, :-1] * b[:, 1:]).sum(dim=(1, 2, 3))
            + (b[:, :-1] * a[:, 1:]).sum(dim=(1, 2, 3))
            + (a[:, :, :-1] * b[:, :, 1:]).sum(dim=(1, 2, 3))
            + (b[:, :, :-1] * a[:, :, 1:]).sum(dim=(1, 2, 3))
            + (a[:, :, :, :-1] * b[:, :, :, 1:]).sum(dim=(1, 2, 3))
            + (b[:, :, :, :-1] * a[:, :, :, 1:]).sum(dim=(1, 2, 3))
        )
        return c

    contact_ni_ysz = _soft_contact(ni, ysz)
    contact_pore_ni = _soft_contact(pore, ni)
    denom = float(x3.shape[2] * x3.shape[3] * x3.shape[4])
    tpb = (contact_ni_ysz + 0.5 * contact_pore_ni) / max(denom, 1.0)
    return tpb[:, None]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="基于 fallback 数据训练 GAN（轻量）")
    p.add_argument("--config", default="configs/paper_params.yaml")
    p.add_argument("--data-dir", default="data/processed_fallback")
    p.add_argument("--out-dir", default="outputs/gan_fallback")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader 进程数（GPU 建议 4~16）")
    p.add_argument("--pin-memory", action="store_true", help="GPU 训练时开启 pin_memory")
    p.add_argument(
        "--tpb-mode",
        choices=["strict", "fast"],
        default="strict",
        help="TPB 计算模式：strict=严格连通域(慢)，fast=6邻域近似(快，适合GPU大规模训练)",
    )
    p.add_argument(
        "--tpb-grad-mode",
        choices=["ste_soft", "none"],
        default="none",
        help="strict 模式下物理损失梯度策略：ste_soft=前向strict/反向soft；none=不做梯度近似",
    )
    p.add_argument("--amp", action="store_true", help="启用混合精度训练（仅 CUDA 生效）")
    p.add_argument("--phys-dnn-ckpt", default="outputs/phys_models/phys_dnn.pth")
    p.add_argument(
        "--training-stage",
        choices=["normal", "physics"],
        default="physics",
        help="训练阶段：normal=仅WGAN-GP预训练，physics=按论文物理损失微调",
    )
    p.add_argument("--resume-gen", default="", help="可选：从已有 generator 权重继续")
    p.add_argument("--resume-critic", default="", help="可选：从已有 critic 权重继续")
    p.add_argument("--save-every", type=int, default=0, help="每隔多少 epoch 额外保存一次中间权重（0关闭）")
    p.add_argument(
        "--disable-strict-paper",
        action="store_true",
        help="关闭严格论文模式。默认严格按论文损失：仅 L_G_org + gamma*MAE",
    )
    p.add_argument(
        "--wdist-stable-window",
        type=int,
        default=0,
        help="用最近多少个 epoch 的「批均值 Wasserstein(w_dist)」判断稳定；0=关闭（跑满 --epochs）。论文：距离变 steady 时停，常在千轮量级",
    )
    p.add_argument(
        "--wdist-stable-std-tol",
        type=float,
        default=0.03,
        help="最近 window 个 epoch 的 w_dist epoch 均值标准差 < 该值则该 epoch 记为「稳定」",
    )
    p.add_argument(
        "--wdist-stable-patience",
        type=int,
        default=0,
        help="连续多少个 epoch 均「稳定」则提前结束；0=关闭。与 window 同时>0 时生效",
    )
    p.add_argument(
        "--physics-closs-std-tol",
        type=float,
        default=None,
        help="physics：c_loss 批均值窗口 std 上界；省略则读配置文件 paper_repro.gan_physics_closs_std_tol；均为空则不启用第二判据。",
    )
    p.add_argument(
        "--physics-skip-closs-stable",
        action="store_true",
        help="physics 阶段：仅按 w_dist 稳定早停，不强制 c_loss 同步稳定。",
    )
    p.add_argument(
        "--gan-preview-every",
        type=int,
        default=0,
        help="每 N 个 epoch 保存一张生成体中间切片 PNG（0=禁用），便于人工目视检查。",
    )
    p.add_argument(
        "--force-tiny",
        action="store_true",
        help="强制使用 gan.tiny_channels（覆盖 yaml 的 debug_tiny；24GB 显存下 paper 网 + 大 batch 易 OOM）",
    )
    return p.parse_args()


def _export_gan_preview_png(
    gen: Generator3D,
    device: torch.device,
    z_fix: torch.Tensor,
    label_fix: torch.Tensor,
    path: Path,
    *,
    use_amp: bool,
) -> None:
    """保存生成体中间深度切片 RGB PNG，便于人工目视。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    was_training = gen.training
    gen.eval()
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                x = gen(z_fix, label_fix)
        d = int(x.shape[2])
        mid = x[0, :, d // 2, :, :].detach().float().cpu().numpy()
        mid = np.clip(mid, 0.0, 1.0)
        rgb = np.transpose(mid, (1, 2, 0))
        plt.imsave(str(path), rgb, vmin=0.0, vmax=1.0, format="png")
        plt.close("all")
    finally:
        if was_training:
            gen.train()


def main() -> int:
    args = parse_args()
    strict_paper = not bool(args.disable_strict_paper)
    is_physics_stage = args.training_stage == "physics"
    if is_physics_stage and strict_paper and args.tpb_mode != "strict":
        raise ValueError("严格论文模式下必须使用 --tpb-mode strict。")
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pr_cfg = cfg.get("paper_repro") or {}
    yaml_closs_tol = pr_cfg.get("gan_physics_closs_std_tol", None)
    if args.physics_closs_std_tol is not None:
        resolved_closs_tol: float | None = float(args.physics_closs_std_tol)
    elif yaml_closs_tol is not None:
        resolved_closs_tol = float(yaml_closs_tol)
    else:
        resolved_closs_tol = None

    gan = cfg["gan"]
    use_tiny = bool(args.force_tiny) or bool(gan.get("debug_tiny", True))
    g_channels = gan["tiny_channels"]["generator"] if use_tiny else gan["paper_channels"]["generator"]
    c_channels = gan["tiny_channels"]["critic"] if use_tiny else gan["paper_channels"]["critic"]

    ds = GanFallbackDataset(args.data_dir)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=bool(args.pin_memory and device.type == "cuda"),
        persistent_workers=bool(args.num_workers > 0),
    )

    gen = Generator3D(
        z_channels=gan["latent_channels"],
        num_classes=gan["classes_inverse"],
        embed_dim=gan["embedding_size"],
        channels=g_channels,
    ).to(device)
    critic = Critic2D(
        in_channels=3,
        num_classes=gan["classes_inverse"],
        embed_dim=gan["embedding_size"],
        channels=c_channels,
    ).to(device)
    if args.resume_gen:
        gen.load_state_dict(torch.load(args.resume_gen, map_location=device))
    if args.resume_critic:
        critic.load_state_dict(torch.load(args.resume_critic, map_location=device))

    # 使用预训练 phys-dnn 作为 physics 约束（论文流程为先预训练再用于 GAN）
    phys = PhysDNN(input_size=1, hidden_dim=50, output_size=7).to(device)
    if is_physics_stage:
        ckpt = Path(args.phys_dnn_ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"缺少 phys-dnn 权重: {ckpt}")
        phys.load_state_dict(torch.load(ckpt, map_location=device))
        phys.eval()
        for p in phys.parameters():
            p.requires_grad_(False)

    opt_g = torch.optim.Adam(gen.parameters(), lr=float(gan["learning_rate"]), betas=tuple(gan["adam_betas"]))
    opt_c = torch.optim.Adam(critic.parameters(), lr=float(gan["learning_rate"]), betas=tuple(gan["adam_betas"]))
    use_amp = bool(args.amp and device.type == "cuda")
    scaler_g = torch.cuda.amp.GradScaler(enabled=use_amp)
    scaler_c = torch.cuda.amp.GradScaler(enabled=use_amp)

    use_wdist_stop = int(args.wdist_stable_window) > 0 and int(args.wdist_stable_patience) > 0
    dual_required = bool(
        is_physics_stage
        and use_wdist_stop
        and not bool(args.physics_skip_closs_stable)
        and resolved_closs_tol is not None
    )
    z_preview = torch.randn(1, gan["latent_channels"], 4, 4, 4, device=device)
    lbl_preview = torch.zeros(1, dtype=torch.long, device=device)

    print("=== 开始 GAN fallback 训练 ===")
    print(
        f"stage={args.training_stage}, device={device}, tiny={use_tiny}, max_epochs={args.epochs}, "
        f"batch={args.batch_size}, tpb_mode={args.tpb_mode}, workers={args.num_workers}, amp={use_amp}"
        + (
            f", wdist_early_stop=window{args.wdist_stable_window}/std<{args.wdist_stable_std_tol}/"
            f"patience{args.wdist_stable_patience}"
            if use_wdist_stop
            else ", wdist_early_stop=off"
        )
        + (
            f", physics_dual_stable=c_loss_std<{resolved_closs_tol}"
            if dual_required
            else ""
        ),
        flush=True,
    )
    metrics_path = out_dir / f"metrics_{args.training_stage}.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["epoch", "step", "w_dist", "c_loss", "g_total", "g_org", "g_phys", "tpb_mean"])
    step = 0
    epoch_w_means: list[float] = []
    epoch_c_means: list[float] = []
    stable_streak = 0
    stopped_early = False
    epoch_last = 0
    # 论文：在 Wasserstein 距离 steady、形貌目测可接受时结束（1000/300 为经验上常出现的上限附近）。
    # 此处用 epoch 级 w_dist 均值滚动标准差近似「steady」；目测仍须人工看中间权重或 tensorboard。
    for epoch in range(1, args.epochs + 1):
        epoch_last = int(epoch)
        sum_w = 0.0
        n_w = 0
        sum_c = 0.0
        n_c = 0
        for real, j_target, labels in dl:
            step += 1
            real = real.to(device)
            j_target = j_target.to(device)
            labels = labels.to(device)

            z = torch.randn(real.size(0), gan["latent_channels"], 4, 4, 4, device=device)

            # critic: 与论文超参数一致，默认每次 G 更新前更新 5 次 critic
            n_critic = max(1, int(gan.get("critic_updates_per_g", 5)))
            last_c_loss = 0.0
            w_dist = torch.zeros((), device=device)
            for _ in range(n_critic):
                z_c = torch.randn(real.size(0), gan["latent_channels"], 4, 4, 4, device=device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    fake_c = gen(z_c, labels).detach()
                    real_score = critic(real, labels)
                    fake_score = critic(fake_c, labels)
                w_dist = (real_score.mean() - fake_score.mean()).detach()
                gp = compute_gradient_penalty(critic, real, fake_c, labels, device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    c_loss = critic_wgan_gp_loss(real_score, fake_score, gp, lambda_gp=float(gan["lambda_gp"]))
                opt_c.zero_grad()
                scaler_c.scale(c_loss).backward()
                scaler_c.step(opt_c)
                scaler_c.update()
                last_c_loss = float(c_loss.detach().item())
                if device.type == "cuda":
                    del fake_c, real_score, fake_score, gp
                    torch.cuda.empty_cache()
            sum_w += float(w_dist.item())
            n_w += 1
            sum_c += last_c_loss
            n_c += 1

            # generator
            with torch.cuda.amp.autocast(enabled=use_amp):
                fake = gen(z, labels)
                fake_score = critic(fake, labels)
            if is_physics_stage:
                if args.tpb_mode == "strict":
                    # 严格前向：按逻辑模型计算 active TPB
                    strict_tpb = strict_tpb_density_from_onehot(fake)
                    if args.tpb_grad_mode == "ste_soft":
                        # 梯度修复：前向使用 strict 值，反向梯度来自可导 soft 近似（STE）
                        soft_tpb = soft_tpb_density_from_probs(fake)
                        tpb_scalar = strict_tpb + (soft_tpb - soft_tpb.detach())
                    else:
                        tpb_scalar = strict_tpb
                else:
                    tpb_scalar = fast_tpb_density_from_onehot(fake)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred_j = phys(tpb_scalar)
                    g_total, g_org, g_phys = generator_loss_with_physics(
                        fake_score, pred_j, j_target, gamma_phys=float(gan["gamma_phys"])
                    )
            else:
                # normal 阶段：仅 WGAN-GP 原始生成器目标
                g_org = -fake_score.mean()
                g_phys = torch.zeros((), device=device)
                g_total = g_org
                tpb_scalar = torch.zeros((real.size(0), 1), device=device)

            # 相分数一致性约束：避免生成结果塌缩为单一相分布
            fake_pf = fake.mean(dim=(2, 3, 4))  # [B,3]
            real_pf = real.mean(dim=(2, 3, 4))  # [B,3]
            l_pf = torch.mean(torch.abs(fake_pf - real_pf))

            # 弱多样性约束：鼓励 batch 内样本存在差异
            l_div = -torch.mean(torch.std(fake_pf, dim=0, unbiased=False))
            if strict_paper:
                # 严格对齐论文 Eq.(16): L_G = L_G_org + gamma * MAE
                pass
            else:
                # 非严格模式下保留工程稳定项（不属于论文原式）
                g_total = g_total + 0.5 * l_pf + 0.05 * l_div
            opt_g.zero_grad()
            scaler_g.scale(g_total).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if step % 10 == 0 or step == 1:
                print(
                    f"epoch={epoch} step={step} "
                    f"w_dist={w_dist.item():.4f} "
                    f"c_loss={last_c_loss:.4f} g_total={g_total.item():.4f} "
                    f"g_org={g_org.item():.4f} g_phys={g_phys.item():.4f} "
                    f"pf={l_pf.item():.4f}"
                )
            with metrics_path.open("a", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(
                    [
                        epoch,
                        step,
                        float(w_dist.item()),
                        float(last_c_loss),
                        float(g_total.item()),
                        float(g_org.item()),
                        float(g_phys.item()),
                        float(tpb_scalar.mean().detach().item()),
                    ]
                )

        epoch_mean_w = sum_w / max(n_w, 1)
        epoch_mean_c = sum_c / max(n_c, 1)
        epoch_w_means.append(float(epoch_mean_w))
        epoch_c_means.append(float(epoch_mean_c))
        if use_wdist_stop and len(epoch_w_means) >= int(args.wdist_stable_window):
            tail_w = epoch_w_means[-int(args.wdist_stable_window) :]
            w_std = float(np.std(np.asarray(tail_w, dtype=np.float64)))
            w_stable = w_std < float(args.wdist_stable_std_tol)
            if dual_required:
                tail_c = epoch_c_means[-int(args.wdist_stable_window) :]
                c_std = float(np.std(np.asarray(tail_c, dtype=np.float64)))
                c_stable = c_std < float(resolved_closs_tol)
                stable_now = w_stable and c_stable
            else:
                c_std = 0.0
                stable_now = w_stable
            if stable_now:
                stable_streak += 1
            else:
                stable_streak = 0
            msg = (
                f"[INFO] epoch={epoch} mean_w_dist={epoch_mean_w:.6f} w_tail_std={w_std:.6f} "
                f"stable_streak={stable_streak}/{args.wdist_stable_patience}"
            )
            if dual_required:
                msg += f" mean_c_loss={epoch_mean_c:.6f} c_tail_std={c_std:.6f}"
            print(msg, flush=True)
            if stable_streak >= int(args.wdist_stable_patience):
                if dual_required:
                    print(
                        "[INFO] GAN 早停（physics）：w_dist 与 c_loss 的 epoch 批均值均在滑动窗口内足够稳定 "
                        f"（window={args.wdist_stable_window}, w_std_tol={args.wdist_stable_std_tol}, "
                        f"c_std_tol={resolved_closs_tol}, patience={args.wdist_stable_patience}）。"
                        " 形貌仍建议对照 previews/ 或 save-every 权重做人工抽查。",
                        flush=True,
                    )
                else:
                    print(
                        "[INFO] GAN 早停：Wasserstein 距离（epoch 批均值）在滑动窗口内已足够稳定 "
                        f"（window={args.wdist_stable_window}, std_tol={args.wdist_stable_std_tol}, "
                        f"patience={args.wdist_stable_patience}）。目测形貌请自行抽查 checkpoint 或 previews/。",
                        flush=True,
                    )
                stopped_early = True
                break
        else:
            if epoch <= 3 or epoch % 50 == 0 or epoch == args.epochs:
                print(f"[INFO] epoch={epoch} mean_w_dist={epoch_mean_w:.6f}", flush=True)

        prev_n = int(getattr(args, "gan_preview_every", 0) or 0)
        if prev_n > 0 and epoch % prev_n == 0:
            prev_path = out_dir / "previews" / f"{args.training_stage}_e{epoch:04d}.png"
            _export_gan_preview_png(
                gen, device, z_preview, lbl_preview, prev_path, use_amp=use_amp
            )

        if args.save_every > 0 and (epoch % args.save_every == 0):
            torch.save(gen.state_dict(), out_dir / f"generator_{args.training_stage}_e{epoch}.pth")
            torch.save(critic.state_dict(), out_dir / f"critic_{args.training_stage}_e{epoch}.pth")

    torch.save(gen.state_dict(), out_dir / "generator_fallback.pth")
    torch.save(critic.state_dict(), out_dir / "critic_fallback.pth")
    manifest = {
        "training_stage": args.training_stage,
        "max_epochs": int(args.epochs),
        "epochs_run": int(epoch_last),
        "stopped_by_wdist_plateau": bool(stopped_early),
        "physics_dual_closs_stable": bool(dual_required),
        "physics_skip_closs_stable": bool(args.physics_skip_closs_stable),
        "physics_closs_std_tol": float(resolved_closs_tol) if resolved_closs_tol is not None else None,
        "gan_preview_every": int(getattr(args, "gan_preview_every", 0) or 0),
        "wdist_stable_window": int(args.wdist_stable_window) if use_wdist_stop else 0,
        "wdist_stable_std_tol": float(args.wdist_stable_std_tol),
        "wdist_stable_patience": int(args.wdist_stable_patience) if use_wdist_stop else 0,
        "epoch_w_means_tail": [float(x) for x in epoch_w_means[-16:]],
        "epoch_c_means_tail": [float(x) for x in epoch_c_means[-16:]],
    }
    (out_dir / f"gan_exit_{args.training_stage}.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"训练完成，模型已保存到: {out_dir}（实际 epoch={manifest['epochs_run']}"
        f"{'，已 w_dist 早停' if stopped_early else ''}）",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
