from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, mode as scipy_mode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.generator3d import Generator3D  # noqa: E402
from src.models.phys_dnn import PhysDNN  # noqa: E402
from src.physics.tpb_logic import active_tpb_density_from_label_volume  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def _density_on_grid(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    在 grid 上给出近似概率密度。若方差过小或 gaussian_kde 退化，则退回直方图 + 线性插值，
    避免出现 KDE 病态尖峰（常被误认为「一根竖线」）。
    """
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    g0, g1 = float(grid[0]), float(grid[-1])
    if v.size < 2 or g1 <= g0:
        return np.zeros_like(grid, dtype=np.float64)
    std = float(v.std())
    n_u = len(np.unique(np.round(v, 8)))
    use_hist = std < 1e-5 or n_u < 3 or v.size < 4
    if not use_hist:
        try:
            kde = gaussian_kde(v, bw_method="scott")
            y = kde(grid)
            if not np.all(np.isfinite(y)) or float(np.nanmax(y)) > 1e6:
                use_hist = True
            else:
                return np.maximum(y, 0.0)
        except (np.linalg.LinAlgError, ValueError):
            use_hist = True
    nb = int(min(40, max(8, int(np.sqrt(v.size)) + 3)))
    h, edges = np.histogram(v, bins=nb, range=(g0, g1), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = np.interp(grid.astype(np.float64), centers, h, left=0.0, right=0.0)
    return np.maximum(y, 0.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按论文/PPT风格生成可汇报图")
    p.add_argument("--config", default="configs/paper_params.yaml")
    p.add_argument("--gen-ckpt", default="outputs/gan_fallback/generator_fallback.pth")
    p.add_argument("--phys-dnn-ckpt", default="outputs/phys_models/phys_dnn.pth")
    p.add_argument("--data-dir", default="data/processed_fallback")
    p.add_argument("--out-dir", default="outputs/figures_fallback")
    p.add_argument("--num-samples", type=int, default=128)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--stratify-real",
        action="store_true",
        help="(a)(b) 中灰线也按与绿线相同的「生成 J」分位掩码切 real（更公平）；默认灰线为全体 real，易与条件子集形成视觉「两极」",
    )
    p.add_argument(
        "--cube-voxel-step",
        type=int,
        default=1,
        metavar="S",
        help="(d) 3D 体素图：S=1 时**全显**体素（与数据一致，64³ 可能较慢）；S>1 时用 S³ 块**众数**聚合成一格再画，仅为加速/省内存，不是丢掉多数体素不画",
    )
    return p.parse_args()


def onehot_to_phase_value(x3: np.ndarray) -> np.ndarray:
    """[3,D,H,W] -> [D,H,W] with values {0,128,255}."""
    idx = np.argmax(x3, axis=0)
    out = np.zeros_like(idx, dtype=np.float32)
    out[idx == 1] = 128.0
    out[idx == 2] = 255.0
    return out


def strict_tpb(vol_0128_255: np.ndarray) -> float:
    label = np.zeros_like(vol_0128_255, dtype=np.int8)
    label[vol_0128_255 == 0] = 0
    label[vol_0128_255 == 128] = 1
    label[vol_0128_255 == 255] = 2
    return active_tpb_density_from_label_volume(
        label,
        pore_value=0,
        ion_value=2,
        ele_value=1,
        min_connected_fraction=0.01,
    )


def _downsample_phase_majority(vol: np.ndarray, step: int = 4) -> np.ndarray:
    """
    可选降采样：将每个 S³ 子块聚成**一格**，块内取三相灰度**众数**（避免简单 [::S] 只取角点带来的各向偏差）。
    仅当 --cube-voxel-step S>1 时使用；S=1 时应对整卷直接绘图，不经过本函数。
    """
    d, h, w = vol.shape
    d2, h2, w2 = d // step, h // step, w // step
    if d2 < 1 or h2 < 1 or w2 < 1:
        return vol.astype(np.float32)
    trimmed = vol[: d2 * step, : h2 * step, : w2 * step]
    blocks = trimmed.reshape(d2, step, h2, step, w2, step)
    blocks = blocks.transpose(0, 2, 4, 1, 3, 5).reshape(d2, h2, w2, step**3)
    m = scipy_mode(blocks, axis=-1, keepdims=False)
    return np.asarray(m.mode, dtype=np.float32).reshape(d2, h2, w2)


def two_point_corr(mask: np.ndarray, max_d: int = 30) -> np.ndarray:
    vals = []
    for d in range(1, max_d + 1):
        c = (mask[:, :, :-d] & mask[:, :, d:]).mean()
        vals.append(float(c))
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size > 0 and arr[0] > 0:
        arr = arr / arr[0]
    return arr


def _plot_cube(ax, vol: np.ndarray, title: str, voxel_step: int = 1) -> None:
    step = max(1, int(voxel_step))
    if step <= 1:
        down = np.asarray(vol, dtype=np.float32)
    else:
        down = _downsample_phase_majority(vol, step=step)
    filled = np.ones_like(down, dtype=bool)
    colors = np.empty(down.shape, dtype=object)
    colors[down == 0] = "#1f1b52"      # Pore
    colors[down == 128] = "#20a39e"    # Ni
    colors[down == 255] = "#e5df3f"    # YSZ
    ax.voxels(filled, facecolors=colors, edgecolor="none")
    ax.set_title(title, pad=6)
    ax.set_axis_off()
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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

    phys = PhysDNN(input_size=1, hidden_dim=50, output_size=7).to(device)
    if Path(args.phys_dnn_ckpt).exists():
        phys.load_state_dict(torch.load(args.phys_dnn_ckpt, map_location=device))
    phys.eval()

    real_vol = np.load(Path(args.data_dir) / "volumes.npy").astype(np.float32)  # [N,1,64,64,64]
    real_j = np.load(Path(args.data_dir) / "labels_j.npy").astype(np.float32)  # [N,7]
    real_tpb = np.load(Path(args.data_dir) / "tpb.npy").astype(np.float32).reshape(-1)

    # 采样生成
    n = args.num_samples
    fake_vols = []
    fake_tpb = []
    fake_j = []
    with torch.no_grad():
        for _ in range(n):
            z = torch.randn(1, gan["latent_channels"], 4, 4, 4, device=device)
            label = torch.zeros(1, dtype=torch.long, device=device)
            x3 = gen(z, label)[0].cpu().numpy()  # [3,D,H,W]
            v = onehot_to_phase_value(x3)
            t = strict_tpb(v)
            j = phys(torch.tensor([[t]], dtype=torch.float32, device=device)).cpu().numpy()[0]
            fake_vols.append(v)
            fake_tpb.append(t)
            fake_j.append(j)
    fake_vols = np.stack(fake_vols, axis=0)
    fake_tpb = np.array(fake_tpb)
    fake_j = np.array(fake_j)

    # 论文 Fig.5 风格四联图：(a) J 分布分组，(b) TPB 分布分组，(c) 两点相关，(d) 低/中/高 J 立方体
    #
    # 为何有时看起来「两极分化」：
    # - (a)(b) 默认：绿线 = 按「生成样本在 η120 的 J」三分位切出的 fake 子集；灰线 = 全体 real。
    #   这是「条件分布 vs 边际分布」叠在同轴上，峰值必然拉开，不是真实世界物理量天然双模。
    # - (c)：Base 曲线对真实体素沿 H 轴 shuffle，故意破坏空间相关，用作 null；与 Real 分岔是预期。
    # - (d)：三格刻意选低/中/高分位上的样本，外观差异是展示目的。
    j_ref = fake_j[:, 5]
    q1, q2 = np.quantile(j_ref, [0.33, 0.66])
    groups = [
        ("Low J", j_ref <= q1),
        ("Intermediate J", (j_ref > q1) & (j_ref <= q2)),
        ("High J", j_ref > q2),
    ]
    # 可选：real 用与 fake 相同的分位掩码（在 real 上按 real_j[:,5] 分位，样本数与 fake 子集对齐意义见下）
    rq1, rq2 = np.quantile(real_j[:, 5], [0.33, 0.66])
    real_groups = [
        real_j[:, 5] <= rq1,
        (real_j[:, 5] > rq1) & (real_j[:, 5] <= rq2),
        real_j[:, 5] > rq2,
    ]
    real_tpb_groups = [
        real_tpb[real_groups[0]],
        real_tpb[real_groups[1]],
        real_tpb[real_groups[2]],
    ]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.1, 1.1, 1.2], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.25)

    # (a) 三行 J 分布
    gs_a = gs[0, 0].subgridspec(3, 1, hspace=0.08)
    all_min = float(min(real_j[:, 5].min(), fake_j[:, 5].min()))
    all_max = float(max(real_j[:, 5].max(), fake_j[:, 5].max()))
    xx = np.linspace(all_min, all_max, 200)
    for i, (name, mk) in enumerate(groups):
        ax = fig.add_subplot(gs_a[i, 0])
        if args.stratify_real:
            r_sel = real_j[:, 5][real_groups[i]]
            r = r_sel if r_sel.size > 0 else real_j[:, 5]
        else:
            r = real_j[:, 5]
        g = j_ref[mk] if np.any(mk) else j_ref
        kr_y = _density_on_grid(r, xx)
        kg_y = _density_on_grid(g, xx)
        ax.fill_between(xx, kg_y, color="#97cf62", alpha=0.65, label="Phys-DNN for J")
        ax.fill_between(xx, kr_y, color="#d8dfc0", alpha=0.85, label="Base case")
        ax.plot(xx, kg_y, color="#6fa23a", lw=1.0)
        ax.plot(xx, kr_y, color="#555555", lw=1.0)
        ax.axvline(float(g.mean()), color="#444444", ls="--", lw=0.8, alpha=0.7)
        ax.axvline(float(np.mean(r)), color="#444444", ls=":", lw=0.8, alpha=0.7)
        ax.set_xlim(all_min, all_max)
        if i == 0:
            ax.set_title("(a) Generated J distribution")
            ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.text(0.02, 0.82, name, transform=ax.transAxes, fontsize=10, fontweight="bold")
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Current density J (mA cm$^{-2}$)")
        if i == 1:
            ax.set_ylabel("Frequency")

    # (b) 三行 TPB 分布
    gs_b = gs[0, 1].subgridspec(3, 1, hspace=0.08)
    t_min = float(min(real_tpb.min(), fake_tpb.min()))
    t_max = float(max(real_tpb.max(), fake_tpb.max()))
    tt = np.linspace(t_min, t_max, 200)
    for i, (name, mk) in enumerate(groups):
        ax = fig.add_subplot(gs_b[i, 0])
        if args.stratify_real:
            r_t = real_tpb_groups[i]
            r = r_t if r_t.size > 0 else real_tpb
        else:
            r = real_tpb
        g = fake_tpb[mk] if np.any(mk) else fake_tpb
        kr_y = _density_on_grid(r, tt)
        kg_y = _density_on_grid(g, tt)
        ax.fill_between(tt, kg_y, color="#efd6b0", alpha=0.7, label="Phys-DNN for J")
        ax.fill_between(tt, kr_y, color="#a5a8ad", alpha=0.65, label="Base case")
        ax.plot(tt, kg_y, color="#8f6c3f", lw=1.0)
        ax.plot(tt, kr_y, color="#555555", lw=1.0)
        ax.axvline(float(g.mean()), color="#444444", ls="--", lw=0.8, alpha=0.7)
        ax.axvline(float(np.mean(r)), color="#444444", ls=":", lw=0.8, alpha=0.7)
        ax.set_xlim(t_min, t_max)
        if i == 0:
            ax.set_title("(b) Active TPB distribution")
            ax.legend(loc="upper left", fontsize=8, frameon=False)
        ax.text(0.02, 0.82, name, transform=ax.transAxes, fontsize=10, fontweight="bold")
        if i < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Active TPB length (proxy)")
        if i == 1:
            ax.set_ylabel("Frequency")

    # (c) 两点相关（Real/Base/Phys-DNN）
    axc = fig.add_subplot(gs[0, 2])
    max_d = 30
    xdist = np.arange(1, max_d + 1)
    phase_map = [("Pore", 0, "#000000"), ("Ni", 128, "#1b8a42"), ("YSZ", 255, "#6d2c90")]
    rng = np.random.default_rng(args.seed)
    idx_real = rng.choice(real_vol.shape[0], size=min(24, real_vol.shape[0]), replace=False)
    idx_fake = rng.choice(fake_vols.shape[0], size=min(24, fake_vols.shape[0]), replace=False)
    base = np.copy(real_vol[idx_real, 0])
    # 沿 H 打乱相别：破坏空间连续性，使两点相关衰减为「无结构」参照（与 Real 分岔属预期）
    rng.shuffle(base, axis=1)
    for pname, pval, col in phase_map:
        rc = np.mean([two_point_corr(real_vol[i, 0] == pval, max_d=max_d) for i in idx_real], axis=0)
        bc = np.mean([two_point_corr(base[k] == pval, max_d=max_d) for k in range(base.shape[0])], axis=0)
        fc = np.mean([two_point_corr(fake_vols[i] == pval, max_d=max_d) for i in idx_fake], axis=0)
        axc.plot(xdist, rc, color=col, lw=1.6, label=f"{pname} Real")
        axc.plot(xdist, bc, color=col, lw=1.2, ls="--", alpha=0.65, label=f"{pname} Base")
        axc.plot(xdist, fc, color=col, lw=1.2, ls=":", alpha=0.9, label=f"{pname} Phys-DNN")
    axc.set_title("(c) Two-point correlation")
    axc.set_xlabel("Two-point distance (voxel)")
    axc.set_ylabel("Two-point correlation coefficient")
    axc.set_xlim(1, max_d)
    axc.set_ylim(0.0, 1.05)
    axc.legend(fontsize=7, ncol=1, loc="upper right", frameon=False)

    # (d) 低/中/高 J 对应 3D 结构体
    gs_d = gs[1, :].subgridspec(1, 3, wspace=0.02)
    picks = []
    for _, mk in groups:
        idx = np.where(mk)[0]
        if idx.size == 0:
            picks.append(0)
        else:
            picks.append(int(idx[len(idx) // 2]))
    vstep = max(1, int(args.cube_voxel_step))
    for i, (name, _) in enumerate(groups):
        ax = fig.add_subplot(gs_d[0, i], projection="3d")
        _plot_cube(
            ax,
            fake_vols[picks[i]],
            f"(d) {name} ({j_ref[picks[i]]:.1f} mA cm$^{{-2}}$)",
            voxel_step=vstep,
        )

    note = (
        "灰/棕：全体 real（边际）"
        if not args.stratify_real
        else "灰/棕：与各行同分位的 real 子集（更可比）"
    )
    fig.suptitle(f"Paper-style figure (fallback reproduction)\n{note}；绿：按生成 J 分层的 fake；(c) 虚线 Base 为打乱 null", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out_dir / "fig_paper_style_f5_like.png", dpi=220)
    plt.close(fig)

    out_png = out_dir / "fig_paper_style_f5_like.png"
    print("=== 可汇报图已生成 ===", out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
