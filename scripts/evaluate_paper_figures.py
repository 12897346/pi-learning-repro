from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.generator3d import Generator3D  # noqa: E402
from src.models.phys_dnn import PhysDNN  # noqa: E402
from src.physics.tpb_logic import active_tpb_density_from_label_volume  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.output_manifest import write_figure_bundle_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按论文 Figure2/3/5/6 风格输出图")
    p.add_argument("--config", default="configs/paper_params.yaml")
    p.add_argument("--data-dir", default="data/processed_fallback")
    p.add_argument("--gen-ckpt", default="outputs/gan_fallback/generator_fallback.pth")
    p.add_argument("--phys-dnn-ckpt", default="outputs/phys_models/phys_dnn.pth")
    p.add_argument("--gan-metrics", default="outputs/gan_fallback/metrics_physics.csv")
    p.add_argument("--pso-history", default="outputs/forward_design/pso_history.csv")
    p.add_argument("--prior-j", default="outputs/forward_design/prior_j.npy")
    p.add_argument("--best-latent", default="outputs/forward_design/best_latent.npy")
    p.add_argument("--phys-cnn-ckpt", default="outputs/phys_models/phys_cnn.pth")
    p.add_argument("--openfoam-dir", default="", help="OpenFOAM 导出目录（严格模式必填）")
    p.add_argument("--strict-no-proxy", action="store_true", help="严格模式：禁止代理图，缺少真场输入即报错")
    p.add_argument("--out-dir", default="outputs/paper_figures")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def onehot_to_phase(x3: np.ndarray) -> np.ndarray:
    idx = np.argmax(x3, axis=0)
    out = np.zeros_like(idx, dtype=np.float32)
    out[idx == 1] = 128.0
    out[idx == 2] = 255.0
    return out


def strict_tpb(v: np.ndarray) -> float:
    label = np.zeros_like(v, dtype=np.int8)
    label[v == 0] = 0
    label[v == 128] = 1
    label[v == 255] = 2
    return active_tpb_density_from_label_volume(label, pore_value=0, ion_value=2, ele_value=1)


def total_tpb(v: np.ndarray) -> float:
    label = np.zeros_like(v, dtype=np.int8)
    label[v == 0] = 0
    label[v == 128] = 1
    label[v == 255] = 2
    c000 = label
    c0m0 = np.roll(label, shift=1, axis=1)
    c00m = np.roll(label, shift=1, axis=2)
    c0mm = np.roll(c0m0, shift=1, axis=2)
    cm00 = np.roll(label, shift=1, axis=0)
    cm0m = np.roll(cm00, shift=1, axis=2)
    cmm0 = np.roll(cm00, shift=1, axis=1)

    def _contains_three(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        has0 = (a == 0) | (b == 0) | (c == 0) | (d == 0)
        has1 = (a == 1) | (b == 1) | (c == 1) | (d == 1)
        has2 = (a == 2) | (b == 2) | (c == 2) | (d == 2)
        return has0 & has1 & has2

    tpb_x = _contains_three(c000, c0m0, c00m, c0mm)
    tpb_y = _contains_three(c000, cm00, c00m, cm0m)
    tpb_z = _contains_three(c000, cm00, c0m0, cmm0)
    count = int(np.sum(tpb_x) + np.sum(tpb_y) + np.sum(tpb_z))
    return float(count / max(float(v.size), 1.0))


def two_point_corr(mask: np.ndarray, max_d: int = 30) -> np.ndarray:
    arr = []
    for d in range(1, max_d + 1):
        arr.append(float((mask[:, :, :-d] & mask[:, :, d:]).mean()))
    x = np.asarray(arr, dtype=np.float32)
    if x.size and x[0] > 0:
        x = x / x[0]
    return x


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.strict_no_proxy:
        # 清理历史代理产物，避免旧文件误导为“仍在代理评估”
        for p in out_dir.glob("*proxy*"):
            p.unlink()
    device = torch.device(args.device)
    produced_png: list[str] = []

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
    phys.load_state_dict(torch.load(args.phys_dnn_ckpt, map_location=device))
    phys.eval()

    real_j = np.load(Path(args.data_dir) / "labels_j.npy").astype(np.float32)
    real_vol = np.load(Path(args.data_dir) / "volumes.npy").astype(np.float32)

    fake_j = []
    fake_tpb = []
    fake_vols = []
    with torch.no_grad():
        for _ in range(args.num_samples):
            z = torch.randn(1, gan["latent_channels"], 4, 4, 4, device=device)
            label = torch.zeros(1, dtype=torch.long, device=device)
            x3 = gen(z, label)[0].cpu().numpy()
            v = onehot_to_phase(x3)
            t = strict_tpb(v)
            j = phys(torch.tensor([[t]], dtype=torch.float32, device=device)).cpu().numpy()[0]
            fake_j.append(j)
            fake_tpb.append(t)
            fake_vols.append(v)
    fake_j = np.asarray(fake_j, dtype=np.float32)
    fake_tpb = np.asarray(fake_tpb, dtype=np.float32)
    fake_vols = np.asarray(fake_vols, dtype=np.float32)

    # Figure 3 风格：phys-DNN 回归与误差分布
    real_tpb = np.load(Path(args.data_dir) / "tpb.npy").astype(np.float32).reshape(-1)
    with torch.no_grad():
        pred_real = phys(torch.from_numpy(real_tpb[:, None]).to(device)).cpu().numpy().astype(np.float32)
    real_flat = real_j.reshape(-1)
    pred_flat = pred_real.reshape(-1)
    eps = pred_flat - real_flat
    mae = float(np.mean(np.abs(eps)))
    r2 = float(1.0 - np.sum((pred_flat - real_flat) ** 2) / max(np.sum((real_flat - real_flat.mean()) ** 2), 1e-8))
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    eta_mv = np.array([20, 40, 60, 80, 100, 120, 140], dtype=np.float32)
    for i in range(min(12, real_j.shape[0])):
        ax1.plot(eta_mv, real_j[i], color="#444", alpha=0.25, lw=0.8)
        ax1.plot(eta_mv, pred_real[i], color="#f0b429", alpha=0.25, lw=0.8)
    ax1.plot(eta_mv, real_j.mean(axis=0), color="black", lw=1.8, label="Real mean")
    ax1.plot(eta_mv, pred_real.mean(axis=0), color="#e74c3c", lw=1.8, label="Pred mean")
    ax1.set_title("(a) J-eta curves")
    ax1.set_xlabel(r"$\eta$ (mV)")
    ax1.set_ylabel(r"$J$ (mA cm$^{-2}$)")
    ax1.legend(frameon=False, fontsize=8)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(real_flat, pred_flat, s=8, alpha=0.25, color="#8e44ad")
    lo = float(min(real_flat.min(), pred_flat.min()))
    hi = float(max(real_flat.max(), pred_flat.max()))
    ax2.plot([lo, hi], [lo, hi], color="black", lw=1.0)
    ax2.set_title(f"(b) Pred vs Real, $R^2$={r2:.3f}")
    ax2.set_xlabel("Real J")
    ax2.set_ylabel("Predicted J")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.hist(eps, bins=24, density=True, color="#f7d27d", edgecolor="#666")
    ax3.set_title(f"(c) Error density, MAE={mae:.2f}")
    ax3.set_xlabel("Prediction error")
    ax3.set_ylabel("Density")
    fig.tight_layout()
    fig.savefig(out_dir / "figure3_phys_dnn_regression.png", dpi=220)
    produced_png.append("figure3_phys_dnn_regression.png")
    plt.close(fig)

    # Figure 5a 风格：三档 J 分布
    jv = fake_j[:, 5]
    q1, q2 = np.quantile(jv, [0.33, 0.66])
    # 分组标签由当前样本分位数自动生成，避免硬编码“看似论文数值”的虚空赋值
    low_lo, low_hi = float(np.min(jv[jv <= q1])) if np.any(jv <= q1) else float(np.min(jv)), float(q1)
    mid_lo, mid_hi = float(q1), float(q2)
    high_lo, high_hi = float(q2), float(np.max(jv))
    groups = [
        (f"{low_lo:.1f}-{low_hi:.1f}", jv <= q1),
        (f"{mid_lo:.1f}-{mid_hi:.1f}", (jv > q1) & (jv <= q2)),
        (f"{high_lo:.1f}-{high_hi:.1f}", jv > q2),
    ]
    fig, axs = plt.subplots(3, 2, figsize=(12, 9), sharex="col")
    xj = np.linspace(min(real_j[:, 5].min(), jv.min()), max(real_j[:, 5].max(), jv.max()), 300)
    xt = np.linspace(float(min(fake_tpb.min(), 1e9)), float(max(fake_tpb.max(), -1e9)), 300)
    for i, (name, mk) in enumerate(groups):
        gj = jv[mk] if np.any(mk) else jv
        gt = fake_tpb[mk] if np.any(mk) else fake_tpb
        axs[i, 0].fill_between(xj, gaussian_kde(gj)(xj), alpha=0.7, color="#94c86f", label="Phys-DNN for J")
        axs[i, 0].fill_between(xj, gaussian_kde(real_j[:, 5])(xj), alpha=0.6, color="#d9dfbf", label="Base case")
        axs[i, 0].text(0.03, 0.86, name, transform=axs[i, 0].transAxes, fontsize=9, fontweight="bold")
        axs[i, 1].fill_between(xt, gaussian_kde(gt)(xt), alpha=0.7, color="#efcfa7", label="Phys-DNN for J")
        axs[i, 1].fill_between(xt, gaussian_kde(real_tpb)(xt), alpha=0.6, color="#acb0b3", label="Base case")
    axs[0, 0].set_title("(a) Generated J distributions")
    axs[0, 1].set_title("(b) Active TPB distributions")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 1].set_ylabel("Frequency")
    axs[2, 0].set_xlabel("Current density J (mA cm$^{-2}$)")
    axs[2, 1].set_xlabel("Active TPB length (proxy)")
    axs[0, 0].legend(frameon=False, fontsize=8)
    axs[0, 1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figure5_ab_distributions.png", dpi=220)
    produced_png.append("figure5_ab_distributions.png")
    plt.close(fig)

    # Figure 5c 风格：two-point correlation
    fig = plt.figure(figsize=(7, 5))
    max_d = 30
    xs = np.arange(1, max_d + 1)
    rng = np.random.default_rng(args.seed)
    ridx = rng.choice(real_vol.shape[0], size=min(32, real_vol.shape[0]), replace=False)
    fidx = rng.choice(fake_vols.shape[0], size=min(32, fake_vols.shape[0]), replace=False)
    for name, val, c in [("Pore", 0, "black"), ("Ni", 128, "#188a42"), ("YSZ", 255, "#6c2f90")]:
        rr = np.mean([two_point_corr(real_vol[i, 0] == val, max_d=max_d) for i in ridx], axis=0)
        ff = np.mean([two_point_corr(fake_vols[i] == val, max_d=max_d) for i in fidx], axis=0)
        plt.plot(xs, rr, color=c, lw=1.6, label=f"{name} real")
        plt.plot(xs, ff, color=c, lw=1.4, ls=":", label=f"{name} phys-DNN")
    plt.xlabel("Two-point distance (voxel)")
    plt.ylabel("Two-point correlation coefficient")
    plt.title("(c) Two-point correlation")
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "figure5_c_two_point.png", dpi=220)
    produced_png.append("figure5_c_two_point.png")
    plt.close()

    # Figure 6a 风格：prior + PSO 轨迹
    if Path(args.prior_j).exists() and Path(args.pso_history).exists():
        prior_j = np.load(args.prior_j).astype(np.float32)
        pso = np.genfromtxt(args.pso_history, delimiter=",", names=True)
        plt.figure(figsize=(8, 5))
        plt.hist(prior_j, bins=60, density=True, alpha=0.4, color="#8ccf73", label="Prior distribution")
        plt.plot(pso["iter"], pso["best_j"], color="#2f5aa8", lw=2.0, label="PSO best J")
        plt.xlabel("Iteration / J value axis")
        plt.ylabel("Density / J")
        plt.title("Figure 6a style: Prior J and PSO trajectory")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / "figure6_a_prior_pso.png", dpi=220)
        produced_png.append("figure6_a_prior_pso.png")
        plt.close()
        # Figure 6b/c/d 风格：最优点对比+潜变量可视化+最优结构快照
        if Path(args.best_latent).exists():
            zflat = np.load(args.best_latent).astype(np.float32)
            z = torch.from_numpy(zflat[None, :]).to(device).view(1, gan["latent_channels"], 4, 4, 4)
            with torch.no_grad():
                x3 = gen(z, torch.zeros(1, dtype=torch.long, device=device))[0].cpu().numpy()
            vbest = onehot_to_phase(x3)
            tbest = strict_tpb(vbest)
            with torch.no_grad():
                j_dnn = float(phys(torch.tensor([[tbest]], dtype=torch.float32, device=device)).cpu().numpy()[0, 5])
            # phys-cnn 对比
            j_cnn = np.nan
            if Path(args.phys_cnn_ckpt).exists():
                from src.models.phys_cnn import PhysCNN  # 局部导入，避免不必要依赖
                from src.physics.tpb_logic import active_union_mask_from_label_volume
                pc = PhysCNN(in_channels=4, out_dim=7).to(device)
                pc.load_state_dict(torch.load(args.phys_cnn_ckpt, map_location=device))
                pc.eval()
                label = np.zeros_like(vbest, dtype=np.int8)
                label[vbest == 0] = 0
                label[vbest == 128] = 1
                label[vbest == 255] = 2
                feat = np.stack(
                    [
                        (vbest == 0).astype(np.float32),
                        (vbest == 128).astype(np.float32),
                        (vbest == 255).astype(np.float32),
                        active_union_mask_from_label_volume(label, pore_value=0, ion_value=2, ele_value=1).astype(np.float32),
                    ],
                    axis=0,
                )[None, ...]
                with torch.no_grad():
                    j_cnn = float(pc(torch.from_numpy(feat).to(device)).cpu().numpy()[0, 5])
            fig = plt.figure(figsize=(12, 4))
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.bar(["phys-CNN", "phys-DNN"], [j_cnn, j_dnn], color=["#8e44ad", "#27ae60"])
            ax1.set_title("(b) Optimal J by surrogate")
            ax1.set_ylabel(r"$J$ at 120 mV")
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(zflat.reshape(32, 32), cmap="viridis")
            ax2.set_title("(c) Optimal latent (32x32)")
            ax2.axis("off")
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(vbest[vbest.shape[0] // 2], cmap="viridis")
            ax3.set_title("(d) Optimal structure slice")
            ax3.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / "figure6_bcd_optimal.png", dpi=220)
            produced_png.append("figure6_bcd_optimal.png")
            plt.close(fig)

    # Figure 2c,d 风格：Wasserstein 距离收敛
    if Path(args.gan_metrics).exists():
        m = np.genfromtxt(args.gan_metrics, delimiter=",", names=True)
        plt.figure(figsize=(8, 4))
        plt.plot(m["step"], m["w_dist"], lw=1.2, color="#5d3a9b")
        plt.xlabel("Training step")
        plt.ylabel("Wasserstein distance")
        plt.title("Figure 2c,d style: convergence by Wasserstein distance")
        plt.tight_layout()
        plt.savefig(out_dir / "figure2_cd_wasserstein.png", dpi=220)
        produced_png.append("figure2_cd_wasserstein.png")
        plt.close()

    # Figure 7a/b/c 风格：J-eta 对比 + 相分数 + active/all TPB ratio
    # 选取 low/inter/global 三个代表
    low_idx = int(np.argmin(jv))
    mid_idx = int(np.argsort(jv)[len(jv) // 2])
    glo_idx = int(np.argmax(jv))
    sel = [("Low", low_idx), ("Intermediate", mid_idx), ("Global optimum", glo_idx)]

    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(2, 2, 1)
    for name, idx in sel:
        ax1.plot(eta_mv, fake_j[idx], marker="o", lw=1.5, label=name)
    ax1.set_title("(a) J-eta curves")
    ax1.set_xlabel(r"$\eta$ (mV)")
    ax1.set_ylabel(r"$J$ (mA cm$^{-2}$)")
    ax1.legend(frameon=False, fontsize=8)

    ax2 = fig.add_subplot(2, 2, 2)
    def _pf(vol: np.ndarray) -> np.ndarray:
        n = float(vol.size)
        return np.array([(vol == 0).sum() / n, (vol == 128).sum() / n, (vol == 255).sum() / n], dtype=np.float32)
    x = np.arange(3)
    labels = ["Pore", "Ni", "YSZ"]
    wbar = 0.22
    vals = np.stack([_pf(fake_vols[idx]) for _, idx in sel], axis=0)
    for k, (name, _) in enumerate(sel):
        ax2.bar(x + (k - 1) * wbar, vals[k], width=wbar, label=name)
    ax2.set_xticks(x, labels)
    ax2.set_ylim(0, 0.8)
    ax2.set_title("(b) Phase fraction")
    ax2.legend(frameon=False, fontsize=8)

    ax3 = fig.add_subplot(2, 1, 2)
    ratios = []
    for _, idx in sel:
        a = strict_tpb(fake_vols[idx])
        t = total_tpb(fake_vols[idx])
        ratios.append(a / max(t, 1e-8))
    ax3.bar(["Low", "Intermediate", "Global"], ratios, color=["#95a5a6", "#f1948a", "#85c1e9"])
    ax3.set_ylim(0, 1.0)
    ax3.set_title("(c) Active / All TPB ratio")
    ax3.set_ylabel("ratio")
    fig.tight_layout()
    fig.savefig(out_dir / "figure7_abc_performance_insight.png", dpi=220)
    produced_png.append("figure7_abc_performance_insight.png")
    plt.close(fig)

    # 若提供 OpenFOAM 场导出，生成论文同构的 d/e/f（优先级高于代理版）
    if args.openfoam_dir:
        ofroot = Path(args.openfoam_dir)
        req = [
            ofroot / "phase_low.npy",
            ofroot / "phase_intermediate.npy",
            ofroot / "phase_global.npy",
            ofroot / "phi_ion_low.npy",
            ofroot / "phi_ion_intermediate.npy",
            ofroot / "phi_ion_global.npy",
        ]
        if all(p.exists() for p in req):
            from src.physics.tpb_logic import active_union_mask_from_label_volume
            groups_of = [("low", "Low"), ("intermediate", "Intermediate"), ("global", "Global optimum")]
            fig = plt.figure(figsize=(13, 7))
            for i, (k, label_txt) in enumerate(groups_of, start=1):
                phase = np.load(ofroot / f"phase_{k}.npy").astype(np.float32)
                phi = np.load(ofroot / f"phi_ion_{k}.npy").astype(np.float32)
                lab = np.zeros_like(phase, dtype=np.int8)
                lab[phase == 0] = 0
                lab[phase == 128] = 1
                lab[phase == 255] = 2
                active = active_union_mask_from_label_volume(lab, pore_value=0, ion_value=2, ele_value=1)
                iso = np.sum((~active).astype(np.float32), axis=0)

                zmid = phi.shape[0] // 2
                ax1 = fig.add_subplot(2, 3, i)
                ax1.imshow(phi[zmid], cmap="plasma")
                ax1.set_title(f"(d/e) {label_txt} $\\phi_{{ion}}$")
                ax1.axis("off")
                ax2 = fig.add_subplot(2, 3, i + 3)
                ax2.imshow(iso, cmap="magma")
                ax2.set_title(f"(f) {label_txt} isolated density")
                ax2.axis("off")
            fig.tight_layout()
            fig.savefig(out_dir / "figure7_def_openfoam.png", dpi=220)
            produced_png.append("figure7_def_openfoam.png")
            plt.close(fig)
    elif args.strict_no_proxy:
        raise ValueError("严格无代理模式下必须提供 --openfoam-dir 以生成 figure7_def_openfoam.png")

    if args.strict_no_proxy:
        still_proxy = list(out_dir.glob("*proxy*"))
        if still_proxy:
            raise RuntimeError(f"strict-no-proxy 模式下仍检测到代理产物: {[str(x) for x in still_proxy]}")

    write_figure_bundle_manifest(
        out_dir,
        produced_png=produced_png,
        strict_no_proxy=bool(args.strict_no_proxy),
        openfoam_dir=(args.openfoam_dir or "").strip(),
    )

    print(f"论文风格图已输出到: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

