"""
从体素体（0/128/255）导出插图 PNG。

默认 matplotlib：版式与 scripts/evaluate_plot_fallback.py 中 2×3 图一致（上三表、下三体素块）；
可选 PyVista 做光滑等值面（需 mesa/xvfb 等）。
与训练用 npy 解耦：仅额外生成 PNG（及 manifest），不改变原始体素数据。
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _reconfigure_stdio_utf8() -> None:
    """Windows 终端常见 GBK，含中文的 --help 会 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        if stream is None:
            continue
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8", errors="replace")
            except (OSError, ValueError, AttributeError):
                pass


_reconfigure_stdio_utf8()

import numpy as np
import torch
import yaml
from scipy.stats import mode as scipy_mode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.generator3d import Generator3D  # noqa: E402
from src.models.phys_cnn import PhysCNN  # noqa: E402
from src.models.phys_dnn import PhysDNN  # noqa: E402
from src.physics.tpb_logic import (  # noqa: E402
    active_tpb_density_from_label_volume,
    active_union_mask_from_label_volume,
)
from src.utils.output_manifest import PAPER_ETA_MV_MILLIVOLT, output_spec_block, write_json  # noqa: E402
from src.viz.microstructure_surface import (  # noqa: E402
    DEFAULT_MASK_BLUR_SIGMA_VOXELS,
    DEFAULT_MESH_RELAXATION,
    DEFAULT_SMOOTH_ITERATIONS,
)


def _default_surface_backend() -> str:
    """允许用 PI_SURFACE_BACKEND 覆盖默认；集群无头节点可 export PI_SURFACE_BACKEND=matplotlib。"""
    v = (os.environ.get("PI_SURFACE_BACKEND") or "").strip().lower()
    return v if v in ("matplotlib", "pyvista") else "matplotlib"


def _linux_without_display() -> bool:
    return sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or "").strip()


def _refuse_pyvista_on_headless_linux() -> None:
    """避免 VTK 在无 X/EGL/OSMesa 上初始化窗口后直接段错误。"""
    if os.environ.get("PI_ALLOW_HEADLESS_PYVISTA", "").strip() == "1":
        return
    raise SystemExit(
        "当前为 Linux 且未设置 DISPLAY，使用 --surface-backend pyvista 时 VTK 会依次尝试 "
        "X / EGL / OSMesa；未装 mesa 等库时常在警告后出现 Segmentation fault。\n\n"
        "请改用其一：\n"
        "  • 命令行: --surface-backend matplotlib（推荐集群无头节点）\n"
        "  • 环境变量: export PI_SURFACE_BACKEND=matplotlib\n"
        "  • 若已配置 OSMesa/EGL 且确信可用: export PI_ALLOW_HEADLESS_PYVISTA=1\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="体素三相微结构 → 表面重建 → 渲染 PNG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
默认 --surface-backend matplotlib（无 OpenGL/OSMesa）：输出版式与 scripts/evaluate_plot_fallback.py
中 2×3 图一致（上三表、下三体素块，体素色 #1f1b52 / #20a39e / #e5df3f）。
环境变量 PI_SURFACE_BACKEND=matplotlib|pyvista 可覆盖该参数的默认值（未识别则仍为 matplotlib）。
若要用光滑等值面：--surface-backend pyvista，并安装 mesalib 或使用 xvfb-run。
Linux 无 DISPLAY 时使用 pyvista 会在启动阶段被拒绝（除非 PI_ALLOW_HEADLESS_PYVISTA=1），避免段错误。
""",
    )
    p.add_argument("--volume-npy", default="", help="单个体素 .npy，形状 [D,H,W] 或 [1,D,H,W]")
    p.add_argument("--data-dir", default="", help="含 volumes.npy 的目录，与 --sample-index 联用")
    p.add_argument(
        "--best-latent-npy",
        default="",
        help="PSO 输出的 best_latent.npy；与 --gen-ckpt（及 --gan-config）联用，经 GAN 生成体素再渲染",
    )
    p.add_argument("--gen-ckpt", default="", help="与 --best-latent-npy 联用：generator_fallback.pth 等")
    p.add_argument("--gan-config", default="configs/paper_params.yaml", help="读取 gan 结构（与训练一致）")
    p.add_argument(
        "--gan-force-tiny",
        action="store_true",
        help="与 --best-latent-npy 联用：强制按 tiny_channels 建 Generator（ckpt 为试跑/小网时常用；覆盖 yaml 里 debug_tiny=false）",
    )
    p.add_argument(
        "--gan-force-paper",
        action="store_true",
        help="与 --best-latent-npy 联用：强制按 paper_channels 建 Generator（覆盖 yaml）",
    )
    p.add_argument("--device", default="cuda", help="best-latent 前向所用设备")
    p.add_argument("--sample-index", type=int, default=0, help="data-dir 模式下取第几个样本")
    p.add_argument("--out-dir", default="outputs/surface_figures", help="输出目录")
    p.add_argument("--basename", default="microstructure_surface", help="输出文件名前缀")
    p.add_argument(
        "--voxel-size-nm",
        type=float,
        default=1.0,
        help="各向同性体素边长 (nm)，用于几何缩放；未知时保持 1 表示体素单位",
    )
    p.add_argument(
        "--smooth-iters",
        type=int,
        default=DEFAULT_SMOOTH_ITERATIONS,
        help="网格拉普拉斯平滑迭代数（仅显示用；默认较温和，可改小/改 0 更贴原始台阶）",
    )
    p.add_argument(
        "--mesh-relaxation",
        type=float,
        default=DEFAULT_MESH_RELAXATION,
        help="平滑松弛系数，越小越不易收缩薄结构（仅显示网格）",
    )
    p.add_argument(
        "--mask-blur-sigma",
        type=float,
        default=DEFAULT_MASK_BLUR_SIGMA_VOXELS,
        help="等值面前对二值掩膜的高斯 σ（体素单位）；0 关闭，略大于 0 可减轻台阶突变",
    )
    p.add_argument(
        "--views",
        choices=["iso", "both"],
        default="both",
        help="iso 仅一张等轴视角；both 另存三视角（matplotlib 为三子图；pyvista 为 1×3 正交）",
    )
    p.add_argument(
        "--surface-backend",
        choices=["matplotlib", "pyvista"],
        default=_default_surface_backend(),
        help="matplotlib=不依赖 OSMesa（体素块示意）；pyvista=光滑等值面，需 mesa/xvfb；"
        "默认可由环境变量 PI_SURFACE_BACKEND 覆盖",
    )
    p.add_argument(
        "--mpl-downsample",
        type=int,
        default=2,
        help="matplotlib 后端：S³ 众数降采样（与 evaluate_plot_fallback --cube-voxel-step 一致）；64³ 建议 2 或 4",
    )
    p.add_argument(
        "--phys-dnn-ckpt",
        default="outputs/phys_models/phys_dnn.pth",
        help="已学 Phys-DNN：TPB→J(η)，用于 (b) 学习曲线",
    )
    p.add_argument(
        "--phys-cnn-ckpt",
        default="outputs/phys_models/phys_cnn.pth",
        help="已学 Phys-CNN：体素→J(η)，与 PSO forward design 一致",
    )
    p.add_argument(
        "--reference-data-dir",
        default="data/processed_fallback",
        help="含 tpb.npy、labels_j.npy 的训练集目录，用于 (b) 散点背景",
    )
    p.add_argument(
        "--pso-history",
        default="",
        help="PSO 历史 CSV（iter,best_j,mean_j）；与 best_latent 联用时可对照优化目标 J@120 mV",
    )
    p.add_argument(
        "--j-index",
        type=int,
        default=5,
        help="性能对照用的 η 列下标（默认 5 → 120 mV，与 forward_design manifest 一致）",
    )
    return p.parse_args()


def _torch_load_state_dict(path: Path, map_location: torch.device) -> dict:
    """加载纯 state_dict；新 PyTorch 用 weights_only=True 消除 FutureWarning。"""
    try:
        return torch.load(str(path), map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def _load_volume(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"单样本体素期望 [D,H,W] 或 [1,D,H,W]，当前 {arr.shape}")
    return arr.astype(np.float32)


def _volume_from_best_latent(
    best_latent_path: Path,
    gen_ckpt: Path,
    gan_config_path: Path,
    device: torch.device,
    *,
    force_tiny: bool,
    force_paper: bool,
) -> np.ndarray:
    """best_latent.npy + Generator3D → [D,H,W] 0/128/255"""
    if force_tiny and force_paper:
        raise ValueError("不能同时指定 --gan-force-tiny 与 --gan-force-paper。")
    zflat = np.load(best_latent_path).astype(np.float32)
    if zflat.ndim == 2:
        zflat = zflat[0]
    if zflat.ndim != 1:
        raise ValueError(f"best_latent 期望一维向量，当前 shape={zflat.shape}")

    with open(gan_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    gan = cfg["gan"]
    if force_tiny:
        use_tiny = True
    elif force_paper:
        use_tiny = False
    else:
        use_tiny = bool(gan.get("debug_tiny", True))
    g_channels = gan["tiny_channels"]["generator"] if use_tiny else gan["paper_channels"]["generator"]
    latent_dim = int(gan["latent_channels"] * 4 * 4 * 4)
    if zflat.size != latent_dim:
        raise ValueError(f"潜向量长度 {zflat.size} 与配置不符（期望 {latent_dim}）")

    gen = Generator3D(
        z_channels=gan["latent_channels"],
        num_classes=gan["classes_inverse"],
        embed_dim=gan["embedding_size"],
        channels=g_channels,
    ).to(device)
    gen.load_state_dict(_torch_load_state_dict(gen_ckpt, device))
    gen.eval()
    z = torch.from_numpy(zflat).to(device).view(1, gan["latent_channels"], 4, 4, 4)
    labels = torch.zeros(1, dtype=torch.long, device=device)
    with torch.no_grad():
        x3 = gen(z, labels)
    hard = torch.argmax(x3[0], dim=0).detach().cpu().numpy().astype(np.int8)
    out = np.zeros_like(hard, dtype=np.float32)
    out[hard == 1] = 128.0
    out[hard == 2] = 255.0
    return out


def _downsample_phase_majority(vol: np.ndarray, step: int = 4) -> np.ndarray:
    """与 scripts/evaluate_plot_fallback.py 一致：S³ 块内三相灰度众数。"""
    d, h, w = vol.shape
    d2, h2, w2 = d // step, h // step, w // step
    if d2 < 1 or h2 < 1 or w2 < 1:
        return vol.astype(np.float32)
    trimmed = vol[: d2 * step, : h2 * step, : w2 * step]
    blocks = trimmed.reshape(d2, step, h2, step, w2, step)
    blocks = blocks.transpose(0, 2, 4, 1, 3, 5).reshape(d2, h2, w2, step**3)
    m = scipy_mode(blocks, axis=-1, keepdims=False)
    return np.asarray(m.mode, dtype=np.float32).reshape(d2, h2, w2)


def _two_point_corr(mask: np.ndarray, max_d: int = 30) -> np.ndarray:
    """与 evaluate_plot_fallback.two_point_corr 一致。"""
    vals = []
    for d in range(1, max_d + 1):
        c = (mask[:, :, :-d] & mask[:, :, d:]).mean()
        vals.append(float(c))
    arr = np.asarray(vals, dtype=np.float32)
    if arr.size > 0 and arr[0] > 0:
        arr = arr / arr[0]
    return arr


def _load_phys_dnn(ckpt_path: Path, device: torch.device) -> PhysDNN | None:
    if not ckpt_path.is_file():
        return None
    m = PhysDNN(input_size=1, hidden_dim=50, output_size=7).to(device)
    m.load_state_dict(_torch_load_state_dict(ckpt_path, device))
    m.eval()
    return m


def _infer_phys_cnn_in_channels(state_dict: dict) -> int:
    w = state_dict.get("features.0.weight")
    if w is None:
        return 4
    return int(w.shape[1])


def _load_phys_cnn(ckpt_path: Path, device: torch.device) -> tuple[PhysCNN | None, int]:
    if not ckpt_path.is_file():
        return None, 4
    sd = _torch_load_state_dict(ckpt_path, device)
    in_ch = _infer_phys_cnn_in_channels(sd)
    m = PhysCNN(in_channels=in_ch, out_dim=7).to(device)
    m.load_state_dict(sd)
    m.eval()
    return m, in_ch


def _predict_j_from_tpb(tpb_scalar: float, ckpt_path: Path, device: torch.device) -> np.ndarray | None:
    """Phys-DNN: active TPB → 7 点 J(η)。"""
    m = _load_phys_dnn(ckpt_path, device)
    if m is None:
        return None
    x = torch.tensor([[float(tpb_scalar)]], dtype=torch.float32, device=device)
    with torch.no_grad():
        return m(x).cpu().numpy()[0].astype(np.float64)


def _predict_j_curve_phys_dnn_on_grid(
    phys: PhysDNN,
    tpb_lo: float,
    tpb_hi: float,
    device: torch.device,
    *,
    n: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(float(tpb_lo), float(tpb_hi), int(n), dtype=np.float32)
    with torch.no_grad():
        j = phys(torch.from_numpy(grid[:, None]).to(device)).cpu().numpy()
    return grid, j.astype(np.float64)


def _volume_to_phys_cnn_feat(vol: np.ndarray, in_channels: int = 4) -> np.ndarray:
    v = np.asarray(vol, dtype=np.float32)
    label = np.zeros_like(v, dtype=np.int8)
    label[v == 0] = 0
    label[v == 128] = 1
    label[v == 255] = 2
    union = active_union_mask_from_label_volume(
        label, pore_value=0, ion_value=2, ele_value=1, min_connected_fraction=0.01
    ).astype(np.float32)
    if in_channels == 1:
        feat = union[None, ...]
    else:
        feat = np.stack(
            [
                (v == 0).astype(np.float32),
                (v == 128).astype(np.float32),
                (v == 255).astype(np.float32),
                union,
            ],
            axis=0,
        )
    return feat[None, ...]


def _predict_j_from_volume_phys_cnn(vol: np.ndarray, ckpt_path: Path, device: torch.device) -> np.ndarray | None:
    m, in_ch = _load_phys_cnn(ckpt_path, device)
    if m is None:
        return None
    feat = _volume_to_phys_cnn_feat(vol, in_channels=in_ch)
    with torch.no_grad():
        return m(torch.from_numpy(feat).to(device)).cpu().numpy()[0].astype(np.float64)


def _load_train_tpb_j(ref_dir: Path, j_index: int) -> tuple[np.ndarray, np.ndarray] | None:
    tpb_p = ref_dir / "tpb.npy"
    lab_p = ref_dir / "labels_j.npy"
    if not tpb_p.is_file() or not lab_p.is_file():
        return None
    tpb = np.load(tpb_p).astype(np.float64).reshape(-1)
    lab = np.load(lab_p).astype(np.float64)
    if lab.ndim != 2 or lab.shape[1] <= j_index:
        return None
    j = lab[:, j_index]
    m = np.isfinite(tpb) & np.isfinite(j)
    return tpb[m], j[m]


def _plot_learned_tpb_j(ax, *, tpb_train, j_train, tpb_grid, j_grid, tpb_now, j_dnn_now, j_cnn_now, j_index: int) -> None:
    eta = list(PAPER_ETA_MV_MILLIVOLT)
    eta_lbl = eta[j_index] if 0 <= j_index < len(eta) else j_index
    if tpb_train is not None and j_train is not None and tpb_train.size > 0:
        n = int(tpb_train.size)
        if n > 2500:
            rng = np.random.default_rng(0)
            idx = rng.choice(n, size=2500, replace=False)
            xt, yt = tpb_train[idx], j_train[idx]
        else:
            xt, yt = tpb_train, j_train
        ax.scatter(xt, yt, s=6, alpha=0.35, color="#acb0b3", label="train (tpb, J)", rasterized=True)
    if tpb_grid is not None and j_grid is not None and tpb_grid.size > 1:
        ax.plot(
            tpb_grid,
            j_grid[:, j_index],
            color="#6fa23a",
            lw=2.0,
            label=f"Phys-DNN J@{eta_lbl}mV",
        )
    if tpb_now is not None and j_dnn_now is not None:
        ax.scatter(
            [tpb_now],
            [j_dnn_now],
            s=90,
            c="#27ae60",
            marker="*",
            zorder=5,
            label="this / Phys-DNN",
        )
    if tpb_now is not None and j_cnn_now is not None:
        ax.scatter(
            [tpb_now],
            [j_cnn_now],
            s=70,
            c="#8e44ad",
            marker="D",
            zorder=5,
            label="this / Phys-CNN",
        )
    ax.set_xlabel("Active TPB (proxy)")
    ax.set_ylabel(f"J @ {eta_lbl} mV (mA cm$^{{-2}}$)")
    ax.legend(fontsize=7, frameon=False, loc="best")


def _plot_learned_j_eta(
    ax,
    *,
    j_dnn: np.ndarray | None,
    j_cnn: np.ndarray | None,
    pso_best_j: float | None,
    j_index: int,
) -> None:
    eta = np.asarray(PAPER_ETA_MV_MILLIVOLT, dtype=np.float64)
    if j_dnn is not None and j_dnn.size == eta.size:
        ax.plot(eta, j_dnn, "o-", color="#6fa23a", lw=1.6, ms=5, label="Phys-DNN(TPB)")
    if j_cnn is not None and j_cnn.size == eta.size:
        ax.plot(eta, j_cnn, "s--", color="#8e44ad", lw=1.4, ms=5, label="Phys-CNN(voxel)")
    if pso_best_j is not None and 0 <= j_index < eta.size:
        ax.scatter(
            [eta[j_index]],
            [pso_best_j],
            c="#2f5aa8",
            s=80,
            marker="*",
            zorder=5,
            label="PSO best",
        )
    ax.set_xlabel(r"$\eta$ (mV)")
    ax.set_ylabel(r"$J$ (mA cm$^{-2}$)")
    ax.legend(fontsize=7, frameon=False, loc="best")


def _read_pso_best_j(csv_path: Path) -> float | None:
    if not csv_path.is_file():
        return None
    rows = np.genfromtxt(csv_path, delimiter=",", names=True)
    if rows.size == 0:
        return None
    if rows.ndim == 0:
        return float(rows["best_j"])
    return float(rows["best_j"][-1])


def _strict_tpb_scalar(vol_0128_255: np.ndarray) -> float:
    label = np.zeros_like(vol_0128_255, dtype=np.int8)
    label[vol_0128_255 == 0] = 0
    label[vol_0128_255 == 128] = 1
    label[vol_0128_255 == 255] = 2
    return float(
        active_tpb_density_from_label_volume(
            label,
            pore_value=0,
            ion_value=2,
            ele_value=1,
            min_connected_fraction=0.01,
        )
    )


def _plot_cube_fallback_style(ax, vol: np.ndarray, title: str, *, elev: float = 22.0, azim: float = 45.0) -> None:
    """与 evaluate_plot_fallback._plot_cube 一致：三相 hex + voxels + edgecolor none。"""
    down = np.asarray(vol, dtype=np.float32)
    filled = np.ones_like(down, dtype=bool)
    colors = np.empty(down.shape, dtype=object)
    colors[down == 0] = "#1f1b52"  # Pore
    colors[down == 128] = "#20a39e"  # Ni
    colors[down == 255] = "#e5df3f"  # YSZ
    ax.voxels(filled, facecolors=colors, edgecolor="none")
    ax.set_title(title, pad=6)
    ax.set_axis_off()
    ax.view_init(elev=float(elev), azim=float(azim))
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def _load_from_data_dir(data_dir: Path, index: int) -> np.ndarray:
    vol_path = data_dir / "volumes.npy"
    if not vol_path.exists():
        raise FileNotFoundError(f"缺少 {vol_path}")
    stack = np.load(vol_path).astype(np.float32)
    if stack.ndim != 5 or stack.shape[1] != 1:
        raise ValueError(f"volumes.npy 期望 [N,1,D,H,W]，当前 {stack.shape}")
    if index < 0 or index >= stack.shape[0]:
        raise IndexError(f"sample-index 越界: {index} 不在 [0, {stack.shape[0]})")
    return stack[index, 0]


def _export_matplotlib_voxel_views(
    vol: np.ndarray,
    out_dir: Path,
    *,
    basename: str,
    views: str,
    downsample: int,
    performance_ctx: dict | None = None,
) -> dict[str, str]:
    """
    无 VTK：版式与 scripts/evaluate_plot_fallback.py 中 Fig.5 风格四联图一致——
    GridSpec(2,3)：上行三表 (a)(b)(c)，下行 (d) 三个体素立方体；体素色与 _plot_cube 一致。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    vol_f = np.asarray(vol, dtype=np.float32)
    step = max(1, int(downsample))
    vol_ds = _downsample_phase_majority(vol_f, step=step) if step > 1 else vol_f.copy()
    d, h, w = vol_ds.shape
    npx = int(vol_f.size)
    fp = float((vol_f == 0).sum()) / npx
    fni = float((vol_f == 128).sum()) / npx
    fysz = float((vol_f == 255).sum()) / npx
    tpb = _strict_tpb_scalar(vol_f)
    perf = performance_ctx or {}
    learned = perf.get("learned") or {}
    j_index = int(perf.get("j_index", 5))
    j_dnn = np.asarray(learned.get("j_dnn_curve"), dtype=np.float64) if learned.get("j_dnn_curve") is not None else None
    j_cnn = np.asarray(learned.get("j_cnn_curve"), dtype=np.float64) if learned.get("j_cnn_curve") is not None else None
    pso_best_j = learned.get("pso_best_j")
    tpb_train = learned.get("tpb_train")
    j_train = learned.get("j_train")
    tpb_grid = learned.get("tpb_grid")
    j_grid = learned.get("j_grid")
    if tpb_train is not None:
        tpb_train = np.asarray(tpb_train, dtype=np.float64)
    if j_train is not None:
        j_train = np.asarray(j_train, dtype=np.float64)
    if tpb_grid is not None:
        tpb_grid = np.asarray(tpb_grid, dtype=np.float64)
    if j_grid is not None:
        j_grid = np.asarray(j_grid, dtype=np.float64)
    j_dnn_now = float(j_dnn[j_index]) if j_dnn is not None and 0 <= j_index < j_dnn.size else None
    j_cnn_now = float(j_cnn[j_index]) if j_cnn is not None and 0 <= j_index < j_cnn.size else None

    dist_show = [1, 5, 10, 20]
    max_d = max(30, max(dist_show))
    tpc_rows: list[list[str]] = []
    for di in dist_show:
        row: list[str] = [str(di)]
        for pval in (0, 128, 255):
            mask = vol_ds == float(pval)
            arr = _two_point_corr(mask, max_d=max_d)
            row.append(f"{float(arr[di - 1]):.4f}")
        tpc_rows.append(row)

    view_triples = [
        (22.0, 45.0, "(d) View 1"),
        (22.0, 135.0, "(d) View 2"),
        (22.0, -45.0, "(d) View 3"),
    ]

    def _draw_paper_panel(save_path: Path) -> None:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(
            2,
            3,
            figure=fig,
            width_ratios=[1.1, 1.1, 1.2],
            height_ratios=[1.0, 1.0],
            wspace=0.28,
            hspace=0.25,
        )

        ax_a = fig.add_subplot(gs[0, 0])
        ax_a.axis("off")
        ax_a.set_title("(a) Phase volume fractions", fontsize=11, pad=8)
        rows_a = [["Pore", f"{100.0 * fp:.2f}%"], ["Ni", f"{100.0 * fni:.2f}%"], ["YSZ", f"{100.0 * fysz:.2f}%"]]
        ta = ax_a.table(cellText=rows_a, colLabels=["Phase", "Vol."], loc="center", cellLoc="center")
        ta.auto_set_font_size(False)
        ta.set_fontsize(9)
        ta.scale(1.05, 2.0)

        gs_b = gs[0, 1].subgridspec(2, 1, hspace=0.32)
        ax_b1 = fig.add_subplot(gs_b[0, 0])
        ax_b2 = fig.add_subplot(gs_b[1, 0])
        _plot_learned_tpb_j(
            ax_b1,
            tpb_train=tpb_train,
            j_train=j_train,
            tpb_grid=tpb_grid,
            j_grid=j_grid,
            tpb_now=float(tpb),
            j_dnn_now=j_dnn_now,
            j_cnn_now=j_cnn_now,
            j_index=j_index,
        )
        ax_b1.set_title("(b) Learned TPB → J @ η", fontsize=10, pad=4)
        _plot_learned_j_eta(
            ax_b2,
            j_dnn=j_dnn,
            j_cnn=j_cnn,
            pso_best_j=float(pso_best_j) if pso_best_j is not None else None,
            j_index=j_index,
        )
        ax_b2.set_title("Learned J–η (Phys-DNN / Phys-CNN)", fontsize=10, pad=4)

        ax_c = fig.add_subplot(gs[0, 2])
        ax_c.axis("off")
        ax_c.set_title("(c) Two-point correlation C(d)", fontsize=11, pad=8)
        hdr = ["d", "Pore", "Ni", "YSZ"]
        tc = ax_c.table(cellText=tpc_rows, colLabels=hdr, loc="center", cellLoc="center")
        tc.auto_set_font_size(False)
        tc.set_fontsize(8)
        tc.scale(1.02, 1.7)

        gs_d = gs[1, :].subgridspec(1, 3, wspace=0.02)
        for i, (elev, azim, ttl) in enumerate(view_triples):
            ax_d = fig.add_subplot(gs_d[0, i], projection="3d")
            _plot_cube_fallback_style(ax_d, vol_ds, ttl, elev=elev, azim=azim)

        fig.suptitle(
            "2x3: phase fractions | π-learning surrogates (train + learned curves) | two-point | voxels",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.02, 1, 0.94))
        fig.savefig(str(save_path), dpi=220)
        plt.close(fig)

    iso_path = out_dir / f"{basename}_iso.png"
    _draw_paper_panel(iso_path)
    paths["iso"] = str(iso_path.resolve())

    if views == "both":
        mv_path = out_dir / f"{basename}_multiview.png"
        fig2 = plt.figure(figsize=(14.0, 4.2))
        for i, (elev, azim, ttl) in enumerate(view_triples):
            ax = fig2.add_subplot(1, 3, i + 1, projection="3d")
            _plot_cube_fallback_style(ax, vol_ds, ttl, elev=elev, azim=azim)
        fig2.tight_layout()
        fig2.savefig(str(mv_path), dpi=220)
        plt.close(fig2)
        paths["multiview"] = str(mv_path.resolve())

    return paths


def main() -> int:
    args = parse_args()
    if args.surface_backend == "pyvista" and _linux_without_display():
        _refuse_pyvista_on_headless_linux()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_src = sum(
        1
        for x in (args.volume_npy.strip(), args.data_dir.strip(), args.best_latent_npy.strip())
        if x
    )
    if n_src != 1:
        raise SystemExit(
            "请三选一且仅选一：--volume-npy、--data-dir，或 --best-latent-npy（与 --gen-ckpt 联用）。"
        )

    if args.best_latent_npy.strip():
        if not args.gen_ckpt.strip():
            raise SystemExit("使用 --best-latent-npy 时必须提供 --gen-ckpt。")
        vol = _volume_from_best_latent(
            Path(args.best_latent_npy),
            Path(args.gen_ckpt),
            Path(args.gan_config),
            torch.device(args.device),
            force_tiny=bool(args.gan_force_tiny),
            force_paper=bool(args.gan_force_paper),
        )
        provenance = f"best_latent={args.best_latent_npy}, gen_ckpt={args.gen_ckpt}"
        if args.gan_force_tiny:
            provenance += ", gan_arch=force_tiny"
        elif args.gan_force_paper:
            provenance += ", gan_arch=force_paper"
    elif args.volume_npy.strip():
        vol = _load_volume(Path(args.volume_npy))
        provenance = f"单文件: {args.volume_npy}"
    else:
        vol = _load_from_data_dir(Path(args.data_dir), args.sample_index)
        provenance = f"data-dir={args.data_dir}, sample_index={args.sample_index}"

    dev = torch.device(args.device)
    j_index = int(args.j_index)
    tpb_scalar = _strict_tpb_scalar(vol)
    performance_ctx: dict = {
        "active_tpb_proxy": float(tpb_scalar),
        "j_index": j_index,
        "learned": {},
    }
    learned: dict = performance_ctx["learned"]

    phys_dnn_ckpt = Path(args.phys_dnn_ckpt)
    phys_cnn_ckpt = Path(args.phys_cnn_ckpt)
    j_dnn = _predict_j_from_tpb(tpb_scalar, phys_dnn_ckpt, dev)
    if j_dnn is not None:
        learned["j_dnn_curve"] = j_dnn.tolist()
        if 0 <= j_index < j_dnn.size:
            learned["j_dnn_at_eta_index"] = float(j_dnn[j_index])

    j_cnn = _predict_j_from_volume_phys_cnn(vol, phys_cnn_ckpt, dev)
    if j_cnn is not None:
        learned["j_cnn_curve"] = j_cnn.tolist()
        if 0 <= j_index < j_cnn.size:
            learned["j_cnn_at_eta_index"] = float(j_cnn[j_index])

    ref_dir = Path(args.reference_data_dir.strip()) if args.reference_data_dir.strip() else None
    if ref_dir is None and args.data_dir.strip():
        ref_dir = Path(args.data_dir)
    if ref_dir is not None and ref_dir.is_dir():
        learned["reference_data_dir"] = str(ref_dir.resolve())
        pair = _load_train_tpb_j(ref_dir, j_index)
        phys_dnn = _load_phys_dnn(phys_dnn_ckpt, dev)
        if pair is not None:
            tpb_tr, j_tr = pair
            learned["tpb_train"] = tpb_tr.tolist()
            learned["j_train"] = j_tr.tolist()
            if phys_dnn is not None and tpb_tr.size > 1:
                lo, hi = np.percentile(tpb_tr, [1, 99])
                pad = 0.05 * max(float(hi - lo), 1e-8)
                grid, j_all = _predict_j_curve_phys_dnn_on_grid(
                    phys_dnn, lo - pad, hi + pad, dev, n=80
                )
                learned["tpb_grid"] = grid.tolist()
                learned["j_grid"] = j_all.tolist()

    pso_csv = args.pso_history.strip()
    if not pso_csv and args.best_latent_npy.strip():
        pso_csv = str(Path(args.best_latent_npy).resolve().parent / "pso_history.csv")
    if pso_csv:
        best_j = _read_pso_best_j(Path(pso_csv))
        if best_j is not None:
            learned["pso_best_j"] = float(best_j)
            learned["pso_history"] = str(Path(pso_csv).resolve())

    if args.surface_backend == "matplotlib":
        paths = _export_matplotlib_voxel_views(
            vol,
            out_dir,
            basename=args.basename,
            views=args.views,
            downsample=int(args.mpl_downsample),
            performance_ctx=performance_ctx,
        )
    else:
        # 延迟导入：matplotlib 路径不加载 PyVista/VTK，避免 import 副作用在无头环境踩 GL
        from src.viz.microstructure_surface import export_surface_bundle as _export_surface_bundle

        paths = _export_surface_bundle(
            vol,
            out_dir,
            basename=args.basename,
            voxel_size_nm=float(args.voxel_size_nm),
            smooth_iterations=int(args.smooth_iters),
            mesh_relaxation=float(args.mesh_relaxation),
            mask_blur_sigma_voxels=float(args.mask_blur_sigma),
            views=args.views,
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "purpose": "表面重建与渲染仅用于插图；结构统计仍以原始体素为准",
        "surface_backend": str(args.surface_backend),
        "mpl_downsample": int(args.mpl_downsample) if args.surface_backend == "matplotlib" else None,
        "mpl_panel_layout": "evaluate_plot_fallback_2x3_with_pi_learning_surrogates"
        if args.surface_backend == "matplotlib"
        else None,
        "data_provenance": provenance,
        "voxel_size_nm_isotropic": float(args.voxel_size_nm),
        "smooth_iterations_display_only": int(args.smooth_iters),
        "mesh_relaxation_display_only": float(args.mesh_relaxation),
        "mask_blur_sigma_voxels_display_only": float(args.mask_blur_sigma),
        "views": args.views,
        "active_tpb_proxy": float(tpb_scalar),
        "pi_learning_learned": performance_ctx.get("learned"),
        "outputs": paths,
        "output_spec": output_spec_block(),
    }
    write_json(out_dir / f"{args.basename}_surface_manifest.json", manifest)
    print("=== 表面渲染完成 ===")
    for k, v in paths.items():
        print(f"  [{k}] {v}")
    print(f"  manifest: {out_dir / (args.basename + '_surface_manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
