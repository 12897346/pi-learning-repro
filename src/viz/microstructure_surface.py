"""
三相体素（0 / 128 / 255）→ Marching Cubes 等值面 → 网格可选平滑 → 离屏渲染。

说明：平滑与渲染仅用于「发表级观感」，不改变磁盘上的体素训练数据；TPB/体分数仍应以原始体素为准。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

# 与仓库其余部分一致的相显示色（RGBA 0–1）；透明度略拉近，减轻相间视觉突变
PHASE_COLORS: dict[int, tuple[float, float, float, float]] = {
    0: (0.76, 0.76, 0.80, 0.42),  # pore
    1: (0.18, 0.48, 0.32, 0.48),  # Ni
    2: (0.48, 0.28, 0.68, 0.48),  # YSZ
}

# 默认略作软化：减弱体素台阶与高光突变（仍不写入训练体素）
DEFAULT_MASK_BLUR_SIGMA_VOXELS = 0.42
DEFAULT_SMOOTH_ITERATIONS = 14
DEFAULT_MESH_RELAXATION = 0.06  # 拉普拉斯平滑松弛，过大易收缩薄结构

# Linux 无 DISPLAY 时只尝试一次虚拟帧缓冲（需系统/conda 已装 xvfb）
_xvfb_started = False


def volume_to_phase_ids(vol: np.ndarray) -> np.ndarray:
    """
    将体素体转为 0/1/2 整数标签。
    vol: [D,H,W]，取值约 0, 128, 255（float 允许微小误差）。
    """
    if vol.ndim != 3:
        raise ValueError(f"期望 [D,H,W]，当前 shape={vol.shape}")
    v = vol.astype(np.float32)
    out = np.zeros(vol.shape, dtype=np.uint8)
    out[np.abs(v - 128.0) < 1.0] = 1
    out[np.abs(v - 255.0) < 1.0] = 2
    return out


def _require_pyvista():
    global _xvfb_started
    # 无图形终端时强制离屏；否则 VTK 易走坏掉的 DISPLAY/EGL 并段错误
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    try:
        import pyvista as pv  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "表面重建与渲染需要 pyvista，请执行: pip install pyvista"
        ) from exc

    pv.OFF_SCREEN = True
    if hasattr(pv, "global_theme"):
        try:
            pv.global_theme.off_screen = True
        except Exception:
            pass

    if (
        not _xvfb_started
        and sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY", "").strip()
    ):
        _xvfb_started = True
        try:
            # PyVista 在部分版本提供；需可执行 `Xvfb`（如 yum install xorg-x11-server-Xvfb）
            if hasattr(pv, "start_xvfb"):
                pv.start_xvfb()
        except Exception:
            pass

    return pv


def _soften_mask_for_display(mask: np.ndarray, sigma_voxels: float) -> np.ndarray:
    """等值面前对二值掩膜做极小高斯软化，减轻台阶感（sigma=0 则跳过）。"""
    m = mask.astype(np.float32)
    if sigma_voxels <= 0:
        return m
    from scipy.ndimage import gaussian_filter  # noqa: WPS433

    blurred = gaussian_filter(m.astype(np.float64), sigma=float(sigma_voxels)).astype(np.float32)
    return np.clip(blurred, 0.0, 1.0)


def _binary_mask_surface(
    pv,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float],
    smooth_iterations: int,
    mesh_relaxation: float,
    mask_blur_sigma_voxels: float,
) -> Any | None:
    """对二值体做 0.5 等值面；若 mask 全空则返回 None。"""
    if not np.any(mask):
        return None
    m = _soften_mask_for_display(mask, mask_blur_sigma_voxels)
    img = pv.ImageData(dimensions=mask.shape)
    img.spacing = spacing
    img.origin = origin
    img.point_data["m"] = m.ravel(order="F")
    surf = img.contour(isosurfaces=[0.5], scalars="m", progress_bar=False)
    if surf.n_points == 0:
        return None
    if smooth_iterations > 0:
        surf = surf.smooth(
            n_iter=int(smooth_iterations),
            relaxation_factor=float(mesh_relaxation),
        )
    return surf


def build_phase_surfaces(
    volume_0128_255: np.ndarray,
    *,
    spacing_nm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin_nm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    smooth_iterations: int = DEFAULT_SMOOTH_ITERATIONS,
    mesh_relaxation: float = DEFAULT_MESH_RELAXATION,
    mask_blur_sigma_voxels: float = DEFAULT_MASK_BLUR_SIGMA_VOXELS,
    phases: tuple[int, ...] = (0, 1, 2),
) -> list[tuple[int, Any]]:
    """
    为指定相分别生成外壳等值面（每相独立二值体）。

    返回 [(phase_id, PolyData), ...]，空相跳过。
    """
    pv = _require_pyvista()
    labels = volume_to_phase_ids(volume_0128_255)
    spacing_m = tuple(s * 1e-9 for s in spacing_nm)
    origin_m = tuple(o * 1e-9 for o in origin_nm)
    meshes: list[tuple[int, Any]] = []
    for pid in phases:
        mask = (labels == pid).astype(np.float32)
        surf = _binary_mask_surface(
            pv,
            mask,
            spacing_m,
            origin_m,
            smooth_iterations,
            mesh_relaxation,
            mask_blur_sigma_voxels,
        )
        if surf is not None:
            meshes.append((pid, surf))
    return meshes


def render_surfaces_to_png(
    meshes: list[tuple[int, Any]],
    out_path: Path | str,
    *,
    window_size: tuple[int, int] = (900, 900),
    multiview: bool = False,
) -> None:
    """
    离屏渲染：默认单窗口等轴视角；multiview=True 时 1×3 正交视角（沿 x/y/z 观察）。
    """
    pv = _require_pyvista()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not meshes:
        raise ValueError("没有可渲染的网格（可能体素全为单一相）")

    if not multiview:
        plotter = pv.Plotter(off_screen=True, window_size=window_size)
        plotter.set_background("white")
        for pid, surf in meshes:
            c = PHASE_COLORS.get(pid, (0.5, 0.5, 0.5, 0.5))
            plotter.add_mesh(
                surf,
                color=c[:3],
                opacity=c[3],
                smooth_shading=True,
                ambient=0.5,
                diffuse=0.48,
                specular=0.06,
            )
        plotter.camera_position = "iso"
        plotter.show(screenshot=str(out_path), auto_close=True)
        return

    w, h = window_size
    plotter = pv.Plotter(shape=(1, 3), off_screen=True, window_size=(w * 3, h))
    plotter.set_background("white")
    view_fns = (lambda pl: pl.view_xy(), lambda pl: pl.view_xz(), lambda pl: pl.view_yz())
    for col, view_fn in enumerate(view_fns):
        plotter.subplot(0, col)
        for pid, surf in meshes:
            c = PHASE_COLORS.get(pid, (0.5, 0.5, 0.5, 0.5))
            plotter.add_mesh(
                surf,
                color=c[:3],
                opacity=c[3],
                smooth_shading=True,
                ambient=0.5,
                diffuse=0.48,
                specular=0.06,
            )
        view_fn(plotter)
    plotter.show(screenshot=str(out_path), auto_close=True)


def export_surface_bundle(
    volume_0128_255: np.ndarray,
    out_dir: Path | str,
    *,
    basename: str = "microstructure_surface",
    voxel_size_nm: float = 1.0,
    smooth_iterations: int = DEFAULT_SMOOTH_ITERATIONS,
    mesh_relaxation: float = DEFAULT_MESH_RELAXATION,
    mask_blur_sigma_voxels: float = DEFAULT_MASK_BLUR_SIGMA_VOXELS,
    views: Literal["iso", "both"] = "both",
) -> dict[str, str]:
    """
    写出 iso 与（可选）multiview 两张 PNG，并返回输出路径字典。

    voxel_size_nm: 各向同性体素边长（nm），用于物理标尺感；未知时可填 1 表示「体素单位」。
    默认启用轻度 mask 模糊 + 网格平滑 + 柔和光照，以减轻视觉突变；量化请仍用原始体素。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = (float(voxel_size_nm),) * 3
    meshes = build_phase_surfaces(
        volume_0128_255,
        spacing_nm=s,
        smooth_iterations=smooth_iterations,
        mesh_relaxation=mesh_relaxation,
        mask_blur_sigma_voxels=mask_blur_sigma_voxels,
    )
    paths: dict[str, str] = {}
    iso_path = out_dir / f"{basename}_iso.png"
    render_surfaces_to_png(meshes, iso_path, multiview=False)
    paths["iso"] = str(iso_path.resolve())
    if views == "both":
        mv_path = out_dir / f"{basename}_multiview.png"
        render_surfaces_to_png(meshes, mv_path, multiview=True)
        paths["multiview"] = str(mv_path.resolve())
    return paths
