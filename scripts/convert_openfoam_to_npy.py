from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.output_manifest import write_openfoam_export_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "将 OpenFOAM 导出场（VTK/CSV）转换为论文 Figure7 d/e/f 所需 6 个 .npy 文件："
            "phase_low/intermediate/global.npy 与 phi_ion_low/intermediate/global.npy"
        )
    )
    p.add_argument("--low", required=True, help="low 组输入文件（.csv/.vtk/.vtu/.vti）")
    p.add_argument("--intermediate", required=True, help="intermediate 组输入文件（.csv/.vtk/.vtu/.vti）")
    p.add_argument("--global-opt", required=True, help="global 组输入文件（.csv/.vtk/.vtu/.vti）")
    p.add_argument("--out-dir", default="outputs/openfoam_export", help="输出目录")
    p.add_argument("--phase-col", default="phase", help="相字段名（CSV列名或VTK数组名）")
    p.add_argument("--phi-col", default="phi_ion", help="离子势字段名（CSV列名或VTK数组名）")
    p.add_argument("--x-col", default="x", help="CSV 中 x 坐标列名")
    p.add_argument("--y-col", default="y", help="CSV 中 y 坐标列名")
    p.add_argument("--z-col", default="z", help="CSV 中 z 坐标列名")
    p.add_argument(
        "--phase-values",
        default="0,128,255",
        help="允许的 phase 值集合（逗号分隔），会映射到最近值",
    )
    return p.parse_args()


def _nearest_map(values: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    v = values.reshape(-1, 1).astype(np.float32)
    a = allowed.reshape(1, -1).astype(np.float32)
    idx = np.argmin(np.abs(v - a), axis=1)
    return allowed[idx].reshape(values.shape).astype(np.float32)


def _infer_grid_from_xyz(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    phase: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # 使用唯一坐标重建规则网格；按 (x,y,z) 排序写回 3D 数组
    ux = np.unique(x)
    uy = np.unique(y)
    uz = np.unique(z)
    nx, ny, nz = len(ux), len(uy), len(uz)
    n = x.size
    if nx * ny * nz != n:
        raise ValueError(
            f"坐标点不是完整规则网格：nx*ny*nz={nx*ny*nz} != n={n}。"
            "请先在 ParaView/OpenFOAM 里导出规则体网格后再转换。"
        )

    ix = np.searchsorted(ux, x)
    iy = np.searchsorted(uy, y)
    iz = np.searchsorted(uz, z)

    vol_phase = np.empty((nx, ny, nz), dtype=np.float32)
    vol_phi = np.empty((nx, ny, nz), dtype=np.float32)
    vol_phase[ix, iy, iz] = phase.astype(np.float32)
    vol_phi[ix, iy, iz] = phi.astype(np.float32)
    return vol_phase, vol_phi


def _load_csv_volume(
    path: Path,
    phase_col: str,
    phi_col: str,
    x_col: str,
    y_col: str,
    z_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 无表头: {path}")
        required = [x_col, y_col, z_col, phase_col, phi_col]
        miss = [c for c in required if c not in reader.fieldnames]
        if miss:
            raise ValueError(f"CSV 缺少列 {miss}，可用列: {reader.fieldnames}")

        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        phase: list[float] = []
        phi: list[float] = []
        for row in reader:
            xs.append(float(row[x_col]))
            ys.append(float(row[y_col]))
            zs.append(float(row[z_col]))
            phase.append(float(row[phase_col]))
            phi.append(float(row[phi_col]))

    return _infer_grid_from_xyz(
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(zs, dtype=np.float64),
        np.asarray(phase, dtype=np.float32),
        np.asarray(phi, dtype=np.float32),
    )


def _extract_vtk_array(data_dict: Iterable[str], getter, candidates: list[str]) -> np.ndarray | None:
    names = {str(k) for k in data_dict}
    for c in candidates:
        if c in names:
            arr = getter(c)
            if arr is not None:
                return np.asarray(arr)
    return None


def _load_vtk_volume(path: Path, phase_col: str, phi_col: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "读取 VTK 需要 pyvista，请先安装：pip install pyvista"
        ) from exc

    mesh = pv.read(str(path))

    phase = None
    phi = None

    if hasattr(mesh, "point_data"):
        phase = _extract_vtk_array(mesh.point_data.keys(), mesh.point_data.get_array, [phase_col, "phase", "Phase"])
        phi = _extract_vtk_array(mesh.point_data.keys(), mesh.point_data.get_array, [phi_col, "phi_ion", "phiIon", "phi"])

    # 若 point_data 没有，尝试 cell_data，并用 cell centers 当坐标
    use_cell_centers = False
    if phase is None or phi is None:
        if hasattr(mesh, "cell_data"):
            phase = _extract_vtk_array(mesh.cell_data.keys(), mesh.cell_data.get_array, [phase_col, "phase", "Phase"])
            phi = _extract_vtk_array(mesh.cell_data.keys(), mesh.cell_data.get_array, [phi_col, "phi_ion", "phiIon", "phi"])
            if phase is not None and phi is not None:
                use_cell_centers = True

    if phase is None or phi is None:
        raise ValueError(
            f"VTK 中未找到字段 phase={phase_col} / phi={phi_col}。"
            "请在 ParaView 导出时检查数组名。"
        )

    if use_cell_centers:
        pts = np.asarray(mesh.cell_centers().points)
    else:
        pts = np.asarray(mesh.points)

    if pts.shape[0] != phase.shape[0] or pts.shape[0] != phi.shape[0]:
        raise ValueError("坐标点数量与字段长度不一致，无法重建规则体网格。")

    return _infer_grid_from_xyz(pts[:, 0], pts[:, 1], pts[:, 2], phase, phi)


def _load_volume(path: Path, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    ext = path.suffix.lower()
    if ext == ".csv":
        return _load_csv_volume(
            path,
            phase_col=args.phase_col,
            phi_col=args.phi_col,
            x_col=args.x_col,
            y_col=args.y_col,
            z_col=args.z_col,
        )
    if ext in {".vtk", ".vtu", ".vti"}:
        return _load_vtk_volume(path, phase_col=args.phase_col, phi_col=args.phi_col)
    raise ValueError(f"不支持的输入格式: {path}")


def _process_one(tag: str, src: Path, out_dir: Path, args: argparse.Namespace, allowed: np.ndarray) -> tuple[int, ...]:
    phase, phi = _load_volume(src, args)
    phase = _nearest_map(phase, allowed)
    np.save(out_dir / f"phase_{tag}.npy", phase.astype(np.float32))
    np.save(out_dir / f"phi_ion_{tag}.npy", phi.astype(np.float32))
    print(f"[{tag}] {src.name} -> shape={phase.shape}")
    return tuple(int(x) for x in phase.shape)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed = np.array([float(x.strip()) for x in args.phase_values.split(",")], dtype=np.float32)
    if allowed.size < 3:
        raise ValueError("phase-values 至少应包含三相值，例如 0,128,255")

    low = Path(args.low)
    inter = Path(args.intermediate)
    glob = Path(args.global_opt)
    for p in [low, inter, glob]:
        if not p.exists():
            raise FileNotFoundError(f"输入文件不存在: {p}")

    sh_low = _process_one("low", low, out_dir, args, allowed)
    sh_inter = _process_one("intermediate", inter, out_dir, args, allowed)
    sh_glob = _process_one("global", glob, out_dir, args, allowed)

    write_openfoam_export_manifest(
        out_dir,
        source_files={"low": str(low.resolve()), "intermediate": str(inter.resolve()), "global": str(glob.resolve())},
        phase_col=args.phase_col,
        phi_col=args.phi_col,
        x_col=args.x_col,
        y_col=args.y_col,
        z_col=args.z_col,
        phase_values=[float(x) for x in allowed.tolist()],
        grid_shapes={"low": list(sh_low), "intermediate": list(sh_inter), "global": list(sh_glob)},
    )

    print(f"\n转换完成，输出目录: {out_dir}")
    print("已生成:")
    for name in [
        "phase_low.npy",
        "phase_intermediate.npy",
        "phase_global.npy",
        "phi_ion_low.npy",
        "phi_ion_intermediate.npy",
        "phi_ion_global.npy",
    ]:
        print(f"- {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

