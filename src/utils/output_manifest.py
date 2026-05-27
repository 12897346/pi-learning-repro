"""
与论文及孔隙尺度建模常见约定对齐的输出元数据（单位、η 轴、相编码）。

用于在生成 npy/图/场转换结果时附带 dataset_manifest.json 等，避免后续混淆量纲与列含义。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# 与 scripts/evaluate_paper_figures.py 中 eta_mv 及 phys 头输出 7 维一致
PAPER_ETA_MV_MILLIVOLT: tuple[int, ...] = (20, 40, 60, 80, 100, 120, 140)

OUTPUT_SPEC_VERSION = "1.0"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def output_spec_block() -> dict[str, Any]:
    """论文与本仓库约定的物理量输出说明（不含具体文件路径）。"""
    return {
        "output_spec_version": OUTPUT_SPEC_VERSION,
        "paper_reference": "Adv. Energy Mater. 2023, 2300244 (π-learning)",
        "labels_j": {
            "shape_suffix": "[N, 7]",
            "quantity": "current_density_J",
            "unit": "mA cm^-2",
            "axis_eta_mV": list(PAPER_ETA_MV_MILLIVOLT),
            "axis_note": (
                "第 i 列对应 eta_mv[i]；与 OpenFOAM 孔隙尺度模型在同一参考面积定义下提取。"
                "论文印刷一处将七个 η 写成「20,40,60,80,120,140 mV」漏写 100 mV；"
                "本仓库与 Fig.7 / 常用七档采样一致，保留 100 mV。"
            ),
        },
        "tpb": {
            "shape_suffix": "[N, 1]",
            "quantity": "active_tpb_density_proxy",
            "unit": "与 src.physics.tpb_logic 中 active_tpb_density_from_label_volume 定义一致（无量纲密度代理）",
        },
        "volumes": {
            "shape_suffix": "[N, 1, D, H, W]",
            "phase_voxel_values": {"pore": 0, "ni": 128, "ysz": 255},
            "note": "与论文 Experimental Section 三相灰度编码一致；D=H=W=64 时为论文体素块尺寸",
        },
        "openfoam_export_npy": {
            "phase_npy": "0/128/255 三相，经 _nearest_map 与 phase-values 对齐",
            "phi_ion_npy": "离子势场；量纲须与 ParaView/OpenFOAM 导出时所用场定义一致，本仓库不自动换算",
        },
    }


def write_training_bundle_manifest(
    out_dir: Path | str,
    *,
    volumes_shape: tuple[int, ...],
    labels_j_shape: tuple[int, ...],
    tpb_shape: tuple[int, ...],
    data_provenance: str,
    extra: dict[str, Any] | None = None,
    filename: str = "dataset_manifest.json",
) -> Path:
    """
    在 volumes.npy / labels_j.npy / tpb.npy 同目录写入清单。
    data_provenance: 简短说明数据来源，例如 OpenFOAM 批算、作者提供、proxy 合成等。
    """
    root = Path(out_dir)
    body: dict[str, Any] = {
        "generated_at": _now_iso(),
        "data_provenance": data_provenance,
        "arrays": {
            "volumes.npy": {"shape": list(volumes_shape)},
            "labels_j.npy": {"shape": list(labels_j_shape)},
            "tpb.npy": {"shape": list(tpb_shape)},
        },
        "output_spec": output_spec_block(),
    }
    if extra:
        body["extra"] = extra
    path = root / filename
    write_json(path, body)
    return path


def write_openfoam_export_manifest(
    out_dir: Path | str,
    *,
    source_files: dict[str, str],
    phase_col: str,
    phi_col: str,
    x_col: str,
    y_col: str,
    z_col: str,
    phase_values: list[float],
    grid_shapes: dict[str, list[int]],
) -> Path:
    """OpenFOAM/ParaView 导出经 convert_openfoam_to_npy 后的目录说明。"""
    root = Path(out_dir)
    body: dict[str, Any] = {
        "generated_at": _now_iso(),
        "converter": "scripts/convert_openfoam_to_npy.py",
        "source_files": source_files,
        "columns_or_array_names": {
            "phase": phase_col,
            "phi_ion": phi_col,
            "x": x_col,
            "y": y_col,
            "z": z_col,
        },
        "phase_values_allowed": phase_values,
        "grid_shapes_by_tag": grid_shapes,
        "output_spec": output_spec_block(),
    }
    path = root / "openfoam_export_manifest.json"
    write_json(path, body)
    return path


def write_figure_bundle_manifest(
    out_dir: Path | str,
    *,
    produced_png: list[str],
    strict_no_proxy: bool,
    openfoam_dir: str,
) -> Path:
    """列出本脚本写出的图文件及物理含义摘要。"""
    root = Path(out_dir)
    catalog: list[dict[str, str]] = []
    for name in produced_png:
        note = "风格复现图；单位见各图坐标轴"
        if "figure3" in name:
            note = "phys-DNN：J-η / 散点 / 误差；J 单位 mA cm^-2，η 单位 mV"
        elif "figure6" in name:
            note = "PSO：prior J 与迭代最优；J 单位 mA cm^-2"
        elif "figure7_def_openfoam" in name:
            note = "OpenFOAM 导出 ionic potential 与孤立相密度代理；φ 与导出场一致"
        elif "figure7_def_isolation_proxy" in name:
            note = "无真场时的结构代理图，不得当作 OpenFOAM 验证"
        catalog.append({"file": name, "note": note})

    body: dict[str, Any] = {
        "generated_at": _now_iso(),
        "script": "scripts/evaluate_paper_figures.py",
        "strict_no_proxy": strict_no_proxy,
        "openfoam_dir": openfoam_dir or None,
        "figures": catalog,
        "output_spec": output_spec_block(),
    }
    path = root / "figure_manifest.json"
    write_json(path, body)
    return path


def pipeline_manifest_output_block(args_summary: dict[str, Any]) -> dict[str, Any]:
    """供 run_paper_repro_pipeline 写入 pipeline_manifest.json 的固定字段。"""
    return {
        "output_spec": output_spec_block(),
        "run_args": args_summary,
    }
