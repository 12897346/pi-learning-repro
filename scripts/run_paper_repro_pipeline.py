from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
import sys

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.output_manifest import output_spec_block  # noqa: E402

# 与 forward_design_pso 共用 PSO 平台耐心缩放逻辑
from scripts.forward_design_pso import resolve_pso_plateau_patience  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按论文流程串联：normal预训练 -> physics微调 -> PSO -> 论文图输出")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-dir", default="data/processed_fallback")
    p.add_argument(
        "--phys-epochs",
        type=int,
        default=None,
        help="若指定则 phys-DNN 与 phys-CNN 共用该轮数；省略则按 configs/paper_params.yaml 的 train_epochs（论文约 DNN 100、CNN 70）",
    )
    p.add_argument("--normal-epochs", type=int, default=None, help="省略则读 paper_repro.normal_pretrain_epochs（默认 200）")
    p.add_argument("--physics-epochs", type=int, default=None, help="省略则读 paper_repro.physics_finetune_epochs（默认 80）")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="GAN batch；省略则按 gan_force_tiny / yaml 自动选 8 或 4（paper 网勿超过 4）",
    )
    p.add_argument(
        "--gan-force-tiny",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="GAN/PSO 强制 tiny 通道（省显存）；默认读 paper_repro.gan_force_tiny",
    )
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--num-samples-fig", type=int, default=1000)
    p.add_argument(
        "--pso-eval-microbatch",
        type=int,
        default=32,
        help="forward_design_pso 中 gen 前向微批；CUDA OOM 时改为 8 或 16",
    )
    p.add_argument(
        "--pso-particles",
        type=int,
        default=None,
        help="PSO 粒子数；省略则读 configs/paper_params.yaml 的 pso.particles（默认 200）",
    )
    p.add_argument(
        "--pso-iters",
        type=int,
        default=None,
        help="PSO 迭代轮数；省略则读 paper_repro.pso_iterations（默认 50）",
    )
    p.add_argument(
        "--pso-plateau-patience",
        type=int,
        default=None,
        help="PSO 平台早停耐心；省略则读 YAML pso.plateau_patience 并按 --pso-iters 比例缩放；0=关闭",
    )
    p.add_argument(
        "--pso-plateau-tol",
        type=float,
        default=None,
        help="PSO 判定 gbest_J 有实质提升的最小增量（mA/cm²）；省略则读 YAML pso.plateau_tol（默认 1e-3）",
    )
    p.add_argument(
        "--pso-prior-samples",
        type=int,
        default=None,
        help="PSO 先验 J 采样数；省略则读 pso.prior_samples（默认 400）；0=forward_design 内置规则",
    )
    p.add_argument(
        "--pso-connectivity-mode",
        choices=["fast", "strict"],
        default=None,
        help="PSO 中 Phys-CNN 连通性通道；省略则读 paper_repro.phys_connectivity_mode",
    )
    p.add_argument(
        "--phys-connectivity-mode",
        choices=["fast", "strict"],
        default=None,
        help="训练 phys-CNN 时连通性通道；省略则读 paper_repro.phys_connectivity_mode",
    )
    p.add_argument(
        "--gan-tpb-mode",
        choices=["strict", "fast"],
        default=None,
        help="GAN 训练 TPB：fast=GPU 6邻域（默认）；strict=连通域（慢，仅离线评估推荐）",
    )
    p.add_argument(
        "--no-eval-strict-tpb-at-end",
        action="store_true",
        help="PSO 结束后跳过对 best_latent 的一次 strict TPB 离线评估",
    )
    p.add_argument(
        "--gan-wdist-stable-window",
        type=int,
        default=None,
        help="GAN w_dist 早停窗口；省略则从 configs/paper_params.yaml 的 paper_repro.gan_wdist_early_stop 读取",
    )
    p.add_argument(
        "--gan-wdist-stable-std-tol",
        type=float,
        default=None,
        help="GAN w_dist 窗口内 epoch 均值 std 阈值；省略则从同上 YAML 读取",
    )
    p.add_argument(
        "--gan-wdist-stable-patience",
        type=int,
        default=None,
        help="GAN 连续判稳 epoch 数；0=关闭；省略则从同上 YAML 读取",
    )
    p.add_argument("--skip-openfoam", action="store_true", help="跳过 OpenFOAM 导出转换与真场图")
    p.add_argument("--strict-no-proxy", action="store_true", help="严格无代理模式：要求真场图输入，禁用代理输出")
    p.add_argument("--openfoam-low", default="", help="OpenFOAM low 组文件(csv/vtk)")
    p.add_argument("--openfoam-intermediate", default="", help="OpenFOAM intermediate 组文件(csv/vtk)")
    p.add_argument("--openfoam-global", default="", help="OpenFOAM global 组文件(csv/vtk)")
    p.add_argument("--openfoam-phase-col", default="phase")
    p.add_argument("--openfoam-phi-col", default="phi_ion")
    p.add_argument("--openfoam-x-col", default="x")
    p.add_argument("--openfoam-y-col", default="y")
    p.add_argument("--openfoam-z-col", default="z")
    p.add_argument(
        "--phys-cnn-hybrid-gan-samples",
        type=int,
        default=0,
        help="对齐论文 phys-CNN 混合集：在 GAN physics 之后导出 N 个生成体素，与 --data-dir 合并后仅重训 phys-CNN；0=关闭",
    )
    p.add_argument(
        "--hybrid-gan-labels-j-npy",
        default="",
        help="GAN 子集对应的 OpenFOAM labels_j.npy，形状须为 [N,7] 与 N=phys-cnn-hybrid-gan-samples；空则导出脚本用默认 label-mode",
    )
    p.add_argument(
        "--phys-disable-early-stop",
        action="store_true",
        help="不向 train_phys_models 传入 --paper-early-stop（跑满 yaml 的 epoch 上限）",
    )
    p.add_argument(
        "--gan-preview-every",
        type=int,
        default=None,
        help="每 N epoch 写 previews/ 切片 PNG；0=关闭；省略则从 paper_params.yaml 的 paper_repro.gan_preview_every 读取",
    )
    p.add_argument(
        "--gan-physics-skip-closs-stable",
        action="store_true",
        help="physics GAN：仅 w_dist 判稳早停（等价于 train_gan 的 --physics-skip-closs-stable）",
    )
    p.add_argument(
        "--gan-physics-closs-std-tol",
        type=float,
        default=None,
        help="physics GAN：c_loss 窗口 std 阈值；省略则读 YAML paper_repro.gan_physics_closs_std_tol；均为空则不启用第二判据",
    )
    p.add_argument(
        "--export-pso-surface",
        choices=["off", "iso", "both"],
        default="off",
        help="PSO 最优潜向量经 GAN→体素后导出 3D 示意 PNG（默认 matplotlib 后端，无 OSMesa；默认 off）。",
    )
    p.add_argument(
        "--export-surface-device",
        default=None,
        help="表面导出步骤中 GAN 前向设备；默认与 --device 相同，登录节点可显式传 cpu",
    )
    p.add_argument(
        "--export-surface-gan-arch",
        choices=["auto", "tiny", "paper"],
        default="auto",
        help="表面导出加载的 Generator 通道：auto=与 configs/paper_params.yaml 中 gan.debug_tiny 一致",
    )
    p.add_argument(
        "--export-surface-backend",
        choices=["matplotlib", "pyvista"],
        default="matplotlib",
        help="表面 PNG：matplotlib 无 OpenGL（集群推荐）；pyvista 需 mesa/xvfb",
    )
    p.add_argument(
        "--export-surface-mpl-downsample",
        type=int,
        default=2,
        help="matplotlib 表面体素下采样步长（>=1）",
    )
    p.add_argument(
        "--export-surface-reference-data-dir",
        default="",
        help="表面图 (b) 学习曲线用的 tpb/labels_j 目录；默认与 --data-dir 相同",
    )
    return p.parse_args()


def _phys_loader_flags(args: argparse.Namespace) -> list[str]:
    flags = ["--num-workers", str(int(args.num_workers))]
    if str(args.device).startswith("cuda"):
        flags.append("--pin-memory")
    return flags


def _bootstrap_pipeline_log(log_path: Path, py: str, args: argparse.Namespace) -> None:
    """在任何子进程写入前先落盘几行，避免作业早退时 pipeline.log 仍为 0 字节。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    banner = (
        f"{'=' * 60}\n"
        f"[{ts}] pipeline 主进程已启动（先于子步骤写入，保证本文件可见）\n"
        f"cwd: {Path.cwd()}\n"
        f"python: {py}\n"
        f"--out-dir: {args.out_dir}\n"
        f"--data-dir: {args.data_dir}\n"
        f"--device: {args.device}\n"
        f"提示: 子进程带 PYTHONUNBUFFERED=1；若仍几乎空请确认作业 cwd 与 --out-dir 是否为相对路径。\n"
        f"{'=' * 60}\n"
    )
    with log_path.open("w", encoding="utf-8") as f:
        f.write(banner)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def _write_manifest(manifest_path: Path, manifest: dict) -> None:
    """每次检查点落盘，便于作业被杀/超时后仍能看出停在哪一步。"""
    manifest["last_checkpoint_at"] = datetime.now().isoformat()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run_step(
    cmd: list[str],
    step_name: str,
    log_path: Path,
    *,
    manifest_path: Path,
    manifest: dict,
    step_id: str,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"\n[{ts}] >>> {step_name} [step_id={step_id}]\n$ {' '.join(cmd)}\n"
    print(header.strip())
    manifest["pipeline_phase"] = f"running_{step_id}"
    manifest["current_step_id"] = step_id
    manifest["current_step_name"] = step_name
    manifest["current_step_started_at"] = datetime.now().isoformat()
    _write_manifest(manifest_path, manifest)

    sub_env = os.environ.copy()
    sub_env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(header)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
        proc = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            env=sub_env,
        )
    ts_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer = f"[{ts_end}] <<< {step_name} [step_id={step_id}] exit={proc.returncode}\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(footer)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

    if proc.returncode != 0:
        manifest["status"] = "failed"
        manifest["failed_step_id"] = step_id
        manifest["failed_step_name"] = step_name
        manifest["failed_exit_code"] = proc.returncode
        manifest["pipeline_phase"] = f"failed_after_{step_id}"
        manifest.pop("current_step_id", None)
        _write_manifest(manifest_path, manifest)
        raise RuntimeError(f"{step_name} 失败，退出码 {proc.returncode}。日志: {log_path}")

    manifest.setdefault("steps", []).append(
        {
            "step_id": step_id,
            "step_name": step_name,
            "finished_at": datetime.now().isoformat(),
            "exit_code": proc.returncode,
        }
    )
    manifest["last_completed_step"] = step_id
    manifest["pipeline_phase"] = f"after_{step_id}"
    manifest.pop("current_step_id", None)
    manifest.pop("current_step_name", None)
    manifest.pop("current_step_started_at", None)
    manifest["status"] = "running"
    _write_manifest(manifest_path, manifest)


def assert_exists(paths: list[Path], hint: str = "") -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        extra = f"\n提示: {hint}" if hint else ""
        raise FileNotFoundError("缺少预期文件:\n- " + "\n- ".join(missing) + extra)


def _data_dir_hint(data_root: Path) -> str:
    """在缺文件时给出可操作说明（含占位路径检测）。"""
    lines = [
        "请先准备训练数据或运行数据构建脚本。",
        "仓库根目录可生成替代数据: python scripts/build_fallback_training_data.py --out-dir data/processed_fallback",
        "Slurm 提交示例: export DATA_DIR=\"$PROJECT_DIR/data/processed_fallback\"（路径须为集群上真实目录）。",
    ]
    s = str(data_root.resolve() if data_root.exists() else data_root).replace("\\", "/").lower()
    if "path/to" in s or "/your/npy" in s or "your_npy_dir" in s:
        lines.insert(
            1,
            "当前 --data-dir 疑似文档占位路径，请改为真实目录；不要沿用示例中的 /path/to/...。",
        )
    if not data_root.is_dir():
        lines.insert(1, f"数据目录不存在或不是目录: {data_root}")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    py = args.python
    # 一律解析为绝对路径，避免「以为在 pi-learning/outputs 写 log，实际 cwd 不对」导致 pipeline.log 空或找不到
    out = Path(args.out_dir).expanduser().resolve()
    gan_out = out / "gan_fallback"
    phys_out = out / "phys_models"
    fwd_out = out / "forward_design"
    fig_out = out / "paper_figures"
    openfoam_out = out / "openfoam_export"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "pipeline.log"
    manifest_path = out / "pipeline_manifest.json"
    _bootstrap_pipeline_log(log_path, py, args)

    # 预检查输入数据
    data_root = Path(args.data_dir)
    assert_exists(
        [
            data_root / "volumes.npy",
            data_root / "labels_j.npy",
            data_root / "tpb.npy",
        ],
        hint=_data_dir_hint(data_root),
    )

    _paper_yaml_path = _PROJECT_ROOT / "configs" / "paper_params.yaml"
    with _paper_yaml_path.open(encoding="utf-8") as f:
        _paper_cfg = yaml.safe_load(f)
    _paper_repro = _paper_cfg.get("paper_repro") or {}
    _pso_cfg = _paper_cfg.get("pso") or {}
    _gan_es = _paper_repro.get("gan_wdist_early_stop") or {}
    if args.gan_wdist_stable_window is None:
        args.gan_wdist_stable_window = int(_gan_es["window"]) if _gan_es else 0
    if args.gan_wdist_stable_std_tol is None:
        args.gan_wdist_stable_std_tol = float(_gan_es["std_tol"]) if _gan_es else 0.0
    if args.gan_wdist_stable_patience is None:
        args.gan_wdist_stable_patience = int(_gan_es["patience"]) if _gan_es else 0

    if args.normal_epochs is None:
        args.normal_epochs = int(_paper_repro.get("normal_pretrain_epochs", 200) or 200)
    if args.physics_epochs is None:
        args.physics_epochs = int(_paper_repro.get("physics_finetune_epochs", 80) or 80)
    if args.pso_particles is None:
        args.pso_particles = int(_pso_cfg.get("particles", 200) or 200)
    if args.pso_iters is None:
        args.pso_iters = int(_paper_repro.get("pso_iterations", 50) or 50)

    _pso_ref_iters = int(_paper_repro.get("pso_iterations", 300) or 300)
    if args.pso_plateau_patience is None:
        args.pso_plateau_patience = int(_pso_cfg.get("plateau_patience", 200) or 200)
    if int(args.pso_plateau_patience) > 0:
        args.pso_plateau_patience = resolve_pso_plateau_patience(
            int(args.pso_plateau_patience),
            int(args.pso_iters),
            reference_iters=_pso_ref_iters,
            paper_patience=int(_pso_cfg.get("plateau_patience", 200) or 200),
        )
    if args.pso_plateau_tol is None:
        args.pso_plateau_tol = float(_pso_cfg.get("plateau_tol", 1.0e-3) or 1.0e-3)
    if args.pso_prior_samples is None:
        args.pso_prior_samples = int(_pso_cfg.get("prior_samples", 400) or 400)
    if args.gan_tpb_mode is None:
        args.gan_tpb_mode = str(_paper_repro.get("gan_tpb_mode", "fast") or "fast")
    _conn_default = str(_paper_repro.get("phys_connectivity_mode", "fast") or "fast")
    if args.phys_connectivity_mode is None:
        args.phys_connectivity_mode = _conn_default
    if args.pso_connectivity_mode is None:
        args.pso_connectivity_mode = _conn_default
    if args.phys_epochs is None:
        _phys_cap = _paper_repro.get("phys_train_epochs", None)
        if _phys_cap is not None:
            args.phys_epochs = int(_phys_cap)
    _eval_strict_end = bool(_paper_repro.get("eval_strict_tpb_at_end", True))
    if getattr(args, "no_eval_strict_tpb_at_end", False):
        _eval_strict_end = False

    _gan_cfg = _paper_cfg.get("gan") or {}
    if args.gan_force_tiny is None:
        args.gan_force_tiny = bool(_paper_repro.get("gan_force_tiny", True))
    _use_tiny_gan = bool(args.gan_force_tiny) or bool(_gan_cfg.get("debug_tiny", True))
    if args.batch_size is None:
        if _use_tiny_gan:
            args.batch_size = int(_paper_repro.get("gan_batch_size_tiny", _gan_cfg.get("batch_size", 8)) or 8)
        else:
            args.batch_size = int(_paper_repro.get("gan_batch_size", 4) or 4)
    gan_train_extra: list[str] = ["--force-tiny"] if args.gan_force_tiny else []
    gan_infer_extra: list[str] = ["--force-tiny"] if args.gan_force_tiny else []

    if getattr(args, "gan_preview_every", None) is None:
        args.gan_preview_every = int(_paper_repro.get("gan_preview_every", 0) or 0)

    _resolved_gan_physics_closs: float | None = getattr(args, "gan_physics_closs_std_tol", None)
    if _resolved_gan_physics_closs is None:
        _c_yaml = _paper_repro.get("gan_physics_closs_std_tol", None)
        _resolved_gan_physics_closs = float(_c_yaml) if _c_yaml is not None else None

    manifest = {
        "started_at": datetime.now().isoformat(),
        "device": args.device,
        "data_dir": args.data_dir,
        "phys_epochs": args.phys_epochs,
        "normal_epochs": args.normal_epochs,
        "physics_epochs": args.physics_epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "strict_no_proxy": bool(args.strict_no_proxy),
        "skip_openfoam": bool(args.skip_openfoam),
        "phys_cnn_hybrid_gan_samples": int(getattr(args, "phys_cnn_hybrid_gan_samples", 0) or 0),
        "hybrid_gan_labels_j_npy": getattr(args, "hybrid_gan_labels_j_npy", "") or None,
        "pso_particles": int(getattr(args, "pso_particles", 1000) or 1000),
        "pso_iters": int(getattr(args, "pso_iters", 300) or 300),
        "pso_plateau_patience": int(getattr(args, "pso_plateau_patience", 0) or 0),
        "pso_plateau_tol": float(getattr(args, "pso_plateau_tol", 1.0e-3) or 1.0e-3),
        "pso_prior_samples": int(getattr(args, "pso_prior_samples", 0) or 0),
        "gan_tpb_mode": str(getattr(args, "gan_tpb_mode", "fast") or "fast"),
        "phys_connectivity_mode": str(getattr(args, "phys_connectivity_mode", "fast") or "fast"),
        "pso_connectivity_mode": str(getattr(args, "pso_connectivity_mode", "fast") or "fast"),
        "eval_strict_tpb_at_end": bool(_eval_strict_end),
        "gan_force_tiny": bool(args.gan_force_tiny),
        "gan_wdist_stable_window": int(args.gan_wdist_stable_window),
        "gan_wdist_stable_std_tol": float(args.gan_wdist_stable_std_tol),
        "gan_wdist_stable_patience": int(args.gan_wdist_stable_patience),
        "phys_disable_early_stop": bool(getattr(args, "phys_disable_early_stop", False)),
        "gan_preview_every": int(args.gan_preview_every or 0),
        "gan_physics_skip_closs_stable": bool(getattr(args, "gan_physics_skip_closs_stable", False)),
        "gan_physics_closs_std_tol": _resolved_gan_physics_closs,
        "export_pso_surface": str(getattr(args, "export_pso_surface", "off") or "off"),
        "export_surface_device": getattr(args, "export_surface_device", None) or args.device,
        "export_surface_gan_arch": str(getattr(args, "export_surface_gan_arch", "auto") or "auto"),
        "export_surface_backend": str(getattr(args, "export_surface_backend", "matplotlib") or "matplotlib"),
        "export_surface_mpl_downsample": int(getattr(args, "export_surface_mpl_downsample", 2) or 2),
        "export_surface_reference_data_dir": str(
            getattr(args, "export_surface_reference_data_dir", "") or args.data_dir
        ),
        "output_spec": output_spec_block(),
        "status": "running",
        "pipeline_phase": "after_data_check",
        "last_completed_step": "data_check",
        "steps": [
            {
                "step_id": "data_check",
                "step_name": "输入数据 volumes/labels_j/tpb 校验",
                "finished_at": datetime.now().isoformat(),
            }
        ],
    }
    _write_manifest(manifest_path, manifest)

    phys_paper_early: list[str] = [] if args.phys_disable_early_stop else ["--paper-early-stop"]
    gan_preview_flags: list[str] = []
    if int(args.gan_preview_every or 0) > 0:
        gan_preview_flags = ["--gan-preview-every", str(int(args.gan_preview_every))]
    gan_physics_flags: list[str] = []
    if bool(getattr(args, "gan_physics_skip_closs_stable", False)):
        gan_physics_flags.append("--physics-skip-closs-stable")
    if _resolved_gan_physics_closs is not None:
        gan_physics_flags += ["--physics-closs-std-tol", str(float(_resolved_gan_physics_closs))]

    if args.strict_no_proxy:
        # 严格模式下移除所有可能残留的代理评估产物
        fig_out.mkdir(parents=True, exist_ok=True)
        for p in fig_out.glob("*proxy*"):
            p.unlink()

    try:
        # 0) phys surrogate
        run_step(
            [
                py,
                "scripts/train_phys_models.py",
                "--data-dir",
                args.data_dir,
                "--out-dir",
                str(phys_out),
                "--model",
                "both",
                *phys_paper_early,
                *(
                    ["--epochs", str(args.phys_epochs)]
                    if args.phys_epochs is not None
                    else []
                ),
                "--device",
                args.device,
                "--connectivity-mode",
                str(args.phys_connectivity_mode),
                *_phys_loader_flags(args),
            ],
            "训练 phys surrogate",
            log_path,
            manifest_path=manifest_path,
            manifest=manifest,
            step_id="phys_surrogate",
        )
        assert_exists([phys_out / "phys_dnn.pth", phys_out / "phys_cnn.pth"], hint="phys 模型未正确产出。")

        # 1) normal GAN pretrain
        run_step(
            [
                py,
                "scripts/train_gan_fallback.py",
                "--data-dir",
                args.data_dir,
                "--out-dir",
                str(gan_out),
                "--training-stage",
                "normal",
                "--epochs",
                str(args.normal_epochs),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--pin-memory",
                "--amp",
                "--tpb-mode",
                str(getattr(args, "gan_tpb_mode", "strict") or "strict"),
                "--save-every",
                "100",
                *(
                    [
                        "--wdist-stable-window",
                        str(int(args.gan_wdist_stable_window)),
                        "--wdist-stable-std-tol",
                        str(float(args.gan_wdist_stable_std_tol)),
                        "--wdist-stable-patience",
                        str(int(args.gan_wdist_stable_patience)),
                    ]
                    if int(getattr(args, "gan_wdist_stable_patience", 0) or 0) > 0
                    and int(getattr(args, "gan_wdist_stable_window", 0) or 0) > 0
                    else []
                ),
                *gan_preview_flags,
                *gan_train_extra,
            ],
            "GAN normal 预训练",
            log_path,
            manifest_path=manifest_path,
            manifest=manifest,
            step_id="gan_normal",
        )
        assert_exists([gan_out / "generator_fallback.pth", gan_out / "critic_fallback.pth", gan_out / "metrics_normal.csv"])

        # 2) physics-informed finetune
        run_step(
            [
                py,
                "scripts/train_gan_fallback.py",
                "--data-dir",
                args.data_dir,
                "--out-dir",
                str(gan_out),
                "--training-stage",
                "physics",
                "--epochs",
                str(args.physics_epochs),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--pin-memory",
                "--amp",
                "--tpb-mode",
                str(getattr(args, "gan_tpb_mode", "strict") or "strict"),
                "--tpb-grad-mode",
                "ste_soft",
                "--resume-gen",
                str(gan_out / "generator_fallback.pth"),
                "--resume-critic",
                str(gan_out / "critic_fallback.pth"),
                "--phys-dnn-ckpt",
                str(phys_out / "phys_dnn.pth"),
                "--save-every",
                "50",
                *(
                    [
                        "--wdist-stable-window",
                        str(int(args.gan_wdist_stable_window)),
                        "--wdist-stable-std-tol",
                        str(float(args.gan_wdist_stable_std_tol)),
                        "--wdist-stable-patience",
                        str(int(args.gan_wdist_stable_patience)),
                    ]
                    if int(getattr(args, "gan_wdist_stable_patience", 0) or 0) > 0
                    and int(getattr(args, "gan_wdist_stable_window", 0) or 0) > 0
                    else []
                ),
                *gan_preview_flags,
                *gan_physics_flags,
                *gan_train_extra,
            ],
            "GAN physics 微调",
            log_path,
            manifest_path=manifest_path,
            manifest=manifest,
            step_id="gan_physics",
        )
        assert_exists([gan_out / "generator_fallback.pth", gan_out / "critic_fallback.pth", gan_out / "metrics_physics.csv"])

        # 2b) 论文反馈链：真实 + GAN 混合 → 仅重训 phys-CNN（可选）
        if int(getattr(args, "phys_cnn_hybrid_gan_samples", 0) or 0) > 0:
            n_hybrid = int(args.phys_cnn_hybrid_gan_samples)
            gan_export = out / "gan_export_phys_cnn_hybrid"
            hybrid_bundle = out / "phys_cnn_hybrid_bundle"
            cfg_path = str(_PROJECT_ROOT / "configs/paper_params.yaml")
            export_cmd = [
                py,
                "scripts/export_gan_microstructures.py",
                "--config",
                cfg_path,
                "--gen-ckpt",
                str(gan_out / "generator_fallback.pth"),
                "--out-dir",
                str(gan_export),
                "--num-samples",
                str(n_hybrid),
                "--device",
                args.device,
            ]
            if getattr(args, "hybrid_gan_labels_j_npy", "").strip():
                export_cmd += [
                    "--labels-j-npy",
                    str(Path(args.hybrid_gan_labels_j_npy).expanduser().resolve()),
                ]
            run_step(
                export_cmd,
                "导出 GAN 微结构（phys-CNN 混合集）",
                log_path,
                manifest_path=manifest_path,
                manifest=manifest,
                step_id="export_gan_hybrid",
            )
            run_step(
                [
                    py,
                    "scripts/merge_pi_learning_bundles.py",
                    "--bundle-a",
                    str(Path(args.data_dir).resolve()),
                    "--bundle-b",
                    str(gan_export.resolve()),
                    "--out-dir",
                    str(hybrid_bundle.resolve()),
                ],
                "合并真实与 GAN 训练 bundle",
                log_path,
                manifest_path=manifest_path,
                manifest=manifest,
                step_id="merge_phys_cnn_hybrid",
            )
            hybrid_train_cmd = [
                py,
                "scripts/train_phys_models.py",
                "--config",
                cfg_path,
                "--data-dir",
                str(hybrid_bundle.resolve()),
                "--out-dir",
                str(phys_out.resolve()),
                "--model",
                "phys_cnn",
                *phys_paper_early,
                "--device",
                args.device,
                "--connectivity-mode",
                str(args.phys_connectivity_mode),
                *_phys_loader_flags(args),
            ]
            if args.phys_epochs is not None:
                hybrid_train_cmd += ["--epochs", str(args.phys_epochs)]
            run_step(
                hybrid_train_cmd,
                "phys-CNN 混合数据重训（反馈链）",
                log_path,
                manifest_path=manifest_path,
                manifest=manifest,
                step_id="phys_cnn_hybrid_retrain",
            )
            assert_exists([phys_out / "phys_cnn.pth"], hint="混合重训后 phys_cnn 未写出。")

        # 3) forward design PSO
        run_step(
            [
                py,
                "scripts/forward_design_pso.py",
                "--gen-ckpt",
                str(gan_out / "generator_fallback.pth"),
                "--phys-dnn-ckpt",
                str(phys_out / "phys_dnn.pth"),
                "--phys-cnn-ckpt",
                str(phys_out / "phys_cnn.pth"),
                "--out-dir",
                str(fwd_out),
                "--device",
                args.device,
                "--surrogate",
                "phys_cnn",
                "--particles",
                str(int(args.pso_particles)),
                "--iters",
                str(int(args.pso_iters)),
                "--eval-microbatch",
                str(args.pso_eval_microbatch),
                *(
                    ["--plateau-patience", str(int(args.pso_plateau_patience))]
                    if int(getattr(args, "pso_plateau_patience", 0) or 0) > 0
                    else []
                ),
                *(
                    ["--plateau-tol", str(float(args.pso_plateau_tol))]
                    if int(getattr(args, "pso_plateau_patience", 0) or 0) > 0
                    else []
                ),
                *(
                    ["--prior-samples", str(int(args.pso_prior_samples))]
                    if int(getattr(args, "pso_prior_samples", 0) or 0) > 0
                    else []
                ),
                "--connectivity-mode",
                str(args.pso_connectivity_mode),
                *gan_infer_extra,
            ],
            "PSO forward design",
            log_path,
            manifest_path=manifest_path,
            manifest=manifest,
            step_id="pso",
        )
        assert_exists([fwd_out / "prior_j.npy", fwd_out / "pso_history.csv", fwd_out / "best_latent.npy"])

        if _eval_strict_end:
            run_step(
                [
                    py,
                    "scripts/eval_best_latent_strict_tpb.py",
                    "--gen-ckpt",
                    str(gan_out / "generator_fallback.pth"),
                    "--best-latent-npy",
                    str(fwd_out / "best_latent.npy"),
                    "--out-json",
                    str(fwd_out / "strict_tpb_eval.json"),
                    "--device",
                    args.device,
                    *gan_infer_extra,
                ],
                "离线 strict TPB 评估（best_latent）",
                log_path,
                manifest_path=manifest_path,
                manifest=manifest,
                step_id="eval_strict_tpb_best",
            )
            assert_exists([fwd_out / "strict_tpb_eval.json"])

        openfoam_dir_arg = ""
        if not args.skip_openfoam:
            if args.openfoam_low and args.openfoam_intermediate and args.openfoam_global:
                run_step(
                    [
                        py,
                        "scripts/convert_openfoam_to_npy.py",
                        "--low",
                        args.openfoam_low,
                        "--intermediate",
                        args.openfoam_intermediate,
                        "--global-opt",
                        args.openfoam_global,
                        "--out-dir",
                        str(openfoam_out),
                        "--phase-col",
                        args.openfoam_phase_col,
                        "--phi-col",
                        args.openfoam_phi_col,
                        "--x-col",
                        args.openfoam_x_col,
                        "--y-col",
                        args.openfoam_y_col,
                        "--z-col",
                        args.openfoam_z_col,
                    ],
                    "OpenFOAM 导出转换 .npy",
                    log_path,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    step_id="openfoam_export",
                )
                assert_exists(
                    [
                        openfoam_out / "phase_low.npy",
                        openfoam_out / "phase_intermediate.npy",
                        openfoam_out / "phase_global.npy",
                        openfoam_out / "phi_ion_low.npy",
                        openfoam_out / "phi_ion_intermediate.npy",
                        openfoam_out / "phi_ion_global.npy",
                    ]
                )
                openfoam_dir_arg = str(openfoam_out)
            else:
                if args.strict_no_proxy:
                    raise ValueError("strict-no-proxy 模式下必须提供完整 OpenFOAM 输入。")
                print("未提供完整 OpenFOAM 输入（low/intermediate/global），将跳过真场图。")
        elif args.strict_no_proxy:
            raise ValueError("strict-no-proxy 模式下不允许 --skip-openfoam。")

        # 4) paper figures
        fig_cmd = [
            py,
            "scripts/evaluate_paper_figures.py",
            "--data-dir",
            args.data_dir,
            "--gen-ckpt",
            str(gan_out / "generator_fallback.pth"),
            "--phys-dnn-ckpt",
            str(phys_out / "phys_dnn.pth"),
            "--gan-metrics",
            str(gan_out / "metrics_physics.csv"),
            "--pso-history",
            str(fwd_out / "pso_history.csv"),
            "--prior-j",
            str(fwd_out / "prior_j.npy"),
            "--out-dir",
            str(fig_out),
            "--device",
            args.device,
            "--num-samples",
            str(args.num_samples_fig),
        ]
        if openfoam_dir_arg:
            fig_cmd += ["--openfoam-dir", openfoam_dir_arg]
        if args.strict_no_proxy:
            fig_cmd += ["--strict-no-proxy"]
        run_step(
            fig_cmd,
            "论文图输出",
            log_path,
            manifest_path=manifest_path,
            manifest=manifest,
            step_id="paper_figures",
        )
        assert_exists(
            [
                fig_out / "figure2_cd_wasserstein.png",
                fig_out / "figure3_phys_dnn_regression.png",
                fig_out / "figure5_ab_distributions.png",
                fig_out / "figure5_c_two_point.png",
                fig_out / "figure6_a_prior_pso.png",
                fig_out / "figure6_bcd_optimal.png",
                fig_out / "figure7_abc_performance_insight.png",
            ]
        )
        if openfoam_dir_arg:
            assert_exists([fig_out / "figure7_def_openfoam.png"])
        elif args.strict_no_proxy:
            raise FileNotFoundError("strict-no-proxy 模式要求 figure7_def_openfoam.png，但未生成。")
        if args.strict_no_proxy:
            left_proxy = list(fig_out.glob("*proxy*"))
            if left_proxy:
                raise RuntimeError(f"strict-no-proxy 模式下不允许存在代理产物: {[str(x) for x in left_proxy]}")

        # 4b) 可选：PSO 最优结构 → 3D 示意 PNG（默认 matplotlib，无 OSMesa；默认关闭整步）
        surface_mode = str(getattr(args, "export_pso_surface", "off") or "off").lower()
        if surface_mode not in ("off", "iso", "both"):
            surface_mode = "off"
        if surface_mode != "off":
            surface_out = fig_out / "pso_best_surface"
            cfg_gan_path = str(_PROJECT_ROOT / "configs" / "paper_params.yaml")
            surf_dev = str(getattr(args, "export_surface_device", None) or args.device)
            arch = str(getattr(args, "export_surface_gan_arch", "auto") or "auto")
            ref_data = str(getattr(args, "export_surface_reference_data_dir", "") or args.data_dir)
            surf_cmd: list[str] = [
                py,
                "scripts/export_voxel_surface_figure.py",
                "--best-latent-npy",
                str(fwd_out / "best_latent.npy"),
                "--gen-ckpt",
                str(gan_out / "generator_fallback.pth"),
                "--gan-config",
                cfg_gan_path,
                "--device",
                surf_dev,
                "--phys-dnn-ckpt",
                str(phys_out / "phys_dnn.pth"),
                "--phys-cnn-ckpt",
                str(phys_out / "phys_cnn.pth"),
                "--reference-data-dir",
                ref_data,
                "--out-dir",
                str(surface_out),
                "--basename",
                "pso_best_surface",
                "--views",
                surface_mode,
                "--surface-backend",
                str(getattr(args, "export_surface_backend", "matplotlib") or "matplotlib"),
                "--mpl-downsample",
                str(int(getattr(args, "export_surface_mpl_downsample", 2) or 2)),
            ]
            if arch == "tiny":
                surf_cmd.append("--gan-force-tiny")
            elif arch == "paper":
                surf_cmd.append("--gan-force-paper")
            run_step(
                surf_cmd,
                "PSO 最优 3D 表面图（可选）",
                log_path,
                manifest_path=manifest_path,
                manifest=manifest,
                step_id="export_pso_surface",
            )
            assert_exists(
                [surface_out / "pso_best_surface_iso.png"],
                hint="export_voxel_surface_figure 未写出等轴 PNG；若用 pyvista 请检查 mesa/xvfb；gan 通道是否与 ckpt 一致可试 --export-surface-gan-arch tiny。",
            )
            if surface_mode == "both":
                assert_exists(
                    [surface_out / "pso_best_surface_multiview.png"],
                    hint="--export-pso-surface both 时还应存在三视图拼板 PNG。",
                )

        manifest["finished_at"] = datetime.now().isoformat()
        manifest["status"] = "success"
        manifest["pipeline_phase"] = "completed"
        manifest["artifacts"] = {
            "phys_models": str(phys_out),
            "gan": str(gan_out),
            "forward_design": str(fwd_out),
            "paper_figures": str(fig_out),
            "openfoam_npy": str(openfoam_out) if openfoam_dir_arg else None,
            "pso_best_surface": str(fig_out / "pso_best_surface")
            if str(getattr(args, "export_pso_surface", "off") or "off").lower() != "off"
            else None,
        }
        _write_manifest(manifest_path, manifest)
        print(f"论文复现流水线执行完成。日志: {log_path}")
        print(f"检查点说明: 运行 python scripts/diagnose_pipeline_run.py --out-dir {out}")
        return 0

    except Exception as exc:
        # run_step 内已失败时 status 已为 failed，此处只补充 assert 等未捕获原因
        if manifest.get("status") == "running":
            manifest["status"] = "failed"
            manifest["failed_reason"] = f"{type(exc).__name__}: {exc}"
            manifest["pipeline_phase"] = "failed_uncaught"
        _write_manifest(manifest_path, manifest)
        print(f"[pipeline] 已写入检查点: {manifest_path}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

