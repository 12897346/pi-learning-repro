from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# 允许从项目根目录导入 src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.voxel_dataset import (  # noqa: E402
    DataBundle,
    PhysCNNDataset,
    PhysDNNDataset,
    load_data_bundle,
    train_val_split,
)
from src.models.phys_cnn import PhysCNN  # noqa: E402
from src.models.phys_dnn import PhysDNN  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.output_manifest import output_spec_block  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 phys-CNN / phys-DNN")
    parser.add_argument("--config", default="configs/paper_params.yaml", help="配置文件路径")
    parser.add_argument("--data-dir", default="data/processed", help="数据目录")
    parser.add_argument("--out-dir", default="outputs/phys_models", help="模型输出目录")
    parser.add_argument("--model", choices=["phys_cnn", "phys_dnn", "both"], default="both")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="若指定则 phys-DNN 与 phys-CNN 共用该轮数；省略则使用 configs/paper_params.yaml 中各自 train_epochs（论文：DNN≈100、CNN≈70）",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--connectivity-mode",
        choices=["fast", "strict"],
        default="fast",
        help="phys-CNN 第 4 通道：fast=6邻域界面 O(V)；strict=周期连通域并集（慢）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader 子进程数；GPU 训练建议 4~8，与 Slurm cpus-per-task 对齐",
    )
    parser.add_argument("--pin-memory", action="store_true", help="CUDA 时开启 pin_memory 加速 H2D")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--synthetic-if-missing", action="store_true")
    parser.add_argument(
        "--early-stop-val-mae",
        type=float,
        default=None,
        help="验证集 MAE 低于该值则提前结束（正文 phys-DNN：MAE<1；不设则不启用阈值）",
    )
    parser.add_argument(
        "--early-stop-plateau-patience",
        type=int,
        default=0,
        help="val MAE 相对历史最优连续若干 epoch 改善量 < min_delta 则停；0=关闭",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=None,
        help="平台早停：判定 val MAE 有改善的最小下降量；与 patience 同时启用时必填（或由 --paper-early-stop 读 YAML）",
    )
    parser.add_argument(
        "--paper-early-stop",
        action="store_true",
        help="从 configs/paper_params.yaml 的 paper_repro 读取 MAE 阈值与平台判据（首轮默认见 YAML；跑完 metrics 后再标定）",
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def maybe_make_synthetic_bundle() -> DataBundle:
    # 仅用于无数据时快速打通训练流程，不能用于论文结果对比
    n = 24
    volumes = np.random.choice([0.0, 128.0, 255.0], size=(n, 1, 64, 64, 64)).astype(np.float32)
    tpb = np.random.uniform(1200, 2200, size=(n, 1)).astype(np.float32)
    j = np.random.uniform(0.05, 0.25, size=(n, 7)).astype(np.float32)
    return DataBundle(volumes=volumes, j_labels=j, tpb=tpb)


def evaluate_mae(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_mae = 0.0
    total_n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            batch_mae = torch.mean(torch.abs(pred - y)).item()
            total_mae += batch_mae * x.size(0)
            total_n += x.size(0)
    return total_mae / max(total_n, 1)


def train_one_model(
    model_name: str,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    betas: tuple[float, float],
    epochs: int,
    device: torch.device,
    *,
    early_stop_val_mae: float | None = None,
    early_stop_plateau_patience: int = 0,
    early_stop_min_delta: float = 1e-5,
) -> dict:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    loss_fn = torch.nn.L1Loss()

    history: dict = {"train_mae": [], "val_mae": []}
    best_val = float("inf")
    plateau = 0
    exit_reason = "max_epochs"
    epoch_last = 0
    print(f"\n=== 开始训练 {model_name} ===")
    if early_stop_val_mae is not None:
        print(f"[INFO] MAE 阈值早停: val_mae < {early_stop_val_mae} 则结束", flush=True)
    if early_stop_plateau_patience > 0:
        print(
            f"[INFO] 平台早停: val MAE 连续 {early_stop_plateau_patience} epoch "
            f"改善 < {early_stop_min_delta} 则结束",
            flush=True,
        )
    for epoch in range(1, epochs + 1):
        epoch_last = epoch
        model.train()
        running = 0.0
        total = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item() * x.size(0)
            total += x.size(0)

        train_mae = running / max(total, 1)
        val_mae = evaluate_mae(model, val_loader, device)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        print(f"epoch={epoch:03d} train_mae={train_mae:.6f} val_mae={val_mae:.6f}", flush=True)

        if early_stop_val_mae is not None and val_mae < float(early_stop_val_mae):
            exit_reason = "val_mae_below_threshold"
            break

        if early_stop_plateau_patience > 0:
            if val_mae < best_val - float(early_stop_min_delta):
                best_val = val_mae
                plateau = 0
            else:
                plateau += 1
            if plateau >= int(early_stop_plateau_patience):
                exit_reason = "val_mae_plateau"
                break

    history["epochs_run"] = int(epoch_last)
    history["exit_reason"] = exit_reason
    return history


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_yaml(args.config)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        bundle = load_data_bundle(args.data_dir)
        print(f"已加载真实数据: {args.data_dir}")
    except FileNotFoundError as exc:
        if not args.synthetic_if_missing:
            raise FileNotFoundError(
                f"{exc}\n未找到数据。请准备 data/processed 下的 volumes.npy, labels_j.npy, tpb.npy，"
                "或添加 --synthetic-if-missing 先跑通流程。"
            ) from exc
        print("未找到真实数据，使用合成数据仅做流程验证。")
        bundle = maybe_make_synthetic_bundle()

    train_bundle, val_bundle = train_val_split(bundle, val_ratio=args.val_ratio, seed=args.seed)
    summary: dict[str, dict] = {}

    dnn_mae_thr: float | None = args.early_stop_val_mae
    cnn_mae_thr: float | None = args.early_stop_val_mae
    plateau_p = int(args.early_stop_plateau_patience)
    plateau_delta: float | None = args.early_stop_min_delta
    pr = cfg.get("paper_repro") or {}

    if args.paper_early_stop:
        # MAE 阈值仅来自 YAML「phys_dnn_val_mae_early_stop」或 CLI 覆盖；YAML 缺省则不做 MAE 阈值早停
        if dnn_mae_thr is None:
            thr = pr.get("phys_dnn_val_mae_early_stop", None)
            dnn_mae_thr = float(thr) if thr is not None else None
        cnn_mae_thr = None
        if plateau_p == 0:
            p_yaml = pr.get("phys_surrogate_plateau_patience", None)
            plateau_p = int(p_yaml) if p_yaml is not None else 0
        if plateau_p > 0 and plateau_delta is None:
            d_yaml = pr.get("phys_surrogate_plateau_min_delta", None)
            if d_yaml is not None:
                plateau_delta = float(d_yaml)
            else:
                raise ValueError(
                    "已启用 --paper-early-stop 且 phys surrogate 平台耐心>0，但未给出 phys_surrogate_plateau_min_delta。"
                    "正文/SI 未逐字给定时请将 paper_repro.phys_surrogate_plateau_patience 置为 null（仅用 epoch 上沿），"
                    "或在 YAML 中填写你核对 PDF/作者通信后的 phys_surrogate_plateau_min_delta，"
                    "或显式传入 --early-stop-min-delta。"
                )
    elif plateau_p > 0 and plateau_delta is None:
        raise ValueError("使用 --early-stop-plateau-patience>0 时必须指定 --early-stop-min-delta。")

    if plateau_delta is None:
        plateau_delta = 1e-5  # plateau_p==0 时不参与判定，仅占位

    nw = max(0, int(getattr(args, "num_workers", 0) or 0))
    pin_mem = bool(getattr(args, "pin_memory", False))
    loader_kw = dict(num_workers=nw, pin_memory=pin_mem)
    if nw > 0:
        loader_kw["persistent_workers"] = True

    if args.model in ("phys_dnn", "both"):
        dnn_cfg = cfg["phys_dnn"]
        dnn_epochs = args.epochs if args.epochs is not None else int(dnn_cfg.get("train_epochs", 100))
        train_ds = PhysDNNDataset(train_bundle)
        val_ds = PhysDNNDataset(val_bundle)
        train_loader = DataLoader(train_ds, batch_size=dnn_cfg["batch_size"], shuffle=True, **loader_kw)
        val_loader = DataLoader(val_ds, batch_size=dnn_cfg["batch_size"], shuffle=False, **loader_kw)

        model = PhysDNN(
            input_size=dnn_cfg["input_size"],
            hidden_dim=dnn_cfg["hidden_dim"],
            output_size=dnn_cfg["output_size"],
        )
        history = train_one_model(
            model_name="phys_dnn",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=float(dnn_cfg["learning_rate"]),
            betas=tuple(dnn_cfg["adam_betas"]),
            epochs=dnn_epochs,
            device=device,
            early_stop_val_mae=dnn_mae_thr,
            early_stop_plateau_patience=plateau_p,
            early_stop_min_delta=float(plateau_delta),
        )
        torch.save(model.state_dict(), out_dir / "phys_dnn.pth")
        summary["phys_dnn"] = history

    if args.model in ("phys_cnn", "both"):
        cnn_cfg = cfg["phys_cnn"]
        cnn_epochs = args.epochs if args.epochs is not None else int(cnn_cfg.get("train_epochs", 70))
        conn_mode = str(getattr(args, "connectivity_mode", "fast") or "fast")
        train_ds = PhysCNNDataset(train_bundle, connectivity_mode=conn_mode)
        val_ds = PhysCNNDataset(val_bundle, connectivity_mode=conn_mode)
        train_loader = DataLoader(train_ds, batch_size=cnn_cfg["batch_size"], shuffle=True, **loader_kw)
        val_loader = DataLoader(val_ds, batch_size=cnn_cfg["batch_size"], shuffle=False, **loader_kw)

        # 论文补充材料 Table S7: phys-CNN 输入 3+1 通道
        model = PhysCNN(in_channels=4, out_dim=7)
        history = train_one_model(
            model_name="phys_cnn",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=float(cnn_cfg["learning_rate"]),
            betas=tuple(cnn_cfg["adam_betas"]),
            epochs=cnn_epochs,
            device=device,
            early_stop_val_mae=cnn_mae_thr,
            early_stop_plateau_patience=plateau_p,
            early_stop_min_delta=float(plateau_delta),
        )
        torch.save(model.state_dict(), out_dir / "phys_cnn.pth")
        summary["phys_cnn"] = history

    summary["output_spec"] = output_spec_block()
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n训练完成，结果已保存到: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
