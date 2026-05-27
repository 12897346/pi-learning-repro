from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import zoom

from src.physics.tpb_logic import active_tpb_density_from_label_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将结构生成程序输出转换为 pi-learning 标准训练数据："
            "volumes.npy / tpb.npy / labels_j.npy"
        )
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="输入目录。支持 pores4thought 的 tif，或 QSGS 的 Flag-*.dat + Information-*.dat",
    )
    parser.add_argument(
        "--source-type",
        default="auto",
        choices=["auto", "pores4thought_tif", "qsgs_flag_dat", "single_npy_volume"],
        help="输入类型。auto 会自动识别。",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/generated_bridge",
        help="输出目录（写入 volumes.npy / tpb.npy / labels_j.npy）",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=64,
        help="统一重采样体素边长（默认 64，对齐当前训练配置）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="最多处理样本数（0 表示不限制）",
    )
    parser.add_argument(
        "--use-phys-dnn",
        default="",
        help="可选：phys-DNN 权重路径（.pth）。提供后 labels_j 由模型推断，否则用代理公式生成。",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于代理标签噪声）")
    return parser.parse_args()


def _resize_discrete_volume(vol: np.ndarray, target_size: int) -> np.ndarray:
    vol = np.asarray(vol)
    if vol.ndim != 3:
        raise ValueError(f"体素维度必须是 3D，当前为 {vol.shape}")
    if vol.shape == (target_size, target_size, target_size):
        return vol.astype(np.uint8, copy=False)
    scale = (
        target_size / vol.shape[0],
        target_size / vol.shape[1],
        target_size / vol.shape[2],
    )
    out = zoom(vol, scale, order=0)
    return out.astype(np.uint8, copy=False)


def _normalize_three_phase_values(vol: np.ndarray) -> np.ndarray:
    v = np.asarray(vol).astype(np.float32)
    uniq = np.unique(v)
    if uniq.size == 0:
        raise ValueError("空体素数据")

    # 标准值优先：0/127/128/255
    out = np.zeros_like(v, dtype=np.uint8)
    if np.isin(255, uniq):
        out[np.isclose(v, 255)] = 255
    # 中间相可能是 127 或 128
    out[np.isclose(v, 127) | np.isclose(v, 128)] = 128

    # 若不是标准离散值，按三分位阈值粗映射到三相
    assigned = np.isclose(out, 0) | np.isclose(out, 128) | np.isclose(out, 255)
    if not np.all(assigned):
        q1, q2 = np.quantile(v, [0.33, 0.66])
        out[:] = 0
        out[(v > q1) & (v <= q2)] = 128
        out[v > q2] = 255
    return out


def _compute_tpb_proxy(vol_0128_255: np.ndarray) -> float:
    # 统一使用与训练侧一致的“周期边界 + 连通域过滤 + 三相共边”口径
    mapped = np.zeros_like(vol_0128_255, dtype=np.int8)
    mapped[vol_0128_255 == 0] = 0
    mapped[vol_0128_255 == 128] = 1
    mapped[vol_0128_255 == 255] = 2
    return active_tpb_density_from_label_volume(
        mapped,
        pore_value=0,
        ion_value=2,   # 255 对应 YSZ
        ele_value=1,   # 128 对应 Ni
        min_connected_fraction=0.01,
    )


def _labels_from_proxy_tpb(tpb_vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    eta = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14], dtype=np.float32)
    base = 0.03 + 0.8 * tpb_vec[:, None]
    labels = base * (1.0 + 1.8 * eta[None, :]) + rng.normal(0, 0.003, size=(tpb_vec.shape[0], 7))
    return np.clip(labels, 0.0, None).astype(np.float32)


def _labels_from_phys_dnn(tpb_vec: np.ndarray, ckpt_path: Path) -> np.ndarray:
    import torch

    # 延迟导入，避免脚本在无 torch 环境直接崩溃
    from src.models.phys_dnn import PhysDNN

    model = PhysDNN(input_size=1, hidden_dim=50, output_size=7)
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        x = torch.from_numpy(tpb_vec[:, None].astype(np.float32))
        y = model(x).cpu().numpy()
    return y.astype(np.float32)


def _read_qsgs_info_dat(info_path: Path) -> tuple[int, int, int]:
    nums: list[int] = []
    with info_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                nums.append(int(float(s)))
            except ValueError:
                continue
    if len(nums) < 5:
        raise ValueError(f"Information 文件格式异常: {info_path}")
    nx, ny, nz = nums[2], nums[3], nums[4]
    if min(nx, ny, nz) <= 0:
        raise ValueError(f"非法体素尺寸: {(nx, ny, nz)}")
    return nx, ny, nz


def _read_qsgs_flag_dat(flag_path: Path, nx: int, ny: int, nz: int) -> np.ndarray:
    vals = np.loadtxt(str(flag_path), dtype=np.int32)
    if vals.size != nx * ny * nz:
        raise ValueError(f"Flag 长度与尺寸不匹配: {flag_path} -> {vals.size} vs {nx*ny*nz}")
    # MATLAB 侧写入的是 flag_py=permute(Flag,[3,2,1]) 的列优先展开
    flag_py = vals.reshape((nz, ny, nx), order="F")
    flag = np.transpose(flag_py, (2, 1, 0))  # 回到 [NX, NY, NZ]

    # QSGS 连通性后可能包含 3/4/5（孤立相），映射回主相
    mapped = np.zeros_like(flag, dtype=np.uint8)
    mapped[(flag == 1) | (flag == 3)] = 255
    mapped[(flag == 2) | (flag == 4)] = 128
    mapped[(flag == 0) | (flag == 5)] = 0
    return mapped


def _iter_qsgs_samples(source_root: Path) -> Iterable[np.ndarray]:
    flag_files = sorted(source_root.rglob("Flag-*_py_*.dat"))
    for flag_path in flag_files:
        info_name = flag_path.name.replace("Flag-", "Information-")
        info_path = flag_path.with_name(info_name)
        if not info_path.exists():
            continue
        nx, ny, nz = _read_qsgs_info_dat(info_path)
        yield _read_qsgs_flag_dat(flag_path, nx, ny, nz)


def _iter_tif_samples(source_root: Path) -> Iterable[np.ndarray]:
    try:
        import tifffile
    except Exception as exc:
        raise ImportError("读取 tif 需要安装 tifffile：pip install tifffile") from exc

    tifs = sorted(source_root.rglob("*.tif")) + sorted(source_root.rglob("*.tiff"))
    for p in tifs:
        vol = tifffile.imread(str(p))
        if vol.ndim == 4 and vol.shape[0] == 1:
            vol = vol[0]
        if vol.ndim != 3:
            continue
        yield np.asarray(vol)


def _detect_source_type(source_root: Path) -> str:
    has_qsgs_dat = any(source_root.rglob("Flag-*_py_*.dat"))
    has_tif = any(source_root.rglob("*.tif")) or any(source_root.rglob("*.tiff"))
    has_npy = any(source_root.rglob("*.npy"))
    if has_qsgs_dat:
        return "qsgs_flag_dat"
    if has_tif:
        return "pores4thought_tif"
    if has_npy:
        return "single_npy_volume"
    raise FileNotFoundError(
        "未识别到可用输入：既没有 QSGS Flag-*_py_*.dat，也没有 3D tif/tiff 或 npy 文件。"
    )


def _iter_npy_samples(source_root: Path) -> Iterable[np.ndarray]:
    npys = sorted(source_root.rglob("*.npy"))
    for p in npys:
        arr = np.load(str(p))
        if arr.ndim == 3:
            yield arr
        elif arr.ndim == 4:
            for i in range(arr.shape[0]):
                if arr[i].ndim == 3:
                    yield arr[i]


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    source_root = Path(args.source_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not source_root.exists():
        raise FileNotFoundError(f"输入目录不存在: {source_root}")

    source_type = args.source_type
    if source_type == "auto":
        source_type = _detect_source_type(source_root)

    if source_type == "qsgs_flag_dat":
        raw_iter = _iter_qsgs_samples(source_root)
    elif source_type == "pores4thought_tif":
        raw_iter = _iter_tif_samples(source_root)
    else:
        raw_iter = _iter_npy_samples(source_root)

    volumes_list: list[np.ndarray] = []
    tpb_list: list[float] = []
    for idx, raw_vol in enumerate(raw_iter):
        if args.max_samples > 0 and idx >= args.max_samples:
            break
        vol = _normalize_three_phase_values(raw_vol)
        vol = _resize_discrete_volume(vol, args.target_size)
        tpb_proxy = _compute_tpb_proxy(vol)
        volumes_list.append(vol)
        tpb_list.append(tpb_proxy)

    if not volumes_list:
        raise RuntimeError("没有成功读取到任何样本。请先确认输入目录里有生成的体素文件。")

    volumes = np.stack(volumes_list, axis=0).astype(np.float32)[:, None, ...]  # [N,1,D,H,W]
    tpb = np.asarray(tpb_list, dtype=np.float32)[:, None]  # [N,1]

    if args.use_phys_dnn:
        ckpt = Path(args.use_phys_dnn).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"phys-DNN 权重不存在: {ckpt}")
        labels_j = _labels_from_phys_dnn(tpb[:, 0], ckpt)
        label_mode = f"phys_dnn({ckpt.name})"
    else:
        labels_j = _labels_from_proxy_tpb(tpb[:, 0], rng)
        label_mode = "proxy_formula"

    np.save(out_dir / "volumes.npy", volumes)
    np.save(out_dir / "tpb.npy", tpb)
    np.save(out_dir / "labels_j.npy", labels_j)

    summary = [
        "=== 转换完成 ===",
        f"source_root: {source_root}",
        f"source_type: {source_type}",
        f"label_mode: {label_mode}",
        f"volumes: {volumes.shape}, dtype={volumes.dtype}",
        f"tpb: {tpb.shape}, dtype={tpb.dtype}, min={tpb.min():.6f}, max={tpb.max():.6f}",
        f"labels_j: {labels_j.shape}, dtype={labels_j.dtype}",
        "注意: labels_j 若为 proxy_formula，仅用于流程训练与调试，不代表真实实验标注。",
    ]
    text = "\n".join(summary) + "\n"
    (out_dir / "convert_summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
