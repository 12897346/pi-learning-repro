from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.output_manifest import write_training_bundle_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="整理数据为标准训练输入格式")
    parser.add_argument("--out-dir", default="data/processed", help="输出目录")

    # 方式 A：分别提供 3 个输入文件（推荐）
    parser.add_argument("--volumes-in", default="", help="体素文件路径（.npy）")
    parser.add_argument("--labels-j-in", default="", help="J 标签路径（.npy, 形状 [N,7]）")
    parser.add_argument("--tpb-in", default="", help="TPB 标签路径（.npy, [N] 或 [N,1]）")

    # 方式 B：从单个 h5 文件提取
    parser.add_argument("--h5-in", default="", help="单个 h5 文件路径")
    parser.add_argument("--h5-key-volumes", default="volumes", help="h5 中体素数据 key")
    parser.add_argument("--h5-key-labels-j", default="labels_j", help="h5 中 J 标签 key")
    parser.add_argument("--h5-key-tpb", default="tpb", help="h5 中 TPB key")
    parser.add_argument(
        "--data-provenance",
        default="",
        help="数据来源说明（写入 dataset_manifest.json），例如 OpenFOAM 批算、作者邮件、EDX 等",
    )
    return parser.parse_args()


def _load_npy(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"文件不存在: {p}")
    return np.load(p)


def _load_from_h5(path: str, k_vol: str, k_j: str, k_tpb: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        import h5py  # 延迟导入，避免无 h5py 时影响 npy 流程
    except Exception as exc:
        raise ImportError("读取 h5 需要先安装 h5py：pip install h5py") from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"h5 文件不存在: {p}")
    with h5py.File(p, "r") as f:
        if k_vol not in f or k_j not in f or k_tpb not in f:
            keys = list(f.keys())
            raise KeyError(f"h5 key 不匹配。当前 keys={keys}")
        volumes = np.array(f[k_vol])
        labels_j = np.array(f[k_j])
        tpb = np.array(f[k_tpb])
    return volumes, labels_j, tpb


def _normalize_shapes(
    volumes: np.ndarray,
    labels_j: np.ndarray,
    tpb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if volumes.ndim == 4:
        volumes = volumes[:, None, ...]
    if volumes.ndim != 5:
        raise ValueError(f"volumes 需要 [N,D,H,W] 或 [N,C,D,H,W]，当前 {volumes.shape}")

    if labels_j.ndim != 2 or labels_j.shape[1] != 7:
        raise ValueError(f"labels_j 需要 [N,7]，当前 {labels_j.shape}")

    if tpb.ndim == 1:
        tpb = tpb[:, None]
    if tpb.ndim != 2 or tpb.shape[1] != 1:
        raise ValueError(f"tpb 需要 [N,1] 或 [N]，当前 {tpb.shape}")

    n = volumes.shape[0]
    if labels_j.shape[0] != n or tpb.shape[0] != n:
        raise ValueError(
            "样本数量不一致: "
            f"volumes={volumes.shape[0]}, labels_j={labels_j.shape[0]}, tpb={tpb.shape[0]}"
        )

    return volumes.astype(np.float32), labels_j.astype(np.float32), tpb.astype(np.float32)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.h5_in:
        volumes, labels_j, tpb = _load_from_h5(
            args.h5_in,
            args.h5_key_volumes,
            args.h5_key_labels_j,
            args.h5_key_tpb,
        )
    else:
        if not (args.volumes_in and args.labels_j_in and args.tpb_in):
            raise ValueError(
                "请提供以下两种方式之一：\n"
                "1) --volumes-in --labels-j-in --tpb-in\n"
                "2) --h5-in + 对应 key"
            )
        volumes = _load_npy(args.volumes_in)
        labels_j = _load_npy(args.labels_j_in)
        tpb = _load_npy(args.tpb_in)

    volumes, labels_j, tpb = _normalize_shapes(volumes, labels_j, tpb)

    np.save(out_dir / "volumes.npy", volumes)
    np.save(out_dir / "labels_j.npy", labels_j)
    np.save(out_dir / "tpb.npy", tpb)

    provenance = (
        args.data_provenance.strip()
        or "未声明：请使用 --data-provenance 说明 labels_j/tpb 来源（OpenFOAM/作者/代理等）"
    )
    extra: dict = {}
    if args.h5_in:
        extra["h5_in"] = args.h5_in
    else:
        extra["volumes_in"] = args.volumes_in
        extra["labels_j_in"] = args.labels_j_in
        extra["tpb_in"] = args.tpb_in
    write_training_bundle_manifest(
        out_dir,
        volumes_shape=volumes.shape,
        labels_j_shape=labels_j.shape,
        tpb_shape=tpb.shape,
        data_provenance=provenance,
        extra=extra,
    )

    print("=== 数据整理完成 ===")
    print(f"输出目录: {out_dir}")
    print(f"volumes: {volumes.shape}, dtype={volumes.dtype}")
    print(f"labels_j: {labels_j.shape}, dtype={labels_j.dtype}")
    print(f"tpb: {tpb.shape}, dtype={tpb.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
