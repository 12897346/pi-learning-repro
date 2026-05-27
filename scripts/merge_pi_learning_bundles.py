#!/usr/bin/env python3
"""沿样本维拼接两个已对齐的 pi-learning 数据目录（volumes/labels_j/tpb），写 manifest。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.output_manifest import write_training_bundle_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="合并两个训练 bundle（纵向拼接样本）")
    p.add_argument("--bundle-a", type=Path, required=True, help="第一份：须含 volumes.npy / labels_j.npy / tpb.npy")
    p.add_argument("--bundle-b", type=Path, required=True, help="第二份：形状除 N 外须与 A 一致")
    p.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    return p.parse_args()


def _load_triplet(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.load(root / "volumes.npy")
    j = np.load(root / "labels_j.npy")
    t = np.load(root / "tpb.npy")
    if v.ndim == 4:
        v = v[:, None, ...]
    if j.ndim != 2 or j.shape[1] != 7:
        raise ValueError(f"{root} labels_j 须为 [N,7]，当前 {j.shape}")
    if t.ndim == 1:
        t = t[:, None]
    if v.shape[0] != j.shape[0] or v.shape[0] != t.shape[0]:
        raise ValueError(f"{root} 三文件样本数不一致")
    return v.astype(np.float32), j.astype(np.float32), t.astype(np.float32)


def main() -> int:
    args = parse_args()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    a = Path(args.bundle_a).resolve()
    b = Path(args.bundle_b).resolve()

    va, ja, ta = _load_triplet(a)
    vb, jb, tb = _load_triplet(b)
    if va.shape[1:] != vb.shape[1:]:
        raise ValueError(f"volumes 除 N 外形状不一致: A{va.shape} vs B{vb.shape}")

    v = np.concatenate([va, vb], axis=0)
    j = np.concatenate([ja, jb], axis=0)
    t = np.concatenate([ta, tb], axis=0)

    np.save(out / "volumes.npy", v)
    np.save(out / "labels_j.npy", j)
    np.save(out / "tpb.npy", t)

    write_training_bundle_manifest(
        out,
        volumes_shape=v.shape,
        labels_j_shape=j.shape,
        tpb_shape=t.shape,
        data_provenance=f"merge_pi_learning_bundles: A={a} (n={va.shape[0]}) + B={b} (n={vb.shape[0]})",
        extra={"script": "scripts/merge_pi_learning_bundles.py", "bundle_a": str(a), "bundle_b": str(b)},
    )
    meta = {"n_a": int(va.shape[0]), "n_b": int(vb.shape[0]), "n_total": int(v.shape[0])}
    (out / "merge_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("=== 合并完成 ===", f"out={out}", f"volumes={v.shape}", sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
