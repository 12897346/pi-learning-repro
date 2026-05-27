#!/usr/bin/env python3
"""
z 向 2D TIF 切片（*_zNNN.tif）→ 随机裁 patch³ → volumes.npy / tpb.npy / labels_j.npy。
可选 --resize-out：裁块后最近邻放大到边长 N（薄 Z 栈 patch<64 时对齐论文 64³ GAN）。

设计目标（与 Adv. Energy Mater. 2023, 2300244 数据准备一致）：
- **流式读盘**：不整卷堆叠，避免大图占满内存；
- **防漏读**：默认只扫 `--slice-dir` 本层；可选 `--recursive`；若本层无切片则**自动下探一层子目录**并选切片数最多的候选；
- **严格校验**：`--strict-z` 要求 z 序号连续无洞；重复 z、空间尺寸不一致会报错；
- **清单**：写出 `slice_inventory.json` 便于核对「是否读全」；
- **OpenFOAM 标签**：`--labels-j-npy` 提供与裁块顺序无关的 **预生成 [N,7]** 时，须与 `--num-patches` 行数一致（你方数据完整则应能一一对齐；不一致说明转换/配对错误）。

若未提供 `--labels-j-npy`，`labels_j` 仍用 proxy（仅流程占位），与论文真值不同。

**与 OpenFOAM 标签对齐**：`crop_manifest.json` 记录每个 patch 的裁窗
（`z_start_logical`…`z_end_logical`、`dy`、`dx`）。你方应用同一 `seed` 与同一裁窗规则
生成 `[N,7]` 再传入 `--labels-j-npy`；若行数或顺序不一致，应检查标签管线而非假定「缺数据」。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.physics.tpb_logic import active_tpb_density_from_label_volume  # noqa: E402
from src.utils.output_manifest import write_training_bundle_manifest  # noqa: E402


_Z_SLICE_RE = re.compile(r"_z(\d+)\.(?:tif|tiff)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PFIB/pores4thought TIF 切片 → 训练 npy（流式 + 校验）")
    p.add_argument("--slice-dir", type=Path, required=True, help="切片根目录")
    p.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    p.add_argument("--patch-size", type=int, default=64)
    p.add_argument("--num-patches", type=int, default=512, help="随机子块数；论文 PFIB 子块量级常用 2400")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--recursive",
        action="store_true",
        help="递归扫描子树中的切片（默认仅扫 slice-dir 本层，减少误扫到无关 tif）",
    )
    p.add_argument(
        "--strict-z",
        action="store_true",
        help="要求 z 序号从 min 到 max 连续无缺失，否则报错（防漏层）",
    )
    p.add_argument(
        "--resize-out",
        type=int,
        default=None,
        help="裁块为 patch³ 后，用 scipy 最近邻(order=0) 放大到此边长；薄 Z 栈无法裁 64³ 时常用 28→64 以对齐论文 GAN",
    )
    p.add_argument(
        "--labels-j-npy",
        type=Path,
        default=None,
        help="OpenFOAM/论文对齐的 J 标签 [num_patches,7]；与 num_patches 行数必须一致",
    )
    p.add_argument(
        "--label-mode",
        choices=["proxy", "zeros"],
        default="proxy",
        help="未提供 --labels-j-npy 时：proxy 或 zeros",
    )
    return p.parse_args()


def _tifffile():
    try:
        import tifffile
    except Exception as exc:  # pragma: no cover
        raise ImportError("请安装 tifffile：pip install tifffile") from exc
    return tifffile


def _list_tif_paths(root: Path, *, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(root.rglob("*.tif")) + sorted(root.rglob("*.tiff"))
    return sorted(root.glob("*.tif")) + sorted(root.glob("*.tiff"))


def _pairs_from_paths(paths: list[Path]) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    for p in paths:
        if not p.is_file():
            continue
        m = _Z_SLICE_RE.search(p.name)
        if m:
            pairs.append((int(m.group(1)), p))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _resolve_slice_dir(slice_dir: Path, *, recursive: bool) -> tuple[Path, list[tuple[int, Path]], str]:
    """
    返回 (实际使用的目录, 切片列表, 说明)。
    若根目录无匹配切片，则尝试**一层**子目录中选切片数最多者。
    """
    root = slice_dir.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"切片目录不存在: {root}")

    paths = _list_tif_paths(root, recursive=recursive)
    pairs = _pairs_from_paths(paths)
    note = "slice-dir 本层"
    if not pairs and not recursive:
        best: tuple[int, list[tuple[int, Path]], Path] | None = None
        for ch in sorted(root.iterdir()):
            if not ch.is_dir():
                continue
            pp = _pairs_from_paths(_list_tif_paths(ch, recursive=False))
            if not pp:
                continue
            if best is None or len(pp) > best[0]:
                best = (len(pp), pp, ch)
        if best:
            pairs = best[1]
            root = best[2]
            note = f"自动下探子目录: {root.name}"

    if len(pairs) < 2:
        raise FileNotFoundError(
            f"未找到带 `_z数字.tif` 的切片：{slice_dir}\n"
            "请指向含切片的文件夹；若切片在子目录中可省略一层或使用 --recursive。"
        )
    return root, pairs, note


def _validate_pairs(pairs: list[tuple[int, Path]], *, strict_z: bool) -> dict:
    zs = [z for z, _ in pairs]
    dup = [z for z in set(zs) if zs.count(z) > 1]
    if dup:
        raise ValueError(f"存在重复 z 索引（同层多张）: 示例 {dup[:8]}")

    if strict_z:
        span = zs[-1] - zs[0] + 1
        if span != len(pairs):
            raise ValueError(
                f"--strict-z：z 应从 {zs[0]} 连续到 {zs[-1]} 共 {span} 层，实际文件 {len(pairs)}，"
                "说明缺层或多余文件。"
            )
        expected = list(range(zs[0], zs[-1] + 1))
        if [z for z, _ in pairs] != expected:
            raise ValueError("--strict-z：z 序号排序后与连续整数列不一致。")

    tifffile = _tifffile()
    hw0 = None
    shape_samples: dict[str, tuple[int, int]] = {}
    for label, idx in (("first", 0), ("mid", len(pairs) // 2), ("last", len(pairs) - 1)):
        path = pairs[idx][1]
        arr = np.asarray(tifffile.imread(str(path)))
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"切片须为 2D: {path} shape={arr.shape}")
        h, w = int(arr.shape[0]), int(arr.shape[1])
        shape_samples[label] = (h, w)
        if hw0 is None:
            hw0 = (h, w)
        elif (h, w) != hw0:
            raise ValueError(f"切片空间尺寸不一致: {path} {(h,w)} vs 参考 {hw0}")

    return {
        "z_min": zs[0],
        "z_max": zs[-1],
        "n_slices": len(pairs),
        "hw": list(hw0),
        "shape_check": shape_samples,
        "strict_z": strict_z,
    }


def _read_slice_window(path: Path, dy: int, dx: int, patch: int) -> np.ndarray:
    tifffile = _tifffile()
    arr = np.asarray(tifffile.imread(str(path)))
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return np.asarray(arr[dy : dy + patch, dx : dx + patch], dtype=np.float32)


def _normalize_three_phase(vol: np.ndarray) -> np.ndarray:
    v = np.asarray(vol, dtype=np.float32)
    uniq = np.unique(v)
    if uniq.size == 0:
        raise ValueError("空体数据")
    out = np.zeros_like(v, dtype=np.uint8)
    if np.isin(255, uniq):
        out[np.isclose(v, 255)] = 255
    out[np.isclose(v, 127) | np.isclose(v, 128)] = 128
    assigned = (out == 0) | (out == 128) | (out == 255)
    if not np.all(assigned):
        q1, q2 = np.quantile(v, [0.33, 0.66])
        out[:] = 0
        out[(v > q1) & (v <= q2)] = 128
        out[v > q2] = 255
    return out


def _random_crop_stream(
    pairs: list[tuple[int, Path]],
    *,
    patch: int,
    rng: np.random.Generator,
    z_len: int,
    h: int,
    w: int,
) -> tuple[np.ndarray, dict]:
    """返回 (patch 体素 uint8 路径前为 float 再 normalize), 裁窗元数据 dict)。"""
    if z_len < patch or h < patch or w < patch:
        raise ValueError(f"(Z,H,W)=({z_len},{h},{w}) 小于 patch={patch}")
    dz = int(rng.integers(0, z_len - patch + 1))
    dy = int(rng.integers(0, h - patch + 1))
    dx = int(rng.integers(0, w - patch + 1))
    slab = np.zeros((patch, patch, patch), dtype=np.float32)
    for k in range(patch):
        slab[k] = _read_slice_window(pairs[dz + k][1], dy, dx, patch)
    meta = {
        "z_start_logical": int(pairs[dz][0]),
        "z_start_index_in_stack": int(dz),
        "dy": int(dy),
        "dx": int(dx),
        "patch_size": int(patch),
        "z_end_logical": int(pairs[dz + patch - 1][0]),
    }
    return _normalize_three_phase(slab), meta


def _tpb_proxy(vol: np.ndarray) -> float:
    m = np.zeros_like(vol, dtype=np.int8)
    m[vol == 0] = 0
    m[vol == 128] = 1
    m[vol == 255] = 2
    return float(
        active_tpb_density_from_label_volume(
            m, pore_value=0, ion_value=2, ele_value=1, min_connected_fraction=0.01
        )
    )


def _resize_cubes_nearest(patches: np.ndarray, out: int) -> np.ndarray:
    """[N,D,H,W] uint8 三相体 → 各向 [N,out,out,out]，最近邻保持离散灰度。"""
    n, d, h, w = patches.shape
    if d != h or h != w:
        raise ValueError(f"resize 仅支持立方裁块，当前 {patches.shape}")
    if out == d:
        return patches
    if out < d:
        raise ValueError(f"--resize-out({out}) 须 >= 裁块边长({d})")
    zf = float(out) / float(d)
    out_arr = np.empty((n, out, out, out), dtype=np.uint8)
    for i in range(n):
        out_arr[i] = ndimage.zoom(patches[i].astype(np.float32), zoom=(zf, zf, zf), order=0).astype(np.uint8)
    return out_arr


def _labels_from_proxy_tpb(tpb_vec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    eta = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14], dtype=np.float32)
    base = 0.03 + 0.8 * tpb_vec[:, None]
    y = base * (1.0 + 1.8 * eta[None, :]) + rng.normal(0, 0.003, size=(tpb_vec.shape[0], 7))
    return np.clip(y, 0.0, None).astype(np.float32)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    used_dir, pairs, resolve_note = _resolve_slice_dir(Path(args.slice_dir), recursive=args.recursive)
    inv = _validate_pairs(pairs, strict_z=args.strict_z)
    inv["slice_dir_requested"] = str(Path(args.slice_dir).resolve())
    inv["slice_dir_used"] = str(used_dir)
    inv["resolve_note"] = resolve_note
    inv["recursive"] = args.recursive
    inv["filenames_head"] = [pairs[i][1].name for i in range(min(5, len(pairs)))]
    inv["filenames_tail"] = [pairs[i][1].name for i in range(max(0, len(pairs) - 5), len(pairs))]

    z_len = len(pairs)
    h, w = inv["hw"]
    patch = args.patch_size
    n = args.num_patches
    print(f"[INFO] 使用目录: {used_dir} ({resolve_note}) Z={z_len} H×W={h}×{w} patches={n}", file=sys.stderr)

    patches = np.zeros((n, patch, patch, patch), dtype=np.uint8)
    crop_meta: list[dict] = []
    step = max(1, n // 20)
    for i in range(n):
        vol_i, meta = _random_crop_stream(pairs, patch=patch, rng=rng, z_len=z_len, h=h, w=w)
        patches[i] = vol_i
        meta["patch_index"] = int(i)
        crop_meta.append(meta)
        if (i + 1) % step == 0 or i + 1 == n:
            print(f"[INFO] patch {i + 1}/{n}", file=sys.stderr)

    resize_out = args.resize_out
    if resize_out is not None:
        patches = _resize_cubes_nearest(patches, resize_out)
        print(
            f"[INFO] 已 --resize-out {resize_out}（裁块边长 {patch}，最近邻放大，volumes 形状 {patches.shape}）",
            file=sys.stderr,
        )
        for meta in crop_meta:
            meta["resize_out"] = int(resize_out)
    inv["patch_crop"] = int(patch)
    inv["resize_out"] = int(resize_out) if resize_out is not None else int(patch)
    inv["volumes_spatial_shape"] = [int(patches.shape[1]), int(patches.shape[2]), int(patches.shape[3])]
    (out_dir / "slice_inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=2), encoding="utf-8")

    (out_dir / "crop_manifest.json").write_text(
        json.dumps(crop_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    volumes = patches.astype(np.float32)[:, None, ...]
    tpb = np.asarray([_tpb_proxy(patches[i]) for i in range(n)], dtype=np.float32)[:, None]

    if args.labels_j_npy is not None:
        lj_path = Path(args.labels_j_npy).expanduser().resolve()
        if not lj_path.exists():
            raise FileNotFoundError(f"--labels-j-npy 不存在: {lj_path}")
        labels_j = np.load(str(lj_path)).astype(np.float32)
        if labels_j.shape != (n, 7):
            raise ValueError(
                f"labels_j 形状须为 ({n},7) 与 --num-patches 一致，当前 {labels_j.shape}。\n"
                "若你方数据完整，请检查：是否用了错误的标签文件、或 num_patches 与标签行数不一致。"
            )
        label_note = f"外部 OpenFOAM/论文对齐标签: {lj_path}"
    elif args.label_mode == "proxy":
        labels_j = _labels_from_proxy_tpb(tpb[:, 0], rng)
        label_note = "proxy_formula（非 OpenFOAM；未传 --labels-j-npy）"
    else:
        labels_j = np.zeros((n, 7), dtype=np.float32)
        label_note = "zeros 占位"

    np.save(out_dir / "volumes.npy", volumes)
    np.save(out_dir / "tpb.npy", tpb)
    np.save(out_dir / "labels_j.npy", labels_j)

    write_training_bundle_manifest(
        out_dir,
        volumes_shape=volumes.shape,
        labels_j_shape=labels_j.shape,
        tpb_shape=tpb.shape,
        data_provenance=f"stack_pores4thought_tifs: {resolve_note}; {label_note}",
        extra={
            "script": "scripts/stack_pores4thought_tifs_to_bundle.py",
            "slice_dir_used": str(used_dir),
            "num_patches": n,
            "patch_size": patch,
            "resize_out": resize_out,
            "seed": args.seed,
            "strict_z": args.strict_z,
            "recursive": args.recursive,
            "crop_manifest": "crop_manifest.json",
        },
    )

    summary = "\n".join(
        [
            "=== 转换完成 ===",
            f"out: {out_dir}",
            f"slice_inventory: {out_dir / 'slice_inventory.json'}",
            f"volumes: {volumes.shape}",
            f"tpb: {tpb.shape}",
            f"labels_j: {labels_j.shape} ({label_note})",
        ]
    )
    (out_dir / "stack_convert_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
