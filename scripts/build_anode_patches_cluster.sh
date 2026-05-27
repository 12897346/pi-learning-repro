#!/bin/bash
# 在登录节点从原始 TIF 生成训练用 npy（默认路径见 scripts/cluster_env.sh）。
# 用法：
#   bash scripts/build_anode_patches_cluster.sh
#   NUM_PATCHES=500 SEED=1 bash scripts/build_anode_patches_cluster.sh
#   bash scripts/build_anode_patches_cluster.sh /path/to/out_dir
#
# 切片文件名须匹配 *_z数字.tif（见 stack_pores4thought_tifs_to_bundle.py）。
# 默认递归扫描子目录（PI_LEARNING_SLICE_RECURSIVE=0 可关闭，避免同一树下多卷混扫）。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# 若与 slurm 相同，可 source 默认 RAW
if [[ -f "$ROOT/scripts/cluster_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/scripts/cluster_env.sh"
fi

RAW="${PI_LEARNING_RAW_ANODE:?请设置 PI_LEARNING_RAW_ANODE 或在 cluster_env.sh 中配置}"
OUT="${1:-${DATA_DIR:-$ROOT/data/processed_anode_patches}}"
N="${NUM_PATCHES:-2400}"
SEED="${SEED:-42}"

echo "[INFO] ROOT=$ROOT"
echo "[INFO] RAW=$RAW"
echo "[INFO] OUT=$OUT"
echo "[INFO] NUM_PATCHES=$N SEED=$SEED"

# 薄 Z 栈：cluster_env 常设 PI_LEARNING_PATCH_SIZE=28、PI_LEARNING_RESIZE_OUT=64（对齐 GAN 64³）
PATCH="${PI_LEARNING_PATCH_SIZE:-64}"
RESIZE_ARGS=()
if [[ -n "${PI_LEARNING_RESIZE_OUT:-}" && "${PI_LEARNING_RESIZE_OUT}" != "0" ]]; then
  RESIZE_ARGS=(--resize-out "$PI_LEARNING_RESIZE_OUT")
fi
echo "[INFO] PATCH_SIZE=$PATCH PI_LEARNING_RESIZE_OUT=${PI_LEARNING_RESIZE_OUT:-（未设置，不放大）}"

REC_ARGS=()
if [[ "${PI_LEARNING_SLICE_RECURSIVE:-1}" != "0" ]]; then
  REC_ARGS=(--recursive)
  echo "[INFO] 递归扫描切片（PI_LEARNING_SLICE_RECURSIVE=${PI_LEARNING_SLICE_RECURSIVE:-1}，置 0 可关闭）"
fi

PYTHON="${PYTHON_BIN:-python3}"
cd "$ROOT"
exec "$PYTHON" scripts/stack_pores4thought_tifs_to_bundle.py \
  --slice-dir "$RAW" \
  --out-dir "$OUT" \
  --num-patches "$N" \
  --seed "$SEED" \
  --patch-size "$PATCH" \
  "${RESIZE_ARGS[@]}" \
  "${REC_ARGS[@]}" \
  --strict-z
