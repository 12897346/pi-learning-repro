#!/bin/bash
# 仅补跑 PSO 最优结构 → 2×3 表面/学习曲线图（matplotlib，无 OSMesa）。
#
# 前提：$OUT_DIR/forward_design/best_latent.npy 与 $OUT_DIR/gan_fallback/generator_fallback.pth 已存在。
#
# 用法（在仓库根目录）：
#   cd /public/home/student_1/pi-learning
#   sbatch scripts/slurm_export_pso_surface.sh
#
# 或显式覆盖路径 / tiny 网：
#   sbatch --export=ALL,PROJECT_DIR=/public/home/student_1/pi-learning,OUT_DIR=/public/home/student_1/pi-learning/outputs,GAN_FORCE_TINY=1 \
#     scripts/slurm_export_pso_surface.sh

#SBATCH -J pi_surface
#SBATCH -p ksagnormal01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH -o logs/pi_surface_%j.out
#SBATCH -e logs/pi_surface_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
if [[ -f "$PROJECT_DIR/scripts/cluster_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$PROJECT_DIR/scripts/cluster_env.sh"
fi

OUT_DIR="${OUT_DIR:-outputs}"
DATA_DIR="${DATA_DIR:-data/processed_fallback}"
VIEWS="${VIEWS:-iso}"
SURFACE_BACKEND="${SURFACE_BACKEND:-matplotlib}"
MPL_DOWNSAMPLE="${MPL_DOWNSAMPLE:-2}"
GAN_FORCE_TINY="${GAN_FORCE_TINY:-1}"
GAN_FORCE_PAPER="${GAN_FORCE_PAPER:-0}"
DEVICE="${DEVICE:-cuda}"
CONDA_ENV="${CONDA_ENV:-zxh}"

case "$OUT_DIR" in
  /*) ;;
  *) OUT_DIR="$PROJECT_DIR/$OUT_DIR" ;;
esac
case "$DATA_DIR" in
  /*) ;;
  *) DATA_DIR="$PROJECT_DIR/$DATA_DIR" ;;
esac

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
elif [[ -f "/public/home/student_1/miniforge3export/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/public/home/student_1/miniforge3export/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -f "$HOME/miniforge3export/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniforge3export/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  case "${PYTHON_BIN}" in
    python|python3) PYTHON_BIN="${CONDA_PREFIX}/bin/python" ;;
  esac
fi

echo "[INFO] start: $(date '+%F %T')"
echo "[INFO] PROJECT_DIR=$PROJECT_DIR OUT_DIR=$OUT_DIR DEVICE=$DEVICE"
echo "[INFO] SLURM: PARTITION=${SLURM_JOB_PARTITION:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "[WARN] nvidia-smi 不可用"

BEST_LATENT="$OUT_DIR/forward_design/best_latent.npy"
GEN_CKPT="$OUT_DIR/gan_fallback/generator_fallback.pth"
SURFACE_OUT="$OUT_DIR/paper_figures/pso_best_surface"
PHYS_DNN="$OUT_DIR/phys_models/phys_dnn.pth"
PHYS_CNN="$OUT_DIR/phys_models/phys_cnn.pth"

for f in "$BEST_LATENT" "$GEN_CKPT"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] 缺少: $f（请先跑完 PSO / 流水线 forward_design + gan_fallback）"
    exit 2
  fi
done

ARCH_FLAGS=()
if [[ "$GAN_FORCE_TINY" == "1" ]]; then
  ARCH_FLAGS+=(--gan-force-tiny)
elif [[ "$GAN_FORCE_PAPER" == "1" ]]; then
  ARCH_FLAGS+=(--gan-force-paper)
fi

$PYTHON_BIN scripts/export_voxel_surface_figure.py \
  --best-latent-npy "$BEST_LATENT" \
  --gen-ckpt "$GEN_CKPT" \
  --gan-config "$PROJECT_DIR/configs/paper_params.yaml" \
  --device "$DEVICE" \
  --phys-dnn-ckpt "$PHYS_DNN" \
  --phys-cnn-ckpt "$PHYS_CNN" \
  --reference-data-dir "$DATA_DIR" \
  --out-dir "$SURFACE_OUT" \
  --basename pso_best_surface \
  --views "$VIEWS" \
  --surface-backend "$SURFACE_BACKEND" \
  --mpl-downsample "$MPL_DOWNSAMPLE" \
  "${ARCH_FLAGS[@]+"${ARCH_FLAGS[@]}"}"

echo "[INFO] done: $(date '+%F %T')"
echo "[INFO] PNG: $SURFACE_OUT/pso_best_surface_iso.png"
