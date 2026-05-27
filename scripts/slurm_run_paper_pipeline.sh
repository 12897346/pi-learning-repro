#!/bin/bash
# 用法：
# sbatch scripts/slurm_run_paper_pipeline.sh
# 可选覆盖（提交时）：
# sbatch --export=ALL,PROJECT_DIR=/path/pi-learning,PYTHON_BIN=python \
#   scripts/slurm_run_paper_pipeline.sh
#
# === 申请 GPU（与分区是否单独无关）===
# 若 sinfo 只有 ksagnormal01 一类分区：仍用本脚本 -p，靠 #SBATCH --gres=gpu:1（或中心要求的
# --gpus-per-node=1）申请 GPU；无需再「换分区」。
# 若中心另有独立 GPU 分区：再用 sbatch -p <gpu分区名> 覆盖本脚本里的 -p。
# 查看节点上 GRES：sinfo -Nel -o '%N %G %T %C' 或 scontrol show node <nodename>
# 交互调试：salloc -p ksagnormal01 --gres=gpu:1 -t 01:00:00
#
# 内存/CPU 上限：集群 DefMemPerCPU 要求「每核内存」不超过上限。若提交报 memory/CPU，
# 用命令行覆盖，例如：sbatch --mem-per-cpu=3G ... 或增大 --cpus-per-task 再配 --mem。
# 可查：scontrol show partition ksagnormal01 | grep -iE 'DefMem|MaxMem'
#
# 数据目录 DATA_DIR：须含 volumes.npy / labels_j.npy / tpb.npy；不要用文档里的 /path/to/... 占位。
# 无数据时先在仓库根执行（登录节点 `python` 常为 2.7，勿用 bare python）:
#   bash scripts/build_fallback_training_data.sh --out-dir data/processed_fallback
#   或: python3 scripts/build_fallback_training_data.py --out-dir data/processed_fallback
# 再提交: export DATA_DIR="$PROJECT_DIR/data/processed_fallback"（或不 export，脚本默认即该相对路径）
#
# 本校/本账号家目录示例（请按实际仓库路径修改）：
#   PROJECT_DIR=/public/home/student_1/pi-learning
# 默认已是「工程快速复现」：fast TPB、缩小 epoch/PSO、PSO 结束离线 strict TPB 一次。
# 全流程 + 表面图示例：
#   sbatch --time=72:00:00 --export=ALL,EXPORT_PSO_SURFACE=iso,EXPORT_SURFACE_GAN_ARCH=tiny,STRICT_NO_PROXY=0 \
#     scripts/slurm_run_paper_pipeline.sh
# 更激进试跑：export PIPELINE_QUICK=1
#

#SBATCH -J pi_repro
# 与 sinfo 中 PARTITION 一致（你处仅有 ksagnormal01 时保持本行即可）
#SBATCH -p ksagnormal01
# GPU 申请：若集群不认 --gres，可改为命令行 sbatch --gpus-per-node=1，或把下行改成：
# #SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
# 默认按「每核不超过常见 DefMemPerCPU」配内存；需要更大总内存时优先加核数再调高 --mem-per-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=48:00:00
#SBATCH -o logs/pi_repro_%j.out
#SBATCH -e logs/pi_repro_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
# 随仓库上传的默认路径（见 scripts/cluster_env.sh）；本地未放该文件则跳过
if [[ -f "$PROJECT_DIR/scripts/cluster_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$PROJECT_DIR/scripts/cluster_env.sh"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/processed_fallback}"
case "$DATA_DIR" in
  *"/path/to/"*|*"your/npy_dir"*|*"your_npy_dir"*)
    echo "[WARN] DATA_DIR 疑似文档占位路径（当前: $DATA_DIR），已改为 data/processed_fallback"
    DATA_DIR="data/processed_fallback"
    ;;
esac
OUT_DIR="${OUT_DIR:-outputs}"
# 相对路径统一到 PROJECT_DIR 下绝对路径（与 cd 后一致），减少 outputs 写到错误 cwd、pipeline.log 为 0 的排查成本
case "$DATA_DIR" in
  /*) ;;
  *) DATA_DIR="$PROJECT_DIR/$DATA_DIR" ;;
esac
case "$OUT_DIR" in
  /*) ;;
  *) OUT_DIR="$PROJECT_DIR/$OUT_DIR" ;;
esac
# phys 共用轮数（默认 50，见 paper_repro.phys_train_epochs）
# 论文 phys-CNN 混合集（真实+GAN）：export HYBRID_GAN_SAMPLES=3000
# 若 GAN 子集已有 OpenFOAM 标签：export HYBRID_GAN_LABELS_J=/path/to/labels_j.npy
HYBRID_GAN_SAMPLES="${HYBRID_GAN_SAMPLES:-0}"
HYBRID_GAN_LABELS_J="${HYBRID_GAN_LABELS_J:-}"
NORMAL_EPOCHS="${NORMAL_EPOCHS:-200}"
PHYSICS_EPOCHS="${PHYSICS_EPOCHS:-80}"
# paper 通道 GAN 在 24GB 上 batch>4 易 OOM；tiny 可用 8
BATCH_SIZE="${BATCH_SIZE:-4}"
GAN_FORCE_TINY="${GAN_FORCE_TINY:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_SAMPLES_FIG="${NUM_SAMPLES_FIG:-200}"
PHYS_EPOCHS="${PHYS_EPOCHS:-50}"
# PSO 显存不足时减小，例如 8 或 16（传给 run_paper_repro_pipeline --pso-eval-microbatch）
PSO_EVAL_MICROBATCH="${PSO_EVAL_MICROBATCH:-64}"
PSO_PARTICLES="${PSO_PARTICLES:-200}"
PSO_ITERS="${PSO_ITERS:-50}"
PSO_PRIOR_SAMPLES="${PSO_PRIOR_SAMPLES:-400}"
GAN_TPB_MODE="${GAN_TPB_MODE:-fast}"
PHYS_CONNECTIVITY_MODE="${PHYS_CONNECTIVITY_MODE:-fast}"
PSO_CONNECTIVITY_MODE="${PSO_CONNECTIVITY_MODE:-fast}"
# 1=PSO 结束后对 best_latent 算一次 strict TPB；0=跳过
EVAL_STRICT_TPB_AT_END="${EVAL_STRICT_TPB_AT_END:-1}"
# 论文：gbest_J 连续约 200 步「提升 < tol」则收敛；patience 会随 PSO_ITERS 缩放且不超过 iters；0=关闭
PSO_PLATEAU_PATIENCE="${PSO_PLATEAU_PATIENCE:-200}"
# 判定「有实质提升」的最小 gbest_J 增量（mA/cm²）；默认 1e-3，非要求完全不变
PSO_PLATEAU_TOL="${PSO_PLATEAU_TOL:-1e-3}"
# GAN：正文在 Wasserstein steady 时停（1000/300 为常见上限）；滑动 w_dist 判稳早停；patience=0 关闭
GAN_WDIST_STABLE_WINDOW="${GAN_WDIST_STABLE_WINDOW:-7}"
GAN_WDIST_STABLE_STD_TOL="${GAN_WDIST_STABLE_STD_TOL:-0.03}"
GAN_WDIST_STABLE_PATIENCE="${GAN_WDIST_STABLE_PATIENCE:-6}"
# phys surrogate：默认开启 --paper-early-stop；export PHYS_DISABLE_EARLY_STOP=1 则跑满 yaml epoch
PHYS_DISABLE_EARLY_STOP="${PHYS_DISABLE_EARLY_STOP:-0}"
# GAN 中间切片：默认不写 CLI，由 configs/paper_params.yaml 的 paper_repro.gan_preview_every 控制；若需覆盖再 export GAN_PREVIEW_EVERY=50（>0 才传 CLI）
GAN_PREVIEW_EVERY="${GAN_PREVIEW_EVERY:-0}"
# physics GAN：第二判据 c_loss 窗口判稳；export GAN_PHYSICS_SKIP_CLOSS=1 则仅 w_dist
GAN_PHYSICS_SKIP_CLOSS="${GAN_PHYSICS_SKIP_CLOSS:-0}"
# 可选：export GAN_PHYSICS_CLOSS_STD_TOL=0.06 覆盖默认阈值
GAN_PHYSICS_CLOSS_STD_TOL="${GAN_PHYSICS_CLOSS_STD_TOL:-}"
# PSO 最优 3D 表面图：export EXPORT_PSO_SURFACE=iso 或 both（全流程最后一步；默认 off）
EXPORT_PSO_SURFACE="${EXPORT_PSO_SURFACE:-off}"
# 表面 GAN 通道：tiny / paper / auto（yaml debug_tiny=false 且 ckpt 为 tiny 时用 tiny）
EXPORT_SURFACE_GAN_ARCH="${EXPORT_SURFACE_GAN_ARCH:-tiny}"
# 可选：表面步骤里 GAN/Phys 前向设备；默认不传则与 --device cuda 一致
EXPORT_SURFACE_DEVICE="${EXPORT_SURFACE_DEVICE:-}"
EXPORT_SURFACE_BACKEND="${EXPORT_SURFACE_BACKEND:-matplotlib}"
EXPORT_SURFACE_MPL_DOWNSAMPLE="${EXPORT_SURFACE_MPL_DOWNSAMPLE:-2}"
STRICT_NO_PROXY="${STRICT_NO_PROXY:-1}"

# OpenFOAM 三组输入（可留空；留空则自动跳过真场图）
OPENFOAM_LOW="${OPENFOAM_LOW:-}"
OPENFOAM_INTERMEDIATE="${OPENFOAM_INTERMEDIATE:-}"
OPENFOAM_GLOBAL="${OPENFOAM_GLOBAL:-}"
OPENFOAM_PHASE_COL="${OPENFOAM_PHASE_COL:-phase}"
OPENFOAM_PHI_COL="${OPENFOAM_PHI_COL:-phi_ion}"
OPENFOAM_X_COL="${OPENFOAM_X_COL:-x}"
OPENFOAM_Y_COL="${OPENFOAM_Y_COL:-y}"
OPENFOAM_Z_COL="${OPENFOAM_Z_COL:-z}"

# 一键快速预设：export PIPELINE_QUICK=1（强制写入下列试跑参数；需自定义组合请勿设 QUICK，改用手动 export 各变量）
PIPELINE_QUICK="${PIPELINE_QUICK:-0}"
if [[ "$PIPELINE_QUICK" == "1" ]]; then
  NORMAL_EPOCHS=200
  PHYSICS_EPOCHS=80
  PSO_PARTICLES=200
  PSO_ITERS=50
  PSO_PLATEAU_PATIENCE=15
  PSO_PRIOR_SAMPLES=400
  PSO_EVAL_MICROBATCH=64
  BATCH_SIZE=8
  NUM_WORKERS=8
  NUM_SAMPLES_FIG=200
  GAN_TPB_MODE=fast
  PHYS_EPOCHS=50
  GAN_PREVIEW_EVERY=0
  echo "[INFO] PIPELINE_QUICK=1：缩小规模 + 8 核 DataLoader + GAN_TPB_MODE=fast"
fi

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

echo "[INFO] start: $(date '+%F %T')"
echo "[INFO] project: $PROJECT_DIR"
echo "[INFO] python: $PYTHON_BIN"
echo "[INFO] data_dir: $DATA_DIR"
echo "[INFO] out_dir (absolute): $OUT_DIR"
echo "[INFO] data_dir (absolute): $DATA_DIR"
echo "[INFO] PSO: particles=$PSO_PARTICLES iters=$PSO_ITERS prior_samples=$PSO_PRIOR_SAMPLES plateau_patience=$PSO_PLATEAU_PATIENCE plateau_tol=$PSO_PLATEAU_TOL eval_microbatch=$PSO_EVAL_MICROBATCH"
echo "[INFO] GAN: force_tiny=$GAN_FORCE_TINY tpb_mode=$GAN_TPB_MODE w_dist early-stop: window=$GAN_WDIST_STABLE_WINDOW std_tol=$GAN_WDIST_STABLE_STD_TOL patience=$GAN_WDIST_STABLE_PATIENCE (max_epochs normal=$NORMAL_EPOCHS physics=$PHYSICS_EPOCHS) batch=$BATCH_SIZE num_workers=$NUM_WORKERS"
echo "[INFO] phys surrogate: run_paper 默认 --paper-early-stop；PHYS_DISABLE_EARLY_STOP=$PHYS_DISABLE_EARLY_STOP（1=关闭） GAN_PREVIEW_EVERY=$GAN_PREVIEW_EVERY GAN_PHYSICS_SKIP_CLOSS=$GAN_PHYSICS_SKIP_CLOSS EXPORT_PSO_SURFACE=$EXPORT_PSO_SURFACE EXPORT_SURFACE_BACKEND=$EXPORT_SURFACE_BACKEND EXPORT_SURFACE_GAN_ARCH=$EXPORT_SURFACE_GAN_ARCH"

# 计算节点上需加载含 CUDA 的 Python；优先用环境变量，否则尝试常见 miniforge 路径
CONDA_ENV="${CONDA_ENV:-zxh}"
if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
elif [[ -f "$HOME/miniforge3export/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniforge3export/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -f "/public/home/student_1/miniforge3export/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/public/home/student_1/miniforge3export/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -f "/public/home/student_1/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/public/home/student_1/miniforge3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
# 登录/计算节点 PATH 里常有系统 python2；若 PYTHON_BIN 仍是裸的 python/python3，改用当前 conda 解释器
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  case "${PYTHON_BIN}" in
    python|python3) PYTHON_BIN="${CONDA_PREFIX}/bin/python" ;;
  esac
fi
# 若站点要求 module 加载 CUDA，可取消下行注释并按版本修改
# module load cuda/12.1

echo "[INFO] SLURM: JOB_PARTITION=${SLURM_JOB_PARTITION:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L || echo "[WARN] nvidia-smi 不可用（可能未分配到 GPU 或未装驱动）"

$PYTHON_BIN scripts/check_env.py --require-cuda

# 多进程 DataLoader 时避免 8 worker × 多线程 BLAS 过度抢占 CPU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PIPE_PHYS=()
if [[ -n "$PHYS_EPOCHS" ]]; then
  PIPE_PHYS=(--phys-epochs "$PHYS_EPOCHS")
fi
PIPE_HYBRID=()
if [[ "${HYBRID_GAN_SAMPLES:-0}" -gt 0 ]]; then
  PIPE_HYBRID=(--phys-cnn-hybrid-gan-samples "$HYBRID_GAN_SAMPLES")
  if [[ -n "${HYBRID_GAN_LABELS_J:-}" ]]; then
    PIPE_HYBRID+=(--hybrid-gan-labels-j-npy "$HYBRID_GAN_LABELS_J")
  fi
fi

# run_paper 默认已传 train_phys 的 --paper-early-stop；此处仅在显式关闭时追加 --phys-disable-early-stop
PIPE_PAPER_PHYS=()
if [[ "$PHYS_DISABLE_EARLY_STOP" == "1" ]]; then
  PIPE_PAPER_PHYS=(--phys-disable-early-stop)
fi
PIPE_GAN_PREVIEW=()
if [[ "${GAN_PREVIEW_EVERY:-0}" -gt 0 ]]; then
  PIPE_GAN_PREVIEW=(--gan-preview-every "$GAN_PREVIEW_EVERY")
fi
PIPE_GAN_PHYS=()
if [[ "$GAN_PHYSICS_SKIP_CLOSS" == "1" ]]; then
  PIPE_GAN_PHYS+=(--gan-physics-skip-closs-stable)
fi
if [[ -n "${GAN_PHYSICS_CLOSS_STD_TOL:-}" ]]; then
  PIPE_GAN_PHYS+=(--gan-physics-closs-std-tol "$GAN_PHYSICS_CLOSS_STD_TOL")
fi

PIPE_EXPORT_SURF=()
if [[ "${EXPORT_PSO_SURFACE}" != "off" ]]; then
  PIPE_EXPORT_SURF=(--export-pso-surface "$EXPORT_PSO_SURFACE")
fi
PIPE_EXPORT_SURF_DEV=()
if [[ -n "${EXPORT_SURFACE_DEVICE:-}" ]]; then
  PIPE_EXPORT_SURF_DEV=(--export-surface-device "$EXPORT_SURFACE_DEVICE")
fi
PIPE_EXPORT_SURF_BACKEND=(--export-surface-backend "$EXPORT_SURFACE_BACKEND")
PIPE_EXPORT_SURF_MPL=(--export-surface-mpl-downsample "$EXPORT_SURFACE_MPL_DOWNSAMPLE")
PIPE_EXPORT_SURF_ARCH=(--export-surface-gan-arch "$EXPORT_SURFACE_GAN_ARCH")
PIPE_GAN_TPB=(--gan-tpb-mode "$GAN_TPB_MODE")
PIPE_GAN_TINY=()
if [[ "${GAN_FORCE_TINY:-1}" == "1" ]]; then
  PIPE_GAN_TINY=(--gan-force-tiny)
else
  PIPE_GAN_TINY=(--no-gan-force-tiny)
fi
PIPE_CONN=(--phys-connectivity-mode "$PHYS_CONNECTIVITY_MODE" --pso-connectivity-mode "$PSO_CONNECTIVITY_MODE")
PIPE_PSO_PRIOR=()
if [[ "${PSO_PRIOR_SAMPLES:-0}" -gt 0 ]]; then
  PIPE_PSO_PRIOR=(--pso-prior-samples "$PSO_PRIOR_SAMPLES")
fi
PIPE_PSO_PLATEAU=()
if [[ "${PSO_PLATEAU_PATIENCE:-0}" -gt 0 ]]; then
  PIPE_PSO_PLATEAU=(--pso-plateau-patience "$PSO_PLATEAU_PATIENCE" --pso-plateau-tol "$PSO_PLATEAU_TOL")
fi
PIPE_EVAL_STRICT=()
if [[ "${EVAL_STRICT_TPB_AT_END:-1}" == "0" ]]; then
  PIPE_EVAL_STRICT=(--no-eval-strict-tpb-at-end)
fi

if [[ -n "$OPENFOAM_LOW" && -n "$OPENFOAM_INTERMEDIATE" && -n "$OPENFOAM_GLOBAL" ]]; then
  echo "[INFO] OpenFOAM inputs detected, will include true-field figures"
  # set -u 下旧版 bash 会把空数组 "${arr[@]}" 判为未绑定；用 + 形式兼容
  $PYTHON_BIN scripts/run_paper_repro_pipeline.py \
    --device cuda \
    --data-dir "$DATA_DIR" \
    ${PIPE_PHYS[@]+"${PIPE_PHYS[@]}"} \
    ${PIPE_HYBRID[@]+"${PIPE_HYBRID[@]}"} \
    ${PIPE_PAPER_PHYS[@]+"${PIPE_PAPER_PHYS[@]}"} \
    ${PIPE_GAN_PREVIEW[@]+"${PIPE_GAN_PREVIEW[@]}"} \
    ${PIPE_GAN_PHYS[@]+"${PIPE_GAN_PHYS[@]}"} \
    ${PIPE_EXPORT_SURF[@]+"${PIPE_EXPORT_SURF[@]}"} \
    ${PIPE_EXPORT_SURF_DEV[@]+"${PIPE_EXPORT_SURF_DEV[@]}"} \
    ${PIPE_EXPORT_SURF_BACKEND[@]+"${PIPE_EXPORT_SURF_BACKEND[@]}"} \
    ${PIPE_EXPORT_SURF_MPL[@]+"${PIPE_EXPORT_SURF_MPL[@]}"} \
    ${PIPE_EXPORT_SURF_ARCH[@]+"${PIPE_EXPORT_SURF_ARCH[@]}"} \
    ${PIPE_GAN_TPB[@]+"${PIPE_GAN_TPB[@]}"} \
    ${PIPE_GAN_TINY[@]+"${PIPE_GAN_TINY[@]}"} \
    ${PIPE_CONN[@]+"${PIPE_CONN[@]}"} \
    ${PIPE_PSO_PRIOR[@]+"${PIPE_PSO_PRIOR[@]}"} \
    ${PIPE_PSO_PLATEAU[@]+"${PIPE_PSO_PLATEAU[@]}"} \
    ${PIPE_EVAL_STRICT[@]+"${PIPE_EVAL_STRICT[@]}"} \
    --normal-epochs "$NORMAL_EPOCHS" \
    --physics-epochs "$PHYSICS_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --num-samples-fig "$NUM_SAMPLES_FIG" \
    --pso-eval-microbatch "$PSO_EVAL_MICROBATCH" \
    --pso-particles "$PSO_PARTICLES" \
    --pso-iters "$PSO_ITERS" \
    --pso-plateau-patience "$PSO_PLATEAU_PATIENCE" \
    --gan-wdist-stable-window "$GAN_WDIST_STABLE_WINDOW" \
    --gan-wdist-stable-std-tol "$GAN_WDIST_STABLE_STD_TOL" \
    --gan-wdist-stable-patience "$GAN_WDIST_STABLE_PATIENCE" \
    --out-dir "$OUT_DIR" \
    --openfoam-low "$OPENFOAM_LOW" \
    --openfoam-intermediate "$OPENFOAM_INTERMEDIATE" \
    --openfoam-global "$OPENFOAM_GLOBAL" \
    --openfoam-phase-col "$OPENFOAM_PHASE_COL" \
    --openfoam-phi-col "$OPENFOAM_PHI_COL" \
    --openfoam-x-col "$OPENFOAM_X_COL" \
    --openfoam-y-col "$OPENFOAM_Y_COL" \
    --openfoam-z-col "$OPENFOAM_Z_COL" \
    $( [[ "$STRICT_NO_PROXY" == "1" ]] && echo "--strict-no-proxy" )
else
  if [[ "$STRICT_NO_PROXY" == "1" ]]; then
    echo "[ERROR] STRICT_NO_PROXY=1 时必须提供 OPENFOAM_LOW/INTERMEDIATE/GLOBAL"
    exit 9
  fi
  echo "[INFO] OpenFOAM inputs not provided, run pipeline with --skip-openfoam"
  # set -u 下旧版 bash 会把空数组 "${arr[@]}" 判为未绑定；用 + 形式兼容
  $PYTHON_BIN scripts/run_paper_repro_pipeline.py \
    --device cuda \
    --data-dir "$DATA_DIR" \
    ${PIPE_PHYS[@]+"${PIPE_PHYS[@]}"} \
    ${PIPE_HYBRID[@]+"${PIPE_HYBRID[@]}"} \
    ${PIPE_PAPER_PHYS[@]+"${PIPE_PAPER_PHYS[@]}"} \
    ${PIPE_GAN_PREVIEW[@]+"${PIPE_GAN_PREVIEW[@]}"} \
    ${PIPE_GAN_PHYS[@]+"${PIPE_GAN_PHYS[@]}"} \
    ${PIPE_EXPORT_SURF[@]+"${PIPE_EXPORT_SURF[@]}"} \
    ${PIPE_EXPORT_SURF_DEV[@]+"${PIPE_EXPORT_SURF_DEV[@]}"} \
    ${PIPE_EXPORT_SURF_BACKEND[@]+"${PIPE_EXPORT_SURF_BACKEND[@]}"} \
    ${PIPE_EXPORT_SURF_MPL[@]+"${PIPE_EXPORT_SURF_MPL[@]}"} \
    ${PIPE_EXPORT_SURF_ARCH[@]+"${PIPE_EXPORT_SURF_ARCH[@]}"} \
    ${PIPE_GAN_TPB[@]+"${PIPE_GAN_TPB[@]}"} \
    ${PIPE_GAN_TINY[@]+"${PIPE_GAN_TINY[@]}"} \
    ${PIPE_CONN[@]+"${PIPE_CONN[@]}"} \
    ${PIPE_PSO_PRIOR[@]+"${PIPE_PSO_PRIOR[@]}"} \
    ${PIPE_PSO_PLATEAU[@]+"${PIPE_PSO_PLATEAU[@]}"} \
    ${PIPE_EVAL_STRICT[@]+"${PIPE_EVAL_STRICT[@]}"} \
    --normal-epochs "$NORMAL_EPOCHS" \
    --physics-epochs "$PHYSICS_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --num-samples-fig "$NUM_SAMPLES_FIG" \
    --pso-eval-microbatch "$PSO_EVAL_MICROBATCH" \
    --pso-particles "$PSO_PARTICLES" \
    --pso-iters "$PSO_ITERS" \
    --pso-plateau-patience "$PSO_PLATEAU_PATIENCE" \
    --gan-wdist-stable-window "$GAN_WDIST_STABLE_WINDOW" \
    --gan-wdist-stable-std-tol "$GAN_WDIST_STABLE_STD_TOL" \
    --gan-wdist-stable-patience "$GAN_WDIST_STABLE_PATIENCE" \
    --out-dir "$OUT_DIR" \
    --skip-openfoam
fi

echo "[INFO] done: $(date '+%F %T')"
