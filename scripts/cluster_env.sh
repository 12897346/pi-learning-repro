#!/bin/bash
# 集群默认环境（与仓库一起上传到 /public/home/student_1/pi-learning 即可用）。
# 规则：仅当变量「尚未设置」时才写入，你在 sbatch --export=... 里已传的值不会被覆盖。
#
# 训练用的 DATA_DIR 须含 volumes.npy / labels_j.npy / tpb.npy。
# 若该目录还不存在，在登录节点执行：
#   bash scripts/build_anode_patches_cluster.sh

: "${DATA_DIR:=/public/home/student_1/pi-learning/data/processed_anode_patches}"
export DATA_DIR

: "${OUT_DIR:=/public/home/student_1/pi-learning/outputs}"
export OUT_DIR

# 无 OpenFOAM 真场图时必须为 0，否则 pipeline 会要求 OPENFOAM_* 输入
: "${STRICT_NO_PROXY:=0}"
export STRICT_NO_PROXY

# synth_015 仅 Z=29，无法裁 64³；先裁 28³ 再最近邻放大到 64³（见 stack 的 --resize-out）。
# 厚切片全卷可改：PI_LEARNING_PATCH_SIZE=64，且 export PI_LEARNING_RESIZE_OUT=0 关闭放大。
: "${PI_LEARNING_PATCH_SIZE:=28}"
export PI_LEARNING_PATCH_SIZE
: "${PI_LEARNING_RESIZE_OUT:=64}"
export PI_LEARNING_RESIZE_OUT

# 仅用于登录节点跑 stack 脚本（Slurm 训练不读）。
# 须能扫到文件名形如「任意前缀_z整数.tif」的 PFIB 切片；若实际在更深子目录，build 脚本默认
# 已加 --recursive（同一树下若有多套体积会 z 重复报错，此时 export PI_LEARNING_SLICE_RECURSIVE=0
# 并把本变量指到「单套」切片所在文件夹）。
# 定位示例：find /public/home/student_1/pi-learning--data -iname '*_z*.tif' 2>/dev/null | head
# 当前账号数据：PFIB 切片在 synth_015 目录根（非 anode-segmented 下）；换样本时改此路径。
: "${PI_LEARNING_RAW_ANODE:=/public/home/student_1/pi-learning--data/synth_015}"
export PI_LEARNING_RAW_ANODE
