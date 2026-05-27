#!/usr/bin/env bash
# 登录节点上 `python` 常为 2.x；本包装优先用 python3，避免 SyntaxError。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

pick_py() {
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "${CONDA_PREFIX}/bin/python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  echo "[ERROR] 未找到可用的 Python 3。请先:" >&2
  echo "  source .../miniforge3export/etc/profile.d/conda.sh && conda activate zxh" >&2
  echo "再运行本脚本；或安装/加载含 python3 的模块。" >&2
  exit 1
}

PY="$(pick_py)"
exec "$PY" "$HERE/build_fallback_training_data.py" "$@"
