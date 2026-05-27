import sys
import argparse
from importlib import import_module


def _safe_version(pkg_name: str) -> str:
    try:
        module = import_module(pkg_name)
        return getattr(module, "__version__", "unknown")
    except Exception as exc:
        return f"NOT_INSTALLED ({exc.__class__.__name__})"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pi-learning 环境检查")
    p.add_argument(
        "--require-cuda",
        action="store_true",
        help="要求 CUDA 版 PyTorch 且 GPU 可用；不满足时返回非0退出码",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("=== pi-learning 环境检查 ===")
    print(f"Python: {sys.version.split()[0]}")

    np_ver = _safe_version("numpy")
    pd_ver = _safe_version("pandas")
    sk_ver = _safe_version("sklearn")
    torch_ver = _safe_version("torch")

    print(f"NumPy: {np_ver}")
    print(f"Pandas: {pd_ver}")
    print(f"scikit-learn: {sk_ver}")
    print(f"PyTorch: {torch_ver}")

    try:
        import torch

        print(f"PyTorch CUDA build: {torch.version.cuda}")
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA available: {cuda_ok}")
        if cuda_ok:
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device[0]: {torch.cuda.get_device_name(0)}")
        if args.require_cuda:
            if torch.version.cuda is None:
                print("[ERROR] 当前是 CPU 版 PyTorch（未编译 CUDA）。")
                return 2
            if not cuda_ok:
                print("[ERROR] CUDA 版 PyTorch 存在，但当前节点未识别到 GPU。")
                return 3
    except Exception as exc:
        print(f"CUDA check skipped: {exc.__class__.__name__}: {exc}")
        if args.require_cuda:
            return 4

    print("=== 检查完成 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
