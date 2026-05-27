from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载可访问的公开替代资源（绕过 EDX 403）")
    parser.add_argument(
        "--source",
        choices=["github_sofc_img"],
        default="github_sofc_img",
        help="数据源类型",
    )
    parser.add_argument("--out-dir", default="data/downloads/open_fallback", help="下载输出目录")
    return parser.parse_args()


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, timeout=120, stream=True, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 512):
                if chunk:
                    f.write(chunk)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "github_sofc_img":
        url = "https://codeload.github.com/Devetree/SOFC-IMG/zip/refs/heads/main"
        zip_path = out_dir / "SOFC-IMG-main.zip"
        extract_dir = out_dir / "SOFC-IMG-main"

        print(f"开始下载: {url}")
        download_file(url, zip_path)
        print(f"下载完成: {zip_path} ({zip_path.stat().st_size} bytes)")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        print(f"解压完成: {extract_dir}")
        print("说明: 该资源主要是 SOFC 微结构分析代码，不是论文原始 3D 训练库。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
