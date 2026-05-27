#!/usr/bin/env python3
"""根据 outputs/pipeline_manifest.json 与产物文件推断流水线停在哪一步（离线排查）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# 与 run_paper_repro_pipeline.py 中步骤顺序一致
STEP_ORDER: list[tuple[str, str, list[str]]] = [
    ("data_check", "输入 npy 校验", []),
    ("phys_surrogate", "训练 phys surrogate", ["phys_models/phys_dnn.pth", "phys_models/phys_cnn.pth"]),
    (
        "gan_normal",
        "GAN normal 预训练",
        ["gan_fallback/generator_fallback.pth", "gan_fallback/critic_fallback.pth", "gan_fallback/metrics_normal.csv"],
    ),
    (
        "gan_physics",
        "GAN physics 微调",
        ["gan_fallback/generator_fallback.pth", "gan_fallback/critic_fallback.pth", "gan_fallback/metrics_physics.csv"],
    ),
    ("pso", "PSO forward design", ["forward_design/prior_j.npy", "forward_design/pso_history.csv", "forward_design/best_latent.npy"]),
    (
        "openfoam_export",
        "OpenFOAM 转 npy（可选）",
        ["openfoam_export/phase_low.npy"],
    ),
    (
        "paper_figures",
        "论文图 evaluate_paper_figures",
        [
            "paper_figures/figure2_cd_wasserstein.png",
            "paper_figures/figure7_abc_performance_insight.png",
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="解读 pipeline_manifest.json，定位未完成步骤")
    p.add_argument("--out-dir", default="outputs", help="含 pipeline_manifest.json 的目录")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.out_dir)
    mpath = root / "pipeline_manifest.json"
    if not mpath.exists():
        print(f"[ERROR] 未找到: {mpath}", file=sys.stderr)
        return 1

    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    last = manifest.get("last_completed_step")
    phase = manifest.get("pipeline_phase", "")
    status = manifest.get("status", "")
    print("=== pipeline 诊断 ===")
    print(f"manifest: {mpath.resolve()}")
    print(f"status: {status}")
    print(f"pipeline_phase: {phase}")
    print(f"last_completed_step: {last}")
    print(f"last_checkpoint_at: {manifest.get('last_checkpoint_at', '')}")
    if manifest.get("failed_step_id"):
        print(f"failed_step_id: {manifest.get('failed_step_id')}")
        print(f"failed_exit_code: {manifest.get('failed_exit_code')}")
    if manifest.get("failed_reason"):
        print(f"failed_reason: {manifest.get('failed_reason')}")

    # 按文件存在性推断（兼容旧 manifest 无 last_completed_step）
    print("\n=== 产物文件探测（相对 --out-dir）===")
    skip_of = bool(manifest.get("skip_openfoam"))
    idx = 0
    for i, (sid, name, rel_list) in enumerate(STEP_ORDER):
        if sid == "data_check":
            continue
        if sid == "openfoam_export" and skip_of:
            print(f"  [--] {sid}: {name}（已 skip_openfoam，不检查文件）")
            continue
        missing = [r for r in rel_list if not (root / r).exists()]
        ok = len(missing) == 0
        mark = "OK" if ok else "缺"
        print(f"  [{mark}] {sid}: {name}")
        if rel_list and not ok:
            for m in missing:
                print(f"        - 缺 {m}")
        if ok:
            idx = i

    print("\n=== 推断结论 ===")
    if status == "success" and manifest.get("finished_at"):
        print("流水线已成功完成（manifest 含 finished_at 且 status=success）。")
        return 0

    if last:
        nxt = None
        for i, (sid, _, _) in enumerate(STEP_ORDER):
            if sid == last and i + 1 < len(STEP_ORDER):
                nxt = STEP_ORDER[i + 1]
                break
        if nxt:
            print(f"检查点记录：已完成「{last}」。若作业中断，通常卡在下一步「{nxt[0]}: {nxt[1]}」。")
        else:
            print(f"检查点记录：最后一步为「{last}」。")

    # 文件推断
    sid_guess, _, _ = STEP_ORDER[idx] if idx < len(STEP_ORDER) else STEP_ORDER[-1]
    print(f"按磁盘文件推断：至少已完成到「{STEP_ORDER[idx][1]}」（step_id≈{sid_guess}）。")

    if not (root / "gan_fallback" / "metrics_normal.csv").exists() and (root / "phys_models" / "phys_dnn.pth").exists():
        print(
            "\n[常见原因] phys 已完成但无 gan_fallback/metrics_normal.csv："
            "多为 GAN normal 未跑完（Slurm 时间/内存、CUDA OOM、或作业被手动 scancel）。"
            "请查 logs/*.err、outputs/pipeline.log 末尾、sacct 的 State/MaxRSS。"
        )

    if status == "running" and last == "data_check" and (root / "phys_models" / "metrics.json").exists():
        print(
            "\n[注意] manifest 仍为旧版（仅 data_check、无 last_completed_step），"
            "但 phys_models/metrics.json 存在：说明 phys 曾跑过，建议用新版流水线重跑以写入检查点。"
        )

    print(f"\n建议：tail -n 80 {root / 'pipeline.log'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
