from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


def read_curve(csv_path: Path) -> Tuple[List[float], List[float]]:
    epochs: List[float] = []
    values: List[float] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "epoch" not in row or "val_miou" not in row:
                continue
            epochs.append(float(row["epoch"]))
            values.append(float(row["val_miou"]))
    return epochs, values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Jittor mIoU curve with optional baseline")
    parser.add_argument("--jittor-log", required=True, help="Jittor downstream_log.csv")
    parser.add_argument("--baseline-log", default=None, help="可选基线 downstream_log.csv")
    parser.add_argument("--baseline-label", default="Baseline", help="基线曲线名称")
    parser.add_argument("--output", default="work_dirs/alignment/miou_comparison.png", help="输出图像路径")
    return parser.parse_args()


def maybe_read_curve(csv_path: Optional[str]) -> Optional[Tuple[List[float], List[float]]]:
    if not csv_path:
        return None
    path = Path(csv_path)
    if not path.exists():
        print(f"[WARN] 日志不存在，跳过: {path}")
        return None
    return read_curve(path)


def main() -> None:
    args = parse_args()
    jittor_log_path = Path(args.jittor_log)
    if not jittor_log_path.exists():
        print(f"[WARN] Jittor 日志不存在: {jittor_log_path}")
        return

    jt_epochs, jt_values = read_curve(jittor_log_path)
    baseline_curve = maybe_read_curve(args.baseline_log)

    if not jt_epochs:
        print("[WARN] Jittor 日志缺少 epoch/val_miou，跳过绘图")
        return

    if baseline_curve is not None and not baseline_curve[0]:
        print("[WARN] 基线日志缺少 epoch/val_miou，已忽略基线曲线")
        baseline_curve = None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 4.5))
    if baseline_curve is not None:
        base_epochs, base_values = baseline_curve
        plt.plot(base_epochs, base_values, marker="o", linewidth=1.8, label=args.baseline_label)

    plt.plot(jt_epochs, jt_values, marker="s", linewidth=1.8, label="Jittor")
    plt.xlabel("Epoch")
    plt.ylabel("mIoU")
    plt.title("mIoU Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

    print(f"已保存: {output_path}")


if __name__ == "__main__":
    main()
