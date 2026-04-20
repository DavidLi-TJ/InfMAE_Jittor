"""Plot performance curves (FPS and timing) from CSV logs."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(p: Path) -> list:
    with p.open("r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_perf(rows: list, save_path: Path, title: str = "Performance Log"):
    """绘制单框架性能日志（3 个子图）。"""
    epochs = [int(r["epoch"]) for r in rows]
    epoch_time = [float(r["epoch_time_s"]) for r in rows]
    batch_time = [float(r["avg_batch_time_s"]) for r in rows]
    fps = [float(r["train_fps"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Epoch Time
    axes[0].plot(epochs, epoch_time, color="#3498db", linewidth=1.5)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Epoch Time (s)", fontsize=11)
    axes[0].set_title("Epoch Time", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Batch Time
    axes[1].plot(epochs, batch_time, color="#27ae60", linewidth=1.5)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Batch Time (s)", fontsize=11)
    axes[1].set_title("Avg Batch Time", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # FPS
    axes[2].plot(epochs, fps, color="#e74c3c", linewidth=1.5)
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("FPS (samples/sec)", fontsize=11)
    axes[2].set_title("Train FPS", fontsize=12)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_perf_comparison(jittor_rows, pytorch_rows, save_path: Path):
    """绘制 Jittor vs PyTorch 性能日志对比（3 个子图）。"""
    j_epochs = [int(r["epoch"]) for r in jittor_rows]
    p_epochs = [int(r["epoch"]) for r in pytorch_rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Epoch Time
    axes[0].plot(j_epochs, [float(r["epoch_time_s"]) for r in jittor_rows],
                 color="#e74c3c", linewidth=1.5, label="Jittor")
    axes[0].plot(p_epochs, [float(r["epoch_time_s"]) for r in pytorch_rows],
                 color="#3498db", linewidth=1.5, linestyle="--", label="PyTorch")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Epoch Time (s)", fontsize=11)
    axes[0].set_title("Epoch Time", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Batch Time
    axes[1].plot(j_epochs, [float(r["avg_batch_time_s"]) for r in jittor_rows],
                 color="#e74c3c", linewidth=1.5, label="Jittor")
    axes[1].plot(p_epochs, [float(r["avg_batch_time_s"]) for r in pytorch_rows],
                 color="#3498db", linewidth=1.5, linestyle="--", label="PyTorch")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Batch Time (s)", fontsize=11)
    axes[1].set_title("Avg Batch Time", fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # FPS
    axes[2].plot(j_epochs, [float(r["train_fps"]) for r in jittor_rows],
                 color="#e74c3c", linewidth=1.5, label="Jittor")
    axes[2].plot(p_epochs, [float(r["train_fps"]) for r in pytorch_rows],
                 color="#3498db", linewidth=1.5, linestyle="--", label="PyTorch")
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("FPS (samples/sec)", fontsize=11)
    axes[2].set_title("Train FPS", fontsize=12)
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Performance: Jittor vs PyTorch", fontsize=14)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--compare", type=Path, default=None, help="另一框架的 perf log CSV")
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--title", default="Performance Log")
    parser.add_argument("--output", default=None, help="对比图输出文件名")
    args = parser.parse_args()

    rows = read_csv(args.csv_path)
    save_dir = args.save_dir or args.csv_path.parent / "plots"
    fname = args.csv_path.stem + ".png"
    plot_perf(rows, save_dir / fname, title=args.title)

    if args.compare:
        other = read_csv(args.compare)
        out_name = "perf_comparison.png"
        if args.output:
            out_name = args.output
        plot_perf_comparison(rows, other, save_dir / out_name)


if __name__ == "__main__":
    main()
