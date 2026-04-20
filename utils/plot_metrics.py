"""Plot downstream metrics curves from CSV logs."""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(p: Path) -> list:
    with p.open("r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_metrics(rows: list, save_path: Path, title: str = "Downstream Metrics"):
    epochs = [int(r["epoch"]) for r in rows]
    losses = [float(r["train_loss"]) for r in rows]
    miou = [float(r["val_miou"]) for r in rows]
    acc = [float(r["val_acc"]) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, losses, color="#e74c3c", linewidth=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)

    # mIoU
    axes[1].plot(epochs, miou, color="#2980b9", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].set_title("Validation mIoU")
    axes[1].grid(True, alpha=0.3)

    # Accuracy
    axes[2].plot(epochs, acc, color="#27ae60", linewidth=1.5)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Pixel Accuracy")
    axes[2].set_title("Validation Accuracy")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_comparison(jittor_rows, pytorch_rows, save_path: Path):
    """绘制 Jittor vs PyTorch 对比图。"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    j_epochs = [int(r["epoch"]) for r in jittor_rows]
    p_epochs = [int(r["epoch"]) for r in pytorch_rows]

    # Loss 对比
    axes[0].plot(j_epochs, [float(r["train_loss"]) for r in jittor_rows],
                 color="#e74c3c", linewidth=1.5, label="Jittor")
    axes[0].plot(p_epochs, [float(r["train_loss"]) for r in pytorch_rows],
                 color="#3498db", linewidth=1.5, linestyle="--", label="PyTorch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # mIoU 对比
    axes[1].plot(j_epochs, [float(r["val_miou"]) for r in jittor_rows],
                 color="#e74c3c", linewidth=1.5, label="Jittor")
    axes[1].plot(p_epochs, [float(r["val_miou"]) for r in pytorch_rows],
                 color="#3498db", linewidth=1.5, linestyle="--", label="PyTorch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].set_title("Validation mIoU")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Accuracy 对比
    axes[2].plot(j_epochs, [float(r["val_acc"]) for r in jittor_rows],
                 color="#e74c3c", linewidth=1.5, label="Jittor")
    axes[2].plot(p_epochs, [float(r["val_acc"]) for r in pytorch_rows],
                 color="#3498db", linewidth=1.5, linestyle="--", label="PyTorch")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Pixel Accuracy")
    axes[2].set_title("Validation Accuracy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Jittor vs PyTorch - Downstream Comparison", fontsize=14)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--pytorch-csv", type=Path, default=None, help="PyTorch 版本的 downstream log")
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--title", default="Downstream Metrics")
    args = parser.parse_args()

    rows = read_csv(args.csv_path)
    save_dir = args.save_dir or args.csv_path.parent / "metrics_curve"
    fname = args.csv_path.stem + ".png"
    plot_metrics(rows, save_dir / fname, title=args.title)

    if args.pytorch_csv:
        pytorch_rows = read_csv(args.pytorch_csv)
        plot_comparison(rows, pytorch_rows, save_dir / "comparison.png")


if __name__ == "__main__":
    main()
