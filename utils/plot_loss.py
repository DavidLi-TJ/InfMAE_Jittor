"""Plot training loss curves from CSV logs."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(p: Path) -> list:
    with p.open("r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def plot_loss(rows: list, save_path: Path, title: str = "Training Loss", color="#e74c3c", label="Train Loss"):
    epochs = [int(r["epoch"]) for r in rows]
    losses = [float(r["train_loss"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, losses, color=color, linewidth=1.5, label=label)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def plot_comparison(jittor_rows, pytorch_rows, save_path: Path):
    """绘制 Jittor vs PyTorch 预训练 Loss 对比图。"""
    j_epochs = [int(r["epoch"]) for r in jittor_rows]
    p_epochs = [int(r["epoch"]) for r in pytorch_rows]
    j_losses = [float(r["train_loss"]) for r in jittor_rows]
    p_losses = [float(r["train_loss"]) for r in pytorch_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(j_epochs, j_losses, color="#e74c3c", linewidth=1.8, label="Jittor (batch=48)")
    ax.plot(p_epochs, p_losses, color="#3498db", linewidth=1.8, linestyle="--", label="PyTorch (batch=56)")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Pretrain Loss: Jittor vs PyTorch", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 50)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--compare", type=Path, default=None, help="另一框架的 CSV 用于对比")
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--title", default="Training Loss")
    parser.add_argument("--label", default="Train Loss")
    args = parser.parse_args()

    rows = read_csv(args.csv_path)
    save_dir = args.save_dir or args.csv_path.parent / "plots"
    fname = args.csv_path.stem + ".png"
    plot_loss(rows, save_dir / fname, title=args.title, label=args.label)

    if args.compare:
        other = read_csv(args.compare)
        plot_comparison(rows, other, save_dir / "pretrain_loss_comparison.png")


if __name__ == "__main__":
    main()
