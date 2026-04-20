from __future__ import annotations

import jittor as jt
from jittor import nn
import argparse
import gc
import json
import pickle
import random
import sys
import time
import math
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repro.common import collect_image_files, ensure_dir, save_rows_to_csv
from repro.jittor_models import InfMAEMSEPretrainJittor
from scripts import alignment_hparams as hp
from scripts.jittor_weight_loader import format_report, load_weights_into_model

# InfMAE 原始论文使用的归一化参数 (Inf30 红外数据集)
INFMAE_MEAN = np.array([0.425, 0.425, 0.425], dtype=np.float32)
INFMAE_STD = np.array([0.200, 0.200, 0.200], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InfMAE masked autoencoder pretraining (Jittor)")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--weights", default="")
    parser.add_argument("--epochs", type=int, default=hp.PRETRAIN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=hp.PRETRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=hp.PRETRAIN_LR)
    parser.add_argument("--weight-decay", type=float, default=hp.PRETRAIN_WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=int, default=hp.PRETRAIN_WARMUP_EPOCHS)
    parser.add_argument("--image-size", type=int, default=hp.IMAGE_SIZE)
    parser.add_argument("--max-samples", type=int, default=hp.INF30_MAX_SAMPLES)
    parser.add_argument("--mask-ratio", type=float, default=hp.PRETRAIN_MASK_RATIO)
    parser.add_argument("--seed", type=int, default=hp.SEED)
    parser.add_argument("--work-dir", default=hp.PRETRAIN_WORK_DIR)
    parser.add_argument("--resume", default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def iter_batches(paths: List[Path], batch_size: int):
    total = (len(paths) // batch_size) * batch_size
    for start in range(0, total, batch_size):
        yield paths[start : start + batch_size]


def load_and_augment_image(path: Path, image_size: int, rng: random.Random) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    orig_w, orig_h = img.size
    area = orig_w * orig_h
    scale = rng.uniform(0.2, 1.0)
    target_area = area * scale
    aspect_ratio = math.exp(rng.uniform(math.log(3/4), math.log(4/3)))  # w/h
    crop_w = int(round(math.sqrt(target_area * aspect_ratio)))
    crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
    crop_h = min(crop_h, orig_h)
    crop_w = min(crop_w, orig_w)
    crop_h = max(crop_h, 1)
    crop_w = max(crop_w, 1)
    top = rng.randint(0, orig_h - crop_h)
    left = rng.randint(0, orig_w - crop_w)
    img = img.crop((left, top, left + crop_w, top + crop_h))
    img = img.resize((image_size, image_size), Image.BICUBIC)

    # RandomHorizontalFlip (50%)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # ToTensor: HWC [0,255] -> CHW [0.0, 1.0]
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))

    # Normalize
    arr = (arr - INFMAE_MEAN.reshape(3, 1, 1)) / INFMAE_STD.reshape(3, 1, 1)
    return arr

def load_batch(paths: List[Path], image_size: int, rng: random.Random) -> np.ndarray:
    arrays = [load_and_augment_image(path, image_size, rng) for path in paths]
    return np.stack(arrays, axis=0).astype(np.float32)

def safe_save_state_dict(model, path: str) -> None:
    """将 state_dict 转为 numpy 后用 pickle 保存，避免 jt.save 的 GPU 显存问题。"""
    state_dict = model.state_dict()
    cpu_state = {}
    for k, v in state_dict.items():
        try:
            cpu_state[k] = np.asarray(v.numpy()).copy()
        except Exception:
            cpu_state[k] = None
    with open(path, "wb") as f:
        pickle.dump(cpu_state, f, protocol=4)
    del cpu_state
    gc.collect()
    try:
        import jittor as jt
        jt.sync_all()
        jt.gc()
    except Exception:
        pass


def cosine_lr(base_lr: float, epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    """Cosine learning rate decay with linear warmup (与 timm optim_factory 一致)."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))


def add_weight_decay(model, weight_decay: float):
    """参数分组: bias/norm 层不加 weight decay, 与 timm optim_factory 一致."""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower() or "ln" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    

def main() -> None:
    args = parse_args()
    jt.flags.use_cuda = 1
    set_seed(args.seed)

    data_root = Path(args.data_root)

    paths = collect_image_files(data_root)
    if args.max_samples > 0 and len(paths) > args.max_samples:
        rng = random.Random(args.seed)
        rng.shuffle(paths)
        paths = paths[: args.max_samples]

    num_batches = len(paths) // args.batch_size
    print(f"[INFO] 使用 {len(paths)} 张图像, image_size={args.image_size}, "
          f"batch_size={args.batch_size}, mask_ratio={args.mask_ratio}, "
          f"epochs={args.epochs}, lr={args.lr}, wd={args.weight_decay}, "
          f"warmup={args.warmup_epochs}")

    work_dir = Path(args.work_dir)
    ensure_dir(work_dir)

    model = InfMAEMSEPretrainJittor(mask_ratio=args.mask_ratio)

    # 加载预训练权重（可选）
    if args.weights and Path(args.weights).exists():
        report = load_weights_into_model(model, Path(args.weights), strict_shape=False)
        print(format_report(report))

    # 使用 AdamW + weight decay (与原始论文一致)
    param_groups = add_weight_decay(model, args.weight_decay)
    optimizer = nn.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    # 输出文件
    train_log_path = work_dir / "jittor_pretrain_log.csv"
    perf_log_path = work_dir / "jittor_perf_log.csv"
    text_log_path = work_dir / "jittor_pretrain.log"
    train_log_rows: List[dict] = []
    perf_log_rows: List[dict] = []
    cum_time = 0.0

    with text_log_path.open("w", encoding="utf-8") as log_f:
        log_f.write(f"# Jittor Pretrain training log\n")
        log_f.write(f"# Epochs: {args.epochs}, batches/epoch: {num_batches}, batch_size: {args.batch_size}\n\n")

        for epoch in range(1, args.epochs + 1):
            random.Random(args.seed + epoch).shuffle(paths)
            losses: List[float] = []
            batch_times: List[float] = []
            t0 = time.time()

            # Cosine LR schedule
            current_lr = cosine_lr(args.lr, epoch, args.epochs, args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            batch_rng = random.Random(args.seed + epoch)
            for batch_idx, batch_paths in enumerate(iter_batches(paths, args.batch_size)):
                bt0 = time.time()
                image_np = load_batch(batch_paths, args.image_size, batch_rng)
                image_jt = jt.array(image_np)

                loss, pred, mask = model(image_jt)
                optimizer.step(loss)

                loss_val = float(np.asarray(loss.numpy()).reshape(-1)[0])
                losses.append(loss_val)
                batch_times.append(time.time() - bt0)

                if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
                    print(f"  epoch {epoch} batch {batch_idx+1}/{num_batches} "
                          f"loss={loss_val:.6f} lr={current_lr:.6e}")

            elapsed = time.time() - t0
            mean_loss = float(np.mean(losses)) if losses else 0.0
            avg_batch_time = float(np.mean(batch_times)) if batch_times else 0.0
            train_fps = args.batch_size / avg_batch_time if avg_batch_time > 0 else 0.0

            # 1) jittor_pretrain_log.csv: epoch, train_loss, lr
            train_row = {"epoch": epoch, "train_loss": f"{mean_loss:.6f}", "lr": f"{current_lr:.10e}"}
            train_log_rows.append(train_row)

            # 2) jittor_perf_log.csv: epoch, epoch_time, avg_batch_time, train_fps
            perf_row = {"epoch": epoch, "epoch_time": round(elapsed, 1),
                        "avg_batch_time": round(avg_batch_time, 2),
                        "train_fps": round(train_fps, 1)}
            perf_log_rows.append(perf_row)

            # 3) jittor_pretrain.log (text)
            h, rem = divmod(int(cum_time), 3600)
            m, s = divmod(rem, 60)
            log_f.write(f"[{h:02d}:{m:02d}:{s:02d}] [EPOCH {epoch}/{args.epochs}] "
                        f"loss={mean_loss:.6f} lr={current_lr:.6f} time={elapsed:.1f}s\n")
            cum_time += elapsed

            print(f"[EPOCH {epoch}/{args.epochs}] loss={mean_loss:.6f} lr={current_lr:.6e} "
                  f"time={elapsed:.1f}s avg_bt={avg_batch_time:.2f}s fps={train_fps:.1f}")

            # 保存所有 CSV（每 epoch）
            import csv as _csv
            with train_log_path.open("w", encoding="utf-8", newline="") as csv_f:
                _csv.DictWriter(csv_f, fieldnames=["epoch", "train_loss", "lr"]).writeheader()
                _csv.DictWriter(csv_f, fieldnames=["epoch", "train_loss", "lr"]).writerows(train_log_rows)
            with perf_log_path.open("w", encoding="utf-8", newline="") as csv_f:
                _csv.DictWriter(csv_f, fieldnames=["epoch", "epoch_time", "avg_batch_time", "train_fps"]).writeheader()
                _csv.DictWriter(csv_f, fieldnames=["epoch", "epoch_time", "avg_batch_time", "train_fps"]).writerows(perf_log_rows)

            # 每 epoch 保存 checkpoint（支持断点续训）
            jt.sync_all()
            jt.gc()
            safe_save_state_dict(model, str(work_dir / "pretrain_mae_latest.pkl"))

    # 最终保存 checkpoint + hparams
    jt.sync_all()
    jt.gc()
    safe_save_state_dict(model, str(work_dir / "pretrain_mae_latest.pkl"))

    with (work_dir / "hparams.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"[DONE] checkpoint saved to {work_dir / 'pretrain_mae_latest.pkl'}")
    print(f"[DONE] train_log: {train_log_path}")
    print(f"[DONE] perf_log:  {perf_log_path}")
    print(f"[DONE] text_log:  {text_log_path}")


if __name__ == "__main__":
    main()
