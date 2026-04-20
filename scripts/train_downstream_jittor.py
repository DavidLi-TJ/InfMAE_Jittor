from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repro.common import (
    MSRS_CLASSES,
    compute_all_metrics,
    ensure_dir,
    fast_confusion_matrix,
    load_manifest,
    pick_split_entries,
    read_mask,
    read_rgb_image,
    resolve_entry_paths,
    save_rows_to_csv,
)
from repro.jittor_models import InfMAEDownstreamJittor
from scripts import alignment_hparams as hp
from scripts.jittor_weight_loader import format_report, load_weights_into_model


try:
    import jittor as jt
    from jittor import nn
    jt.flags.use_cuda = 1
except Exception as exc:  # pragma: no cover
    raise ImportError("请在 Jittor 环境运行 train_downstream_jittor.py") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jittor downstream finetune with UPerNet-style head")
    parser.add_argument("--data-root", default=hp.MSRS_DATA_ROOT, help="MSRS 数据集根目录")
    parser.add_argument("--manifest", default=hp.MSRS_MANIFEST, help="manifest.json")
    parser.add_argument("--weights", default="weights/full_model_weights.npz",
                        help="backbone npz 或 Jittor 预训练 pkl")
    parser.add_argument("--epochs", type=int, default=hp.DOWNSTREAM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=hp.DOWNSTREAM_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=hp.DOWNSTREAM_LR)
    parser.add_argument("--image-size", type=int, default=hp.IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=hp.SEED)
    parser.add_argument("--num-workers", type=int, default=hp.NUM_WORKERS)
    parser.add_argument("--max-train-samples", type=int, default=hp.MSRS_TRAIN_MAX_SAMPLES)
    parser.add_argument("--max-val-samples", type=int, default=hp.MSRS_VAL_MAX_SAMPLES)
    parser.add_argument("--freeze-backbone", action="store_true", default=False, help="冻结 backbone（默认不冻结）")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false",
                        help="解冻 backbone 参与微调（默认不冻结）")
    parser.add_argument("--backbone-lr", type=float, default=hp.DOWNSTREAM_BACKBONE_LR,
                        help="backbone 学习率，0 表示自动取 0.1 * lr")
    parser.add_argument("--work-dir", default=hp.DOWNSTREAM_WORK_DIR)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def iter_batches(entries: List[dict], batch_size: int):
    for start in range(0, len(entries), batch_size):
        yield entries[start : start + batch_size]


def preload_all(data_root: Path, entries: List[dict], image_size: int) -> tuple:
    """一次性将所有图像和标签预加载到内存，后续 epoch 只做索引切分。"""
    def _load_one(entry):
        image_path, label_path = resolve_entry_paths(data_root, entry)
        img = read_rgb_image(image_path, image_size)
        msk = read_mask(label_path, image_size).astype(np.int32)
        return img, msk

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_load_one, entries))

    images = np.stack([r[0] for r in results], axis=0).astype(np.float32)
    masks = np.stack([r[1] for r in results], axis=0).astype(np.int32)
    return images, masks


def evaluate(model: InfMAEDownstreamJittor, images, masks, batch_size: int) -> dict:
    """images/masks 可以是 np.ndarray 或 jt.Var。"""
    is_np = isinstance(images, np.ndarray)
    n = images.shape[0]
    hist = np.zeros((len(MSRS_CLASSES), len(MSRS_CLASSES)), dtype=np.float64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        if is_np:
            batch_img = jt.array(images[start:end])
            batch_msk = masks[start:end]
        else:
            batch_img = images[start:end]
            batch_msk = masks[start:end]
        logits = model(batch_img)
        pred_np = np.asarray(logits.numpy()).argmax(axis=1).astype(np.int64)
        msk_np = np.asarray(batch_msk.numpy() if not is_np else batch_msk).astype(np.int64)
        for b in range(pred_np.shape[0]):
            hist += fast_confusion_matrix(pred_np[b], msk_np[b], num_classes=len(MSRS_CLASSES))
    return compute_all_metrics(hist)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    manifest = load_manifest(Path(args.manifest))
    train_entries = pick_split_entries(manifest, "train", args.max_train_samples, args.seed)
    val_entries = pick_split_entries(manifest, "val", args.max_val_samples, args.seed)

    if not train_entries and not val_entries:
        print("[WARN] train/val split 均为空，跳过训练")
        return

    if not train_entries:
        print("[WARN] train split 为空，回退为使用 val split 进行训练")
        train_entries = list(val_entries)

    if not val_entries:
        print("[WARN] val split 为空，回退为使用 train split 进行验证")
        val_entries = list(train_entries)

    work_dir = Path(args.work_dir)
    ensure_dir(work_dir)

    print("预加载训练数据到内存...")
    train_images, train_masks = preload_all(data_root, train_entries, args.image_size)
    print("预加载验证数据到内存...")
    val_images, val_masks = preload_all(data_root, val_entries, args.image_size)
    print(f"训练集: {train_images.shape}, 验证集: {val_images.shape}")

    model = InfMAEDownstreamJittor(num_classes=len(MSRS_CLASSES), freeze_backbone=args.freeze_backbone)
    report = load_weights_into_model(model.backbone, Path(args.weights), strict_shape=False)
    print(format_report(report))

    if args.freeze_backbone:
        optimizer = nn.Adam(model.parameters(), lr=args.lr)
    else:
        backbone_lr = args.backbone_lr if args.backbone_lr > 0 else args.lr * 0.1
        backbone_params = list(model.backbone.parameters())
        decoder_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        optimizer = nn.Adam([
            {"params": decoder_params, "lr": args.lr},
            {"params": backbone_params, "lr": backbone_lr},
        ], lr=args.lr)
        print(f"backbone_lr={backbone_lr}, decoder_lr={args.lr}")

    fieldnames = ["epoch", "train_loss", "val_miou", "val_acc", "val_macc", "val_fwiou", "val_mf1", "framework"]
    rows: List[dict] = []
    best_miou = -1.0
    n_train = train_images.shape[0]
    log_path = work_dir / "downstream_log.csv"

    # 如果有已存的 log，加载历史记录（支持断点）
    if log_path.exists():
        import csv as _csv
        with log_path.open("r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if rows:
            existing_miou = max(float(r["val_miou"]) for r in rows)
            if existing_miou > best_miou:
                best_miou = existing_miou
            print(f"[RESUME] {len(rows)} epochs in log, best_miou={best_miou:.4f}")

    for epoch in range(1, args.epochs + 1):
        perm = np.random.RandomState(args.seed + epoch).permutation(n_train)
        train_images = train_images[perm]
        train_masks = train_masks[perm]
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            image_jt = jt.array(train_images[start:end])
            mask_jt = jt.array(train_masks[start:end])

            logits = model(image_jt)
            loss = nn.cross_entropy_loss(logits, mask_jt, ignore_index=255)
            optimizer.step(loss)
            total_loss += float(loss.item())
            n_batches += 1

        train_loss = total_loss / max(n_batches, 1)
        metrics = evaluate(model, val_images, val_masks, args.batch_size)
        val_miou = metrics["miou"]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_miou": val_miou,
            "val_acc": metrics["acc"],
            "val_macc": metrics["macc"],
            "val_fwiou": metrics["fwiou"],
            "val_mf1": metrics["mf1"],
            "framework": "jittor",
        }
        rows.append(row)
        print(row)

        jt.save(model.state_dict(), str(work_dir / "latest.pkl"))
        if val_miou > best_miou:
            best_miou = val_miou
            jt.save(model.state_dict(), str(work_dir / "best_miou.pkl"))

        # 每 epoch 实时写入 CSV
        save_rows_to_csv(rows, log_path, fieldnames=fieldnames)

    with (work_dir / "hparams.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
