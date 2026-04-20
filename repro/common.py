from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

MSRS_CLASSES = [
    "unlabelled",
    "car",
    "person",
    "bicycle",
    "curve",
    "car_stop",
    "guardrail",
    "color_cone",
    "bump",
]

MSRS_PALETTE = [
    [0, 0, 0],
    [64, 0, 128],
    [64, 64, 0],
    [0, 128, 192],
    [0, 0, 192],
    [128, 128, 0],
    [64, 64, 128],
    [192, 128, 128],
    [192, 64, 0],
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_manifest(manifest_path: Path) -> Dict:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pick_split_entries(manifest: Dict, split: str, max_samples: int, seed: int) -> List[Dict]:
    entries = list(manifest["splits"][split])
    if max_samples > 0 and len(entries) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(entries)
        entries = entries[:max_samples]
    return entries


def resolve_entry_paths(data_root: Path, entry: Dict) -> Tuple[Path, Path]:
    image_path = data_root / entry["image_path"]
    label_path = data_root / entry["label_path"]
    return image_path, label_path


# InfMAE 归一化参数 (与预训练一致)
_INFMAE_MEAN = np.array([0.425, 0.425, 0.425], dtype=np.float32).reshape(3, 1, 1)
_INFMAE_STD = np.array([0.200, 0.200, 0.200], dtype=np.float32).reshape(3, 1, 1)

def read_rgb_image(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if image_size > 0 and img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = (arr - _INFMAE_MEAN) / _INFMAE_STD
    return arr


def read_mask(path: Path, image_size: int) -> np.ndarray:
    mask = Image.open(path)
    if image_size > 0 and mask.size != (image_size, image_size):
        mask = mask.resize((image_size, image_size), Image.NEAREST)
    return np.asarray(mask, dtype=np.int64)


def colorize_mask(mask: np.ndarray, palette: Optional[Sequence[Sequence[int]]] = None) -> Image.Image:
    use_palette = palette if palette is not None else MSRS_PALETTE
    canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id, color in enumerate(use_palette):
        canvas[mask == class_id] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(canvas)


def fast_confusion_matrix(pred: np.ndarray, label: np.ndarray, num_classes: int, ignore_index: int = 255) -> np.ndarray:
    valid = label != ignore_index
    pred = pred[valid]
    label = label[valid]
    hist = np.bincount(
        num_classes * label.reshape(-1).astype(np.int64) + pred.reshape(-1).astype(np.int64),
        minlength=num_classes * num_classes,
    )
    return hist.reshape(num_classes, num_classes)


def compute_iou_from_hist(hist: np.ndarray) -> Tuple[float, List[float]]:
    denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
    iou = np.divide(np.diag(hist), np.maximum(denominator, 1), where=denominator > 0)
    valid = denominator > 0
    miou = float(iou[valid].mean()) if valid.any() else 0.0
    return miou, iou.tolist()


def compute_all_metrics(hist: np.ndarray) -> dict:
    """计算所有分割指标: Acc, mAcc, mIoU, FWIoU, F1 + 每类 IoU/F1。"""
    # Pixel Accuracy
    acc = float(np.diag(hist).sum() / max(hist.sum(), 1))
    # Mean Accuracy
    per_class_acc = np.divide(np.diag(hist), np.maximum(hist.sum(1), 1),
                              where=hist.sum(1) > 0)
    valid_acc = hist.sum(1) > 0
    macc = float(per_class_acc[valid_acc].mean()) if valid_acc.any() else 0.0
    # IoU
    denominator = hist.sum(1) + hist.sum(0) - np.diag(hist)
    iou = np.divide(np.diag(hist), np.maximum(denominator, 1), where=denominator > 0)
    valid_iou = denominator > 0
    miou = float(iou[valid_iou].mean()) if valid_iou.any() else 0.0
    # Frequency Weighted IoU
    freq = hist.sum(1) / max(hist.sum(), 1)
    fwiou = float((freq[valid_iou] * iou[valid_iou]).sum())
    # F1 Score (per-class + mean)
    tp = np.diag(hist)
    fp = hist.sum(0) - tp
    fn = hist.sum(1) - tp
    precision = np.divide(tp, np.maximum(tp + fp, 1), where=(tp + fp) > 0)
    recall = np.divide(tp, np.maximum(tp + fn, 1), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, np.maximum(precision + recall, 1),
                   where=(precision + recall) > 0)
    valid_f1 = (tp + fp) > 0  # 有预测的类
    mf1 = float(f1[valid_f1].mean()) if valid_f1.any() else 0.0
    return {"acc": acc, "macc": macc, "miou": miou, "fwiou": fwiou, "mf1": mf1,
            "iou_per_class": iou.tolist(), "f1_per_class": f1.tolist()}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_rows_to_csv(rows: Iterable[Dict], output_path: Path, fieldnames: Sequence[str]) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_image_files(data_root: Path) -> List[Path]:
    paths: List[Path] = []
    for path in sorted(data_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(path)
    return paths
