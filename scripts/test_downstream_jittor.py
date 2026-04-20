from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repro.common import (
    MSRS_CLASSES,
    MSRS_PALETTE,
    colorize_mask,
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
except Exception as exc:  # pragma: no cover
    raise ImportError("请在 Jittor 环境运行 test_downstream_jittor.py") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jittor downstream test and visualization export")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--weights", required=True, help="backbone npz 或 Jittor 预训练 pkl")
    parser.add_argument("--checkpoint", required=True, help="train_downstream_jittor.py 导出的 checkpoint")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--image-size", type=int, default=hp.IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=hp.BATCH_SIZE)
    parser.add_argument("--max-samples", type=int, default=hp.MSRS_TEST_MAX_SAMPLES)
    parser.add_argument("--work-dir", default="work_dirs/alignment/jittor/downstream_test")
    return parser.parse_args()


def iter_batches(entries: List[dict], batch_size: int):
    for start in range(0, len(entries), batch_size):
        yield entries[start : start + batch_size]


def load_batch(data_root: Path, entries: List[dict], image_size: int):
    images = []
    masks = []
    label_names = []
    for entry in entries:
        image_path, label_path = resolve_entry_paths(data_root, entry)
        images.append(read_rgb_image(image_path, image_size))
        masks.append(read_mask(label_path, image_size).astype(np.int64))
        label_names.append(label_path.name)
    return np.stack(images, axis=0).astype(np.float32), np.stack(masks, axis=0), label_names


def load_checkpoint(model: InfMAEDownstreamJittor, checkpoint_path: Path) -> None:
    state = jt.load(str(checkpoint_path))
    if hasattr(model, "load_parameters"):
        model.load_parameters(state)
    elif hasattr(model, "load_state_dict"):
        model.load_state_dict(state)
    else:
        print("[WARN] 当前 Jittor 模型不支持参数加载接口，已跳过 checkpoint 加载")


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    manifest = load_manifest(Path(args.manifest))
    entries = pick_split_entries(manifest, args.split, args.max_samples, seed=hp.SEED)
    if not entries:
        print(f"[WARN] {args.split} split 为空，跳过测试")
        return

    work_dir = Path(args.work_dir)
    pred_dir = work_dir / "predictions"
    color_dir = work_dir / "predictions_color"
    ensure_dir(pred_dir)
    ensure_dir(color_dir)

    model = InfMAEDownstreamJittor(num_classes=len(MSRS_CLASSES), freeze_backbone=True)
    report = load_weights_into_model(model.backbone, Path(args.weights), strict_shape=False)
    print(format_report(report))
    load_checkpoint(model, Path(args.checkpoint))

    hist = np.zeros((len(MSRS_CLASSES), len(MSRS_CLASSES)), dtype=np.float64)

    for batch_entries in iter_batches(entries, args.batch_size):
        image_np, mask_np, label_names = load_batch(data_root, batch_entries, args.image_size)
        logits = model(jt.array(image_np))
        pred_np = np.asarray(logits.numpy()).argmax(axis=1).astype(np.uint8)

        for b in range(pred_np.shape[0]):
            pred = pred_np[b]
            label = mask_np[b].astype(np.int64)
            hist += fast_confusion_matrix(pred, label, num_classes=len(MSRS_CLASSES))

            pred_name = label_names[b]
            Image.fromarray(pred).save(pred_dir / pred_name)
            colorize_mask(pred, MSRS_PALETTE).save(color_dir / pred_name)

    metrics = compute_all_metrics(hist)
    metric_row = {"framework": "jittor", "split": args.split}
    metric_row.update({k: metrics[k] for k in ["acc", "macc", "miou", "fwiou", "mf1"]})
    for idx, value in enumerate(metrics["iou_per_class"]):
        metric_row[f"iou_{MSRS_CLASSES[idx]}"] = value
    for idx, value in enumerate(metrics["f1_per_class"]):
        metric_row[f"f1_{MSRS_CLASSES[idx]}"] = value

    save_rows_to_csv([metric_row], work_dir / "test_metrics.csv", fieldnames=list(metric_row.keys()))
    print(metric_row)


if __name__ == "__main__":
    main()
