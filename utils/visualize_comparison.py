"""生成 Jittor vs PyTorch 语义分割对比可视化。"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CLASS_COLORS = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (255, 165, 0),
]

CLASS_NAMES = [
    "Background",
    "Building",
    "Farmland",
    "Forest",
    "Water",
    "Grassland",
    "Road",
    "Bareland",
    "Industrial",
]


def colorize(label: np.ndarray, num_classes: int = 9) -> np.ndarray:
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(min(num_classes, max(int(label.max()) + 1, 9))):
        rgb[label == cls_idx] = CLASS_COLORS[cls_idx]
    return rgb


def make_legend(width: int = 180, height: int = 230) -> Image.Image:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((5, 5), "Legend", fill=(0, 0, 0))
    for i, (color, name) in enumerate(zip(CLASS_COLORS, CLASS_NAMES)):
        y = 25 + i * 22
        draw.rectangle([5, y, 17, y + 14], fill=color, outline=(0, 0, 0))
        draw.text((22, y), name, fill=(0, 0, 0))
    return img


def compute_miou(pred: np.ndarray, gt: np.ndarray, num_classes: int = 9) -> float:
    ious = []
    for cls_idx in range(num_classes):
        inter = np.sum((pred == cls_idx) & (gt == cls_idx))
        union = np.sum((pred == cls_idx) | (gt == cls_idx))
        ious.append(inter / max(union, 1))
    return float(np.mean(ious))


def load_test_data(manifest_path: Path, split: str = "test", num_samples: int = 8, seed: int = 42):
    random.seed(seed)
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "splits" in data:
        samples = data["splits"].get(split, [])
        image_key, label_key = "image_path", "label_path"
    else:
        samples = data.get(split, [])
        image_key, label_key = "image", "label"

    if not samples:
        return [], []

    picked = random.sample(samples, min(num_samples, len(samples))) if num_samples > 0 else samples
    split_dir = manifest_path.parent

    images, labels = [], []
    for sample in picked:
        img_path = split_dir / sample[image_key]
        lbl_path = split_dir / sample[label_key]
        if img_path.exists() and lbl_path.exists():
            images.append(np.array(Image.open(img_path).convert("RGB")))
            labels.append(np.array(Image.open(lbl_path).convert("L")))
    return images, labels


def try_jittor_predict(weights: Path, checkpoint: Path, image_np: np.ndarray, image_size: int = 224) -> Optional[np.ndarray]:
    try:
        import jittor as jt

        from repro.jittor_models import InfMAEDownstreamJittor
        from scripts.jittor_weight_loader import load_weights_into_model

        jt.flags.use_cuda = 0
        model = InfMAEDownstreamJittor(num_classes=9, freeze_backbone=False)

        if weights.exists():
            load_weights_into_model(model.backbone, weights, strict_shape=False)

        if checkpoint.exists():
            state = jt.load(str(checkpoint))
            if hasattr(model, "load_parameters"):
                model.load_parameters(state)
            else:
                model.load_state_dict(state)

        model.eval()
        img = Image.fromarray(image_np).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.425) / 0.200
        x_np = np.ascontiguousarray(arr.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)
        x = jt.array(x_np)

        with jt.no_grad():
            logits = model(x)
        pred = np.asarray(logits.numpy()).argmax(axis=1)[0].astype(np.uint8)
        pred = np.array(Image.fromarray(pred).resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST))
        return pred
    except Exception as exc:
        print(f"[WARN] Jittor prediction failed: {exc}")
        return None


def try_pytorch_predict(pretrained_ckpt: Path, checkpoint: Path, image_np: np.ndarray, image_size: int = 224) -> Optional[np.ndarray]:
    try:
        import torch
        from repro.pytorch_models import InfMAEDownstreamPyTorch

        device = torch.device("cpu")
        model = InfMAEDownstreamPyTorch(
            num_classes=9,
            pretrained_ckpt=str(pretrained_ckpt),
            image_size=image_size,
            freeze_backbone=False,
        )
        if checkpoint.exists():
            state = torch.load(str(checkpoint), map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)

        model = model.to(device)
        model.eval()
        img = Image.fromarray(image_np).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.425) / 0.200
        x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
        pred = logits[0].cpu().numpy().argmax(axis=0).astype(np.uint8)
        pred = np.array(Image.fromarray(pred).resize((image_np.shape[1], image_np.shape[0]), Image.NEAREST))
        return pred
    except Exception as exc:
        print(f"[WARN] PyTorch prediction failed: {exc}")
        return None


def make_single_panel(image: np.ndarray, gt: np.ndarray, pred: Optional[np.ndarray] = None, title_suffix: str = "", save_path: Optional[Path] = None) -> Image.Image:
    h, w = image.shape[:2]
    gt_rgb = colorize(gt)
    panels = [
        (Image.fromarray(image), "RGB Image"),
        (Image.fromarray(gt_rgb), "Ground Truth"),
    ]

    if pred is not None:
        pred_rgb = colorize(pred)
        error = np.zeros((h, w, 3), dtype=np.uint8)
        error[pred == gt] = [0, 200, 0]
        error[pred != gt] = [220, 20, 20]
        panels.append((Image.fromarray(pred_rgb), f"Prediction{title_suffix}"))
        panels.append((Image.fromarray(error), "Error Map"))

    gap = 3
    legend = make_legend()
    total_w = w * len(panels) + gap * (len(panels) - 1) + legend.width + gap * 2
    total_h = max(h, legend.height) + 28

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (panel, title) in enumerate(panels):
        x = i * (w + gap) + gap
        canvas.paste(panel, (x, 28))
        draw.text((x + 2, 5), title, fill=(0, 0, 0))

    legend_x = len(panels) * (w + gap) + gap
    canvas.paste(legend, (legend_x, 28))

    if save_path is not None:
        canvas.save(save_path)
    return canvas


def make_comparison_panel(image: np.ndarray, gt: np.ndarray, jt_pred: Optional[np.ndarray] = None, pt_pred: Optional[np.ndarray] = None, save_path: Optional[Path] = None) -> Image.Image:
    h, w = image.shape[:2]
    gt_rgb = colorize(gt)
    panels = [
        (Image.fromarray(image), "RGB Image"),
        (Image.fromarray(gt_rgb), "Ground Truth"),
    ]

    if jt_pred is not None:
        panels.append((Image.fromarray(colorize(jt_pred)), f"Jittor (mIoU={compute_miou(jt_pred, gt):.4f})"))
    if pt_pred is not None:
        panels.append((Image.fromarray(colorize(pt_pred)), f"PyTorch (mIoU={compute_miou(pt_pred, gt):.4f})"))

    gap = 3
    legend = make_legend()
    total_w = w * len(panels) + gap * (len(panels) - 1) + legend.width + gap * 2
    total_h = max(h, legend.height) + 28

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (panel, title) in enumerate(panels):
        x = i * (w + gap) + gap
        canvas.paste(panel, (x, 28))
        draw.text((x + 2, 5), title, fill=(0, 0, 0))

    legend_x = len(panels) * (w + gap) + gap
    canvas.paste(legend, (legend_x, 28))

    if save_path is not None:
        canvas.save(save_path)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Jittor/PyTorch segmentation visual comparisons")
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "msrs_shared" / "manifest.json"))
    parser.add_argument("--save-dir", default=str(REPO_ROOT / "work_dirs" / "visualizations"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--jt-weights", default=str(REPO_ROOT / "weights" / "full_model_weights.npz"))
    parser.add_argument("--jt-checkpoint", default=str(REPO_ROOT / "weights" / "best_miou.pkl"))
    parser.add_argument("--pt-pretrained", default="InfMAE.pth")
    parser.add_argument("--pt-checkpoint", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")

    images, labels = load_test_data(manifest_path, split="test", num_samples=args.num_samples, seed=args.seed)
    if not images:
        raise ValueError("未从 manifest 读取到可视化样本")

    save_dir = Path(args.save_dir)
    comp_dir = save_dir / "comparison"
    save_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    jt_weights = Path(args.jt_weights)
    jt_checkpoint = Path(args.jt_checkpoint)
    pt_pretrained = Path(args.pt_pretrained)
    pt_checkpoint = Path(args.pt_checkpoint) if args.pt_checkpoint else Path("")

    jt_pred_func = lambda img: try_jittor_predict(jt_weights, jt_checkpoint, img, image_size=args.image_size)
    pt_pred_func = None
    if args.pt_checkpoint:
        pt_pred_func = lambda img: try_pytorch_predict(pt_pretrained, pt_checkpoint, img, image_size=args.image_size)

    for idx in range(len(images)):
        jt_pred = jt_pred_func(images[idx])
        pt_pred = pt_pred_func(images[idx]) if pt_pred_func is not None else None

        if jt_pred is not None:
            save_path = save_dir / f"jittor_{idx+1:03d}.png"
            miou = compute_miou(jt_pred, labels[idx])
            make_single_panel(images[idx], labels[idx], jt_pred, f" (mIoU={miou:.4f})", save_path)
            print(f"  [saved] Jittor: {save_path} mIoU={miou:.4f}")

        if pt_pred is not None:
            save_path = save_dir / f"pytorch_{idx+1:03d}.png"
            miou = compute_miou(pt_pred, labels[idx])
            make_single_panel(images[idx], labels[idx], pt_pred, f" (mIoU={miou:.4f})", save_path)
            print(f"  [saved] PyTorch: {save_path} mIoU={miou:.4f}")

        if jt_pred is not None or pt_pred is not None:
            save_path = comp_dir / f"compare_{idx+1:03d}.png"
            make_comparison_panel(images[idx], labels[idx], jt_pred, pt_pred, save_path)
            print(f"  [saved] Comparison: {save_path}")

    print(f"\n[DONE] Visualizations saved to {save_dir}")


if __name__ == "__main__":
    main()
