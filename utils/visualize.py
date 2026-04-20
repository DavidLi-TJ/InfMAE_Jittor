"""Run Jittor segmentation visualization on the MSRS test split."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CKPT = REPO_ROOT / "weights" / "best_miou.pkl"
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "full_model_weights.npz"
DEFAULT_MANIFEST = REPO_ROOT / "data" / "msrs_shared" / "manifest.json"
DEFAULT_SAVE_DIR = REPO_ROOT / "work_dirs" / "visualizations"

CLASS_COLORS = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
    (255, 165, 0),
]
CLASS_NAMES = ['Background', 'Building', 'Farmland', 'Forest', 'Water',
               'Grassland', 'Road', 'Bareland', 'Industrial']


def colorize(label):
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(len(CLASS_COLORS), max(int(label.max())+1, 9))):
        rgb[label == c] = CLASS_COLORS[c]
    return rgb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Jittor segmentation outputs on MSRS split")
    parser.add_argument("--ckpt-path", default=str(DEFAULT_CKPT), help="下游 checkpoint 路径")
    parser.add_argument("--weights-path", default=str(DEFAULT_WEIGHTS), help="backbone/full model 权重路径")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="MSRS manifest.json 路径")
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR), help="可视化输出目录")
    parser.add_argument("--num-samples", type=int, default=8, help="随机可视化样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--image-size", type=int, default=224, help="模型输入尺寸")
    parser.add_argument("--use-cuda", type=int, default=0, help="是否启用 CUDA (0/1)")
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt_path)
    weights_path = Path(args.weights_path)
    manifest_path = Path(args.manifest)
    save_dir = Path(args.save_dir)

    import jittor as jt
    jt.flags.use_cuda = 1 if args.use_cuda else 0

    print("[INFO] Building Jittor model...")
    from repro.jittor_models import InfMAEDownstreamJittor
    model = InfMAEDownstreamJittor(num_classes=9, freeze_backbone=False)

    # Load full model weights (backbone + decode_head)
    if weights_path.exists():
        print(f"[INFO] Loading weights: {weights_path}")
        npz = np.load(weights_path, allow_pickle=True)
        jt_state = model.state_dict()
        for k in jt_state:
            if k in npz.files:
                jt_state[k] = jt.array(np.ascontiguousarray(npz[k], dtype=np.float32))
        model.load_state_dict(jt_state)

    # Load downstream checkpoint (decode_head fine-tuned, overrides decode_head weights)
    if ckpt_path.exists():
        print(f"[INFO] Loading checkpoint: {ckpt_path}")
        model.load_state_dict(jt.load(str(ckpt_path)))
    model.eval()

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data['splits']['test']
    sample_count = min(args.num_samples, len(samples))
    samples = random.Random(args.seed).sample(samples, sample_count)
    split_dir = manifest_path.parent

    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples):
        print(f"[{idx+1}/{sample_count}] Processing...")
        img_path = split_dir / sample['image_path']
        lbl_path = split_dir / sample['label_path']
        image = np.array(Image.open(img_path).convert('RGB'))
        gt = np.array(Image.open(lbl_path).convert('L'))
        h, w = image.shape[:2]

        # Preprocess
        img_t = Image.fromarray(image).resize((args.image_size, args.image_size), Image.BILINEAR)
        arr = np.array(img_t, dtype=np.float32) / 255.0
        arr = (arr - 0.425) / 0.200
        x_np = np.ascontiguousarray(arr.transpose(2, 0, 1)[np.newaxis], dtype=np.float32)
        x = jt.array(x_np)

        # Predict
        with jt.no_grad():
            logits = model(x)
        logits = logits[0].numpy()
        pred = np.argmax(logits, axis=0).astype(np.uint8)
        pred = np.array(Image.fromarray(pred).resize((w, h), Image.NEAREST))

        # IoU
        ious = []
        for c in range(9):
            inter = np.sum((pred == c) & (gt == c))
            union = np.sum((pred == c) | (gt == c))
            ious.append(inter / max(union, 1))
        miou = np.mean(ious)

        # Visualize
        gt_rgb = colorize(gt)
        pred_rgb = colorize(pred)
        error = np.zeros((h, w, 3), dtype=np.uint8)
        error[pred == gt] = [0, 200, 0]
        error[pred != gt] = [220, 0, 0]

        gap = 4
        tw = w * 4 + gap * 3
        canvas = Image.new('RGB', (tw, h + 25), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for i, (im, title) in enumerate(zip(
            [Image.fromarray(image), Image.fromarray(gt_rgb),
             Image.fromarray(pred_rgb), Image.fromarray(error)],
            ['Infrared Image', 'Ground Truth', 'Prediction', 'Error Map'])):
            xo = i * (w + gap) + 2
            canvas.paste(im, (xo, 25))
            draw.text((xo+2, 3), title, fill=(0,0,0))

        sp = save_dir / f'jittor_{idx+1:03d}.png'
        canvas.save(sp)
        print(f"  [saved] {sp} mIoU={miou:.4f}")

    print("[DONE]")


if __name__ == '__main__':
    main()
