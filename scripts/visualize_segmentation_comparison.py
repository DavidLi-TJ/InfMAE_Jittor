from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repro.common import MSRS_PALETTE, colorize_mask, load_manifest, pick_split_entries, resolve_entry_paths


def compose_panels(
    input_img: Image.Image,
    gt_img: Image.Image,
    jt_img: Image.Image,
    baseline_img: Optional[Image.Image],
    baseline_label: str,
) -> Image.Image:
    w, h = input_img.size
    panel_count = 4 if baseline_img is not None else 3
    canvas = Image.new("RGB", (w * panel_count, h + 28), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    canvas.paste(input_img.convert("RGB"), (0, 28))
    canvas.paste(gt_img.convert("RGB"), (w, 28))
    canvas.paste(jt_img.convert("RGB"), (w * 2, 28))

    titles = ["Input", "GT", "Jittor"]
    if baseline_img is not None:
        canvas.paste(baseline_img.convert("RGB"), (w * 3, 28))
        titles.append(baseline_label)

    for idx, title in enumerate(titles):
        draw.text((idx * w + 8, 6), title, fill=(10, 10, 10))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Jittor segmentation visualizations with optional baseline")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--jittor-pred-dir", required=True)
    parser.add_argument("--baseline-pred-dir", default=None)
    parser.add_argument("--baseline-label", default="Baseline")
    parser.add_argument("--output-dir", default="work_dirs/alignment/vis_compare")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    manifest = load_manifest(Path(args.manifest))
    entries = pick_split_entries(manifest, args.split, args.max_samples, args.seed)

    jt_pred_dir = Path(args.jittor_pred_dir)
    baseline_pred_dir = Path(args.baseline_pred_dir) if args.baseline_pred_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for entry in entries:
        image_path, label_path = resolve_entry_paths(data_root, entry)
        pred_name = label_path.name
        jt_pred_path = jt_pred_dir / pred_name

        if not jt_pred_path.exists():
            continue

        baseline_pred_path = None
        if baseline_pred_dir is not None:
            baseline_pred_path = baseline_pred_dir / pred_name
            if not baseline_pred_path.exists():
                continue

        input_img = Image.open(image_path).convert("RGB")
        gt_mask = Image.open(label_path)
        jt_mask = Image.open(jt_pred_path)
        baseline_mask = Image.open(baseline_pred_path) if baseline_pred_path is not None else None

        gt_color = colorize_mask(np.asarray(gt_mask), MSRS_PALETTE)
        jt_color = colorize_mask(np.asarray(jt_mask), MSRS_PALETTE)
        baseline_color = colorize_mask(np.asarray(baseline_mask), MSRS_PALETTE) if baseline_mask is not None else None

        quad = compose_panels(input_img, gt_color, jt_color, baseline_color, args.baseline_label)
        quad.save(output_dir / pred_name)
        count += 1

    print(f"已导出对比图数量: {count}")
    print(f"目录: {output_dir}")


if __name__ == "__main__":
    main()
