from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
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


def is_image_file(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def is_zip_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".zip"


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_json(payload: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_text_lines(lines: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def sample_items(items: List[Dict], max_samples: int, seed: int) -> List[Dict]:
    if max_samples > 0 and len(items) > max_samples:
        rng = random.Random(seed)
        shuffled = list(items)
        rng.shuffle(shuffled)
        return shuffled[:max_samples]
    return list(items)


def flatten_relative_path(relative_path: Path) -> str:
    parts = [part for part in relative_path.parts if part not in ("", ".", "..")]
    if not parts:
        return relative_path.name or "sample"
    return "__".join(parts)


def copy_source_item(source_kind: str, source_ref, destination: Path, zip_file: Optional[zipfile.ZipFile] = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_kind == "file":
        shutil.copy2(source_ref, destination)
        return
    if zip_file is None:
        raise ValueError("zip_file is required for zip sources")
    with zip_file.open(source_ref) as src, destination.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def save_resized_source_item(
    source_kind: str,
    source_ref,
    destination: Path,
    resize_size: int,
    zip_file: Optional[zipfile.ZipFile] = None,
    is_mask: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_kind == "file":
        with Image.open(source_ref) as img:
            if is_mask:
                img = img.resize((resize_size, resize_size), Image.NEAREST)
            else:
                img = img.convert("RGB")
                img = img.resize((resize_size, resize_size), Image.BICUBIC)
            img.save(destination)
        return

    if zip_file is None:
        raise ValueError("zip_file is required for zip sources")

    with zip_file.open(source_ref) as src:
        with Image.open(BytesIO(src.read())) as img:
            if is_mask:
                img = img.resize((resize_size, resize_size), Image.NEAREST)
            else:
                img = img.convert("RGB")
                img = img.resize((resize_size, resize_size), Image.BICUBIC)
            img.save(destination)


def collect_inf30_items(source_root: Path) -> List[Dict]:
    items: List[Dict] = []
    if is_zip_path(source_root):
        with zipfile.ZipFile(source_root, "r") as zip_file:
            for member in sorted(zip_file.namelist()):
                if member.endswith("/"):
                    continue
                pure = PurePosixPath(member)
                if not is_image_file(pure.name):
                    continue
                items.append(
                    {
                        "source_kind": "zip",
                        "source_ref": member,
                        "relative_path": Path(*pure.parts),
                    }
                )
        return items

    for path in sorted(source_root.rglob("*")):
        if path.is_file() and is_image_file(path.name):
            items.append(
                {
                    "source_kind": "file",
                    "source_ref": path,
                    "relative_path": path.relative_to(source_root),
                }
            )
    return items


def prepare_inf30_subset(source_root: Path, output_root: Path, max_samples: int, seed: int, overwrite: bool, resize_size: int) -> Dict:
    items = collect_inf30_items(source_root)
    if not items:
        print(f"[WARN] 未找到可用 Inf30 图像: {source_root}")
        ensure_clean_dir(output_root, overwrite)
        manifest = {
            "dataset": "inf30",
            "source_root": str(source_root.as_posix()),
            "output_root": str(output_root.as_posix()),
            "seed": seed,
            "max_samples": max_samples,
            "resize_size": resize_size,
            "num_selected": 0,
            "items": [],
        }
        save_json(manifest, output_root / "manifest.json")
        save_text_lines([], output_root / "selected_images.txt")
        return manifest
    selected = sample_items(items, max_samples, seed)
    ensure_clean_dir(output_root, overwrite)
    images_dir = output_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_items: List[Dict] = []
    zip_file = zipfile.ZipFile(source_root, "r") if is_zip_path(source_root) else None
    try:
        for index, item in enumerate(selected):
            relative_path = item["relative_path"]
            flat_name = "{0:05d}__{1}".format(index, flatten_relative_path(relative_path))
            destination = images_dir / flat_name
            if resize_size > 0:
                save_resized_source_item(item["source_kind"], item["source_ref"], destination, resize_size, zip_file=zip_file, is_mask=False)
            else:
                copy_source_item(item["source_kind"], item["source_ref"], destination, zip_file=zip_file)
            manifest_items.append(
                {
                    "sample_id": "{0:05d}".format(index),
                    "source_path": relative_path.as_posix(),
                    "image_path": str((Path("images") / flat_name).as_posix()),
                }
            )
    finally:
        if zip_file is not None:
            zip_file.close()
    manifest = {
        "dataset": "inf30",
        "source_root": str(source_root.as_posix()),
        "output_root": str(output_root.as_posix()),
        "seed": seed,
        "max_samples": max_samples,
        "resize_size": resize_size,
        "num_selected": len(manifest_items),
        "items": manifest_items,
    }
    save_json(manifest, output_root / "manifest.json")
    save_text_lines([item["image_path"] for item in manifest_items], output_root / "selected_images.txt")
    return manifest


def collect_msrs_split_from_dir(split_root: Path, modality: str) -> List[Dict]:
    image_dir = split_root / modality
    label_dir = split_root / "Segmentation_labels"

    if not image_dir.exists():
        print(f"[WARN] 目录不存在: {image_dir}")
        return []
    if not label_dir.exists():
        print(f"[WARN] 目录不存在: {label_dir}")
        return []

    image_index = {}
    for path in sorted(image_dir.iterdir()):
        if path.is_file() and is_image_file(path.name):
            image_index[path.stem] = path

    label_index = {}
    for path in sorted(label_dir.iterdir()):
        if path.is_file() and is_image_file(path.name):
            label_index[path.stem] = path

    common_keys = sorted(set(image_index) & set(label_index))
    if not common_keys:
        print(f"[WARN] 没有找到可配对样本: {split_root}")
        return []

    missing_images = sorted(set(label_index) - set(image_index))
    missing_labels = sorted(set(image_index) - set(label_index))
    if missing_images:
        print(f"[WARN] 缺少图像文件: {image_dir}，示例: {missing_images[:5]}")
    if missing_labels:
        print(f"[WARN] 缺少标签文件: {label_dir}，示例: {missing_labels[:5]}")

    pairs: List[Dict] = []
    for key in common_keys:
        pairs.append(
            {
                "sample_id": key,
                "image_kind": "file",
                "image_ref": image_index[key],
                "image_ext": image_index[key].suffix,
                "label_kind": "file",
                "label_ref": label_index[key],
                "label_ext": label_index[key].suffix,
            }
        )
    return pairs


def collect_msrs_split_from_zip(zip_file: zipfile.ZipFile, split_name: str, modality: str) -> List[Dict]:
    image_index: Dict[str, str] = {}
    label_index: Dict[str, str] = {}

    for member in sorted(zip_file.namelist()):
        if member.endswith("/"):
            continue
        pure = PurePosixPath(member)
        if not is_image_file(pure.name):
            continue

        parts = pure.parts
        if split_name not in parts:
            continue
        split_idx = parts.index(split_name)
        if split_idx + 1 >= len(parts):
            continue

        folder = parts[split_idx + 1]
        stem = pure.stem
        if folder == modality:
            image_index[stem] = member
        elif folder == "Segmentation_labels":
            label_index[stem] = member

    common_keys = sorted(set(image_index) & set(label_index))
    if not common_keys:
        print(f"[WARN] 没有找到可配对样本: {split_name}")
        return []

    missing_images = sorted(set(label_index) - set(image_index))
    missing_labels = sorted(set(image_index) - set(label_index))
    if missing_images:
        print(f"[WARN] 缺少图像文件: {split_name}/{modality}，示例: {missing_images[:5]}")
    if missing_labels:
        print(f"[WARN] 缺少标签文件: {split_name}/Segmentation_labels，示例: {missing_labels[:5]}")

    pairs: List[Dict] = []
    for key in common_keys:
        image_ref = image_index[key]
        label_ref = label_index[key]
        image_ext = PurePosixPath(image_ref).suffix
        label_ext = PurePosixPath(label_ref).suffix
        pairs.append(
            {
                "sample_id": key,
                "image_kind": "zip",
                "image_ref": image_ref,
                "image_ext": image_ext,
                "label_kind": "zip",
                "label_ref": label_ref,
                "label_ext": label_ext,
            }
        )
    return pairs


def split_train_samples(samples: List[Dict], seed: int, val_ratio: float) -> Tuple[List[Dict], List[Dict]]:
    if not samples:
        return [], []

    if not 0.0 < val_ratio < 1.0:
        print(f"[WARN] 非法 val_ratio={val_ratio}，回退为 0.1")
        val_ratio = 0.1

    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * val_ratio))
    val_samples = shuffled[:val_count]
    train_samples = shuffled[val_count:]
    if not train_samples:
        train_samples = [val_samples[0]]
        val_samples = val_samples[1:]
    return train_samples, val_samples


def materialize_msrs_split(
    output_root: Path,
    split_name: str,
    samples: List[Dict],
    split_source_name: str,
    resize_size: int,
    zip_file: Optional[zipfile.ZipFile] = None,
) -> List[Dict]:
    image_out_dir = output_root / split_name / "images"
    label_out_dir = output_root / split_name / "annotations"
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Dict] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        image_ext = sample["image_ext"]
        label_ext = sample["label_ext"]
        image_name = sample_id + image_ext
        label_name = sample_id + label_ext

        image_dst = image_out_dir / image_name
        label_dst = label_out_dir / label_name
        if resize_size > 0:
            save_resized_source_item(sample["image_kind"], sample["image_ref"], image_dst, resize_size, zip_file=zip_file, is_mask=False)
            save_resized_source_item(sample["label_kind"], sample["label_ref"], label_dst, resize_size, zip_file=zip_file, is_mask=True)
        else:
            copy_source_item(sample["image_kind"], sample["image_ref"], image_dst, zip_file=zip_file)
            copy_source_item(sample["label_kind"], sample["label_ref"], label_dst, zip_file=zip_file)

        entries.append(
            {
                "sample_id": sample_id,
                "split": split_name,
                "modality": split_source_name,
                "image_source": str(sample["image_ref"]),
                "label_source": str(sample["label_ref"]),
                "image_path": str((Path(split_name) / "images" / image_name).as_posix()),
                "label_path": str((Path(split_name) / "annotations" / label_name).as_posix()),
            }
        )
    return entries


def prepare_msrs_subset(
    source_root: Path,
    output_root: Path,
    modality: str,
    seed: int,
    val_ratio: float,
    train_max_samples: int,
    test_max_samples: int,
    resize_size: int,
    overwrite: bool,
) -> Dict:
    ensure_clean_dir(output_root, overwrite)

    zip_file = zipfile.ZipFile(source_root, "r") if is_zip_path(source_root) else None
    try:
        if zip_file is not None:
            train_samples = collect_msrs_split_from_zip(zip_file, "train", modality)
            test_samples = collect_msrs_split_from_zip(zip_file, "test", modality)
        else:
            train_samples = collect_msrs_split_from_dir(source_root / "train", modality)
            test_samples = collect_msrs_split_from_dir(source_root / "test", modality)

        if train_max_samples > 0 and len(train_samples) > train_max_samples:
            train_samples = sample_items(train_samples, train_max_samples, seed)

        train_samples, val_samples = split_train_samples(train_samples, seed, val_ratio)

        if test_max_samples > 0 and len(test_samples) > test_max_samples:
            test_samples = sample_items(test_samples, test_max_samples, seed + 1)

        split_payload = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }

        manifest = {
            "dataset": "msrs",
            "source_root": str(source_root.as_posix()),
            "output_root": str(output_root.as_posix()),
            "modality": modality,
            "seed": seed,
            "val_ratio": val_ratio,
            "train_max_samples": train_max_samples,
            "test_max_samples": test_max_samples,
            "resize_size": resize_size,
            "num_classes": len(MSRS_CLASSES),
            "classes": MSRS_CLASSES,
            "palette": MSRS_PALETTE,
            "splits": {},
        }

        for split_name, samples in split_payload.items():
            entries = materialize_msrs_split(output_root, split_name, samples, modality, resize_size, zip_file=zip_file)
            manifest["splits"][split_name] = entries
            save_text_lines([item["sample_id"] for item in entries], output_root / "splits" / "{0}.txt".format(split_name))

        save_json(manifest, output_root / "manifest.json")
        return manifest
    finally:
        if zip_file is not None:
            zip_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare shared Inf30/MSRS subsets from zip or folders")
    parser.add_argument("--inf30-source", default=None, help="Inf30 zip 或目录")
    parser.add_argument("--inf30-output-root", default="data/inf30_subset", help="Inf30 输出目录")
    parser.add_argument("--inf30-count", type=int, default=200, help="Inf30 抽样数量，0 表示全部")

    parser.add_argument("--msrs-source", default=None, help="MSRS zip 或目录")
    parser.add_argument("--msrs-output-root", default="data/msrs_shared", help="MSRS 输出目录")
    parser.add_argument("--msrs-modality", default="ir", choices=["ir", "vi"], help="作为输入的模态")
    parser.add_argument("--msrs-val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--msrs-train-max-samples", type=int, default=200, help="训练集抽样上限，0 表示全部")
    parser.add_argument("--msrs-test-max-samples", type=int, default=100, help="测试集抽样上限，0 表示全部")
    parser.add_argument("--resize-size", type=int, default=224, help="统一缩放尺寸，0 表示不缩放")

    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--overwrite", action="store_true", help="若输出目录已存在则删除重建")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary: Dict[str, Dict] = {}
    if args.inf30_source:
        source_root = Path(args.inf30_source)
        output_root = Path(args.inf30_output_root)
        manifest = prepare_inf30_subset(source_root, output_root, args.inf30_count, args.seed, args.overwrite, args.resize_size)
        summary["inf30"] = {
            "output_root": manifest["output_root"],
            "num_selected": manifest["num_selected"],
            "manifest": str((output_root / "manifest.json").as_posix()),
        }

    if args.msrs_source:
        source_root = Path(args.msrs_source)
        output_root = Path(args.msrs_output_root)
        manifest = prepare_msrs_subset(
            source_root=source_root,
            output_root=output_root,
            modality=args.msrs_modality,
            seed=args.seed,
            val_ratio=args.msrs_val_ratio,
            train_max_samples=args.msrs_train_max_samples,
            test_max_samples=args.msrs_test_max_samples,
            resize_size=args.resize_size,
            overwrite=args.overwrite,
        )
        summary["msrs"] = {
            "output_root": manifest["output_root"],
            "train": len(manifest["splits"]["train"]),
            "val": len(manifest["splits"]["val"]),
            "test": len(manifest["splits"]["test"]),
            "manifest": str((output_root / "manifest.json").as_posix()),
        }

    if not summary:
        print("[WARN] 未提供 --inf30-source 或 --msrs-source，跳过执行")
        return

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
