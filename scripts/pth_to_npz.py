from __future__ import annotations

import torch
import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np


def load_pth_state_dict(pth_path: Path) -> Dict[str, np.ndarray]:

    checkpoint = torch.load(str(pth_path), map_location="cpu")

    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        maybe_state_dict = getattr(checkpoint, "state_dict", None)
        if callable(maybe_state_dict):
            state_dict = maybe_state_dict()
        else:
            print(f"[WARN] 不支持的 checkpoint 格式: {type(checkpoint)}，导出空权重")
            state_dict = {}

    result = {}
    for key, value in state_dict.items():
        if hasattr(value, "cpu"):
            result[key] = value.cpu().numpy()
        elif isinstance(value, np.ndarray):
            result[key] = value
    return result


def save_npz(state_dict: Dict[str, np.ndarray], output_path: Path, prefix: str = "") -> None:
    #保存为.npz
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if prefix:
        state_dict = {prefix + k: v for k, v in state_dict.items()}

    np.savez(str(output_path), **state_dict)
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"已保存: {output_path} ({size_mb:.1f} MB, {len(state_dict)} 个参数)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_dict = load_pth_state_dict(args.input)
    if not state_dict:
        print("[WARN] 未提取到任何参数")
    output_path = args.output or args.input.with_suffix(".npz")
    save_npz(state_dict, output_path, prefix=args.prefix)

if __name__ == "__main__":
    main()
