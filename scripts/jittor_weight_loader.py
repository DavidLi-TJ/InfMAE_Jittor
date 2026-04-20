"""Utilities for loading converted weights into Jittor models."""
from __future__ import annotations
import jittor as jt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Tuple

import numpy as np


@dataclass
class WeightLoadReport:
    total_model_params: int
    matched_params: int
    skipped_shape_mismatch: int
    missing_params: int
    extra_npz_params: int
    missing_keys_preview: List[str]
    mismatch_preview: List[str]


def _tensor_to_numpy(tensor) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    if hasattr(tensor, "numpy"):
        return tensor.numpy()

    data = getattr(tensor, "data", None)
    if isinstance(data, np.ndarray):
        return data
    if hasattr(data, "numpy"):
        return data.numpy()

    return np.asarray(tensor)


def load_weight_archive(weight_path) -> Dict[str, np.ndarray]:
    """读取 npz 或 pkl 权重包。"""
    weight_path = Path(weight_path)
    suffix = weight_path.suffix.lower()
    if suffix == ".npz":
        archive = np.load(weight_path, allow_pickle=False)
        return {key: archive[key] for key in archive.files}
    archive = jt.load(str(weight_path))
    if isinstance(archive, Mapping):
        state = archive.get("model", archive)
        return {key: _tensor_to_numpy(value) for key, value in state.items()}

    raise TypeError(f"不支持的权重文件格式: {weight_path}")


def normalize_name(name: str) -> str:
    """做最常见的参数名前缀清洗。"""
    normalized = name
    for prefix in ("module.", "backbone.", "encoder.", "model."):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized

def build_candidate_names(model_key: str) -> List[str]:
    """为模型参数名生成候选匹配列表。"""
    base = model_key
    normalized = normalize_name(model_key)

    candidates = [
        base,
        normalized,
        f"model.{base}",
        f"model.{normalized}",
        f"module.{base}",
        f"module.{normalized}",
        f"backbone.{base}",
        f"backbone.{normalized}",
        f"encoder.{base}",
        f"encoder.{normalized}",
    ]

    # 去重并保持顺序
    dedup: List[str] = []
    seen = set()
    for key in candidates:
        if key not in seen:
            seen.add(key)
            dedup.append(key)
    return dedup


def _extract_state_dict(model) -> Mapping[str, object]:
    """抽取模型 state_dict。"""
    if hasattr(model, "state_dict"):
        state = model.state_dict()
        if isinstance(state, Mapping):
            return state
    raise TypeError("模型对象不支持 state_dict() 或返回值不是映射")


def _shape_of_param(param_obj) -> Tuple[int, ...]:
    """读取参数形状并标准化成 tuple。"""
    shape = getattr(param_obj, "shape", None)
    if shape is None:
        return tuple()
    try:
        return tuple(int(x) for x in shape)
    except Exception:
        return tuple()


def match_weights(model_state: Mapping[str, object], npz_state: Mapping[str, np.ndarray], strict_shape: bool = True) -> Tuple[MutableMapping[str, np.ndarray], WeightLoadReport]:
    """按参数名匹配模型参数与 npz 权重。"""
    matched: MutableMapping[str, np.ndarray] = {}
    missing_keys: List[str] = []
    mismatch_keys: List[str] = []

    for model_key, model_param in model_state.items():
        target_shape = _shape_of_param(model_param)
        found = False
        for candidate in build_candidate_names(model_key):
            tensor = npz_state.get(candidate)
            if tensor is None:
                continue

            if strict_shape and target_shape and tuple(tensor.shape) != target_shape:
                mismatch_keys.append(
                    f"{model_key}: model={target_shape}, npz={tuple(tensor.shape)}, src={candidate}"
                )
                found = True
                break

            matched[model_key] = tensor
            found = True
            break

        if not found:
            missing_keys.append(model_key)

    extra_count = max(0, len(npz_state) - len(matched))
    report = WeightLoadReport(
        total_model_params=len(model_state),
        matched_params=len(matched),
        skipped_shape_mismatch=len(mismatch_keys),
        missing_params=len(missing_keys),
        extra_npz_params=extra_count,
        missing_keys_preview=missing_keys[:20],
        mismatch_preview=mismatch_keys[:20],
    )
    return matched, report


def apply_weights(model, matched_weights: Mapping[str, np.ndarray]) -> None:
    """将已匹配权重写入模型。"""
    if hasattr(model, "load_parameters"):
        model.load_parameters(dict(matched_weights))
        return

    if hasattr(model, "load_state_dict"):
        try:
            model.load_state_dict(dict(matched_weights), strict=False)
        except TypeError:
            model.load_state_dict(dict(matched_weights))
        return

    raise TypeError("模型不支持 load_parameters 或 load_state_dict")


def load_weights_into_model(model, weight_path: Path, strict_shape: bool = True) -> WeightLoadReport:
    """完整执行：读取 npz -> 匹配 -> 注入模型。"""
    npz_state = load_weight_archive(weight_path)
    model_state = _extract_state_dict(model)
    matched, report = match_weights(model_state, npz_state, strict_shape=strict_shape)
    apply_weights(model, matched)
    return report


def format_report(report: WeightLoadReport) -> str:
    """将加载报告格式化成可读文本。"""
    lines = [
        f"total_model_params: {report.total_model_params}",
        f"matched_params: {report.matched_params}",
        f"skipped_shape_mismatch: {report.skipped_shape_mismatch}",
        f"missing_params: {report.missing_params}",
        f"extra_npz_params: {report.extra_npz_params}",
    ]
    if report.missing_keys_preview:
        lines.append("missing_keys_preview:")
        lines.extend([f"  - {item}" for item in report.missing_keys_preview])
    if report.mismatch_preview:
        lines.append("mismatch_preview:")
        lines.extend([f"  - {item}" for item in report.mismatch_preview])
    return "\n".join(lines)
