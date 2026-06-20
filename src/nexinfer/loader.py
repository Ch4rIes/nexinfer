from __future__ import annotations

from collections.abc import Callable
from glob import glob
import os
from typing import Any


SafeOpen = Callable[[str, str, str], Any]


def default_weight_loader(param: Any, loaded_weight: Any) -> None:
    """Load one tensor into a parameter using Nano-VLLM's default convention."""

    param.data.copy_(loaded_weight)


def load_model(
    model: Any,
    path: str,
    *,
    safe_open_fn: SafeOpen | None = None,
) -> None:
    """Load safetensors weights into a model using Nano-VLLM-style loaders."""

    if safe_open_fn is None:
        try:
            from safetensors import safe_open as safe_open_fn
        except ImportError as exc:
            raise ImportError(
                "Install safetensors to load model weights from .safetensors files."
            ) from exc

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file_path in glob(os.path.join(path, "*.safetensors")):
        with safe_open_fn(file_path, "pt", "cpu") as opened:
            for weight_name in opened.keys():
                if _load_packed_weight(
                    model,
                    opened,
                    weight_name,
                    packed_modules_mapping,
                ):
                    continue
                param = model.get_parameter(weight_name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, opened.get_tensor(weight_name))


def _load_packed_weight(
    model: Any,
    opened: Any,
    weight_name: str,
    packed_modules_mapping: dict[str, tuple[str, Any]],
) -> bool:
    for packed_name, (param_name_part, shard_id) in packed_modules_mapping.items():
        if packed_name not in weight_name:
            continue
        param_name = weight_name.replace(packed_name, param_name_part)
        param = model.get_parameter(param_name)
        weight_loader = getattr(param, "weight_loader")
        weight_loader(param, opened.get_tensor(weight_name), shard_id)
        return True
    return False
