"""Strict single-device BF16 text loading for the dedicated Qwen3.8 path.

The campaign must authenticate model bytes against its immutable manifest before
calling this loader. This module checks tensor coverage, ownership, shape and
dtype; it does not turn a local index into trusted provenance.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path, PurePosixPath

import torch
from safetensors import safe_open


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate checkpoint JSON key")
        result[key] = value
    return result


def _file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("Invalid checkpoint filename")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or str(relative) != name:
        raise ValueError("Checkpoint path must be canonical and relative")
    result = (root / name).resolve(strict=True)
    if not result.is_relative_to(root) or not result.is_file():
        raise ValueError("Checkpoint file escapes its directory")
    return result


def _parameter_name(key: str) -> str | None:
    if key.startswith("model.language_model."):
        return "model." + key.removeprefix("model.language_model.")
    if key == "lm_head.weight":
        return key
    if key.startswith(("model.visual.", "mtp.")):
        return None
    raise ValueError("Unexpected checkpoint tensor namespace")


def load_weights(model: torch.nn.Module, model_path: str | Path) -> dict:
    """Load every native parameter once; reject incomplete or ambiguous shards."""
    root = Path(model_path).resolve(strict=True)
    index = json.loads(
        _file(root, "model.safetensors.index.json").read_bytes(),
        object_pairs_hook=_object,
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Checkpoint index requires a nonempty weight map")
    parameters = dict(model.named_parameters())
    if not parameters or any(p.dtype != torch.bfloat16 or p.is_meta for p in parameters.values()):
        raise ValueError("Allocate all native parameters as BF16 before loading")
    shards = defaultdict(dict)
    mapped = set()
    skipped = {"vision": 0, "mtp": 0}
    for key, filename in weight_map.items():
        if not isinstance(filename, str) or not filename.endswith(".safetensors"):
            raise ValueError("Invalid indexed shard")
        path = _file(root, filename)
        name = _parameter_name(key)
        if name is None:
            skipped["vision" if key.startswith("model.visual.") else "mtp"] += 1
        elif name not in parameters or name in mapped:
            raise ValueError("Unexpected or duplicated text parameter")
        else:
            mapped.add(name)
        shards[path][key] = name
    if mapped != set(parameters):
        raise ValueError("Checkpoint is missing native text parameters")

    # Check all headers before mutating model parameters. Also inspect excluded
    # tensors' index membership so unnoticed duplicate shards cannot be loaded.
    for path, names in shards.items():
        with safe_open(path, framework="pt", device="cpu") as shard:
            if set(shard.keys()) != set(names):
                raise ValueError("Safetensors contents differ from index ownership")
            for key, name in names.items():
                if name is None:
                    continue
                tensor = shard.get_slice(key)
                if tensor.get_dtype() != "BF16":
                    raise ValueError("Text checkpoint tensors must be BF16")
                if tuple(tensor.get_shape()) != tuple(parameters[name].shape):
                    raise ValueError("Checkpoint tensor shape mismatch")
    with torch.no_grad():
        for path, names in shards.items():
            with safe_open(path, framework="pt", device="cpu") as shard:
                for key, name in names.items():
                    if name is not None:
                        parameters[name].copy_(shard.get_tensor(key))
    return {"loaded_parameters": len(mapped), "excluded_tensors": skipped,
            "scope": "text_only", "dtype": "bfloat16"}


def load_checkpoint(model_path: str | Path, *, device: str = "cuda:0"):
    """Create only the exact supported text architecture on one visible GPU."""
    from nanovllm.qwen38.config import Qwen38TextConfig
    from nanovllm.qwen38.model import Qwen38ForCausalLM

    if device != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Dedicated engine requires exactly one visible CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise ValueError("Dedicated engine requires BF16 support")
    root = Path(model_path).resolve(strict=True)
    raw = json.loads(_file(root, "config.json").read_bytes(), object_pairs_hook=_object)
    config = Qwen38TextConfig.from_dict(raw, strict=True)
    with torch.device("meta"):
        model = Qwen38ForCausalLM(config).to(dtype=torch.bfloat16)
    model.to_empty(device=device)
    report = load_weights(model, root)
    model.eval()
    return model, report
