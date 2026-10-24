"""Strict single-device BF16/block-FP8 loading for the dedicated text path.

The campaign must authenticate model bytes against its immutable manifest before
calling this loader. This module checks tensor coverage, ownership, shape and
dtype; it does not turn a local index into trusted provenance.
"""

from __future__ import annotations

import json
import os
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
    from .model import MLP, DeltaNet, FullAttention, FusedProjection
    from .quant import FP8Linear, FP8MergedLinear, fp8_report

    gdn_modules = [module for module in model.modules() if isinstance(module, DeltaNet)]
    gdn_layouts = {(module.gdn_backend, module.gdn_prefill_qk_layout,
                    module.config.linear_num_key_heads, module.config.linear_num_value_heads)
                   for module in gdn_modules}
    if len(gdn_layouts) > 1:
        raise ValueError("GDN layers disagree on prefill execution layout")
    outnorm_backends = {module.norm.outnorm_backend for module in gdn_modules}
    if len(outnorm_backends) > 1:
        raise ValueError("Mixed GDN output norm backends are unsupported")
    gdn_convolutions = {(module.gdn_conv_backend, module.config.linear_conv_kernel_dim)
                        for module in gdn_modules}
    if len(gdn_convolutions) > 1:
        raise ValueError("GDN layers disagree on convolution backend or window")
    gdn_qk_norms = {module.gdn_qk_norm_backend for module in gdn_modules}
    if len(gdn_qk_norms) > 1:
        raise ValueError("Mixed whole Q/K normalization backends are unsupported")
    parameters = dict(model.named_parameters())
    mlps = [module for module in model.modules() if isinstance(module, MLP)]
    postnorm_backends = {module.postnorm_fp8_backend for module in mlps}
    if len(postnorm_backends) > 1:
        raise ValueError("Mixed post-attention FP8 backends are unsupported")
    activation_backends = {module.activation_backend for module in mlps}
    if len(activation_backends) > 1:
        raise ValueError("Mixed MLP activation backends are unsupported")
    full_attention = [module for module in model.modules() if isinstance(module, FullAttention)]
    rope_backends = {module.rope_backend for module in full_attention}
    if len(rope_backends) > 1:
        raise ValueError("Mixed full-attention RoPE backends are unsupported")
    prefill_backends = {module.prefill_attention for module in full_attention}
    if len(prefill_backends) > 1:
        raise ValueError("Mixed full-attention prefill backends are unsupported")
    # 融合后的投影只有一份权重：把 checkpoint 里的 q_proj/k_proj/v_proj 等
    # 映射到这份权重的行切片视图上，加载时直接原地写入，不额外占显存。
    fused = [(name, module) for name, module in model.named_modules()
             if isinstance(module, (FusedProjection, FP8MergedLinear))]
    for name, _module in fused:
        parameters.pop(name + ".weight", None)
        if isinstance(_module, FP8MergedLinear):
            parameters.pop(name + ".weight_scale_inv")
    fp8_parameter_names = set()
    for name, module in fused:
        # 融合模块挂在 <parent>.qkv_proj / .in_proj_fused / .gate_up_proj 下，
        # 而 checkpoint 里是 <parent>.q_proj.weight 这种原始名，所以要去掉这一段。
        parent = name.rpartition(".")[0]
        weight = module.weight
        for part, (start, end) in module.part_ranges().items():
            parameters[f"{parent}.{part}.weight"] = weight[start:end]
            if isinstance(module, FP8MergedLinear):
                parameters[f"{parent}.{part}.weight_scale_inv"] = module.weight_scale_inv[start // 128:end // 128]
                fp8_parameter_names.add(f"{parent}.{part}")

    fp8_modules = {name: module for name, module in model.named_modules() if isinstance(module, FP8Linear)}
    fp8_loading_report = fp8_report(fp8_modules.values()) if fp8_modules else {}
    expected_dtypes = {name: (torch.bfloat16, "BF16") for name in parameters}
    fp8_parameter_names.update(name for name, module in fp8_modules.items()
                               if not isinstance(module, FP8MergedLinear))
    for name in fp8_parameter_names:
        expected_dtypes[name + ".weight"] = (torch.float8_e4m3fn, "F8_E4M3")
        expected_dtypes[name + ".weight_scale_inv"] = (torch.float32, "BF16")
    if not parameters or any(p.dtype != expected_dtypes[name][0] or p.is_meta for name, p in parameters.items()):
        raise ValueError("Allocate native parameters with the BF16/FP8 dtype contract before loading")
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
                if tensor.get_dtype() != expected_dtypes[name][1]:
                    raise ValueError("Text checkpoint tensor dtype violates BF16/FP8 contract")
                if tuple(tensor.get_shape()) != tuple(parameters[name].shape):
                    raise ValueError("Checkpoint tensor shape mismatch")
                if name.endswith(".weight_scale_inv"):
                    scale = shard.get_tensor(key).float()
                    if not torch.all(torch.isfinite(scale) & (scale > 0)).item():
                        raise ValueError("FP8 weight scales must be finite and positive")
    with torch.no_grad():
        for path, names in shards.items():
            with safe_open(path, framework="pt", device="cpu") as shard:
                for key, name in names.items():
                    if name is not None:
                        parameters[name].copy_(shard.get_tensor(key))
    report = {"loaded_parameters": len(mapped), "excluded_tensors": skipped,
              "scope": "text_only", "dtype": "bfloat16"}
    if gdn_convolutions:
        conv_backend, conv_width = gdn_convolutions.pop()
        report["gdn_convolution"] = {
            "backend": conv_backend, "layers": len(gdn_modules),
            "state_entries": conv_width, "conv_output_dtype": "bfloat16",
            "fresh_prefill": conv_backend, "single_token_decode": conv_backend,
            "cached_multi_token": "native",
        }
    if gdn_layouts:
        backend, layout, key_heads, value_heads = gdn_layouts.pop()
        report["gdn_prefill"] = {
            "backend": backend, "qk_layout": layout, "layers": len(gdn_modules),
            "query_key_heads": key_heads if layout == "grouped" else value_heads,
            "value_heads": value_heads, "decode_qk_layout": "expanded",
        }
        report["gdn_outnorm"] = {
            "backend": outnorm_backends.pop(), "layers": len(gdn_modules),
            "norm_intermediate_dtype": "bfloat16", "silu_intermediate_dtype": "bfloat16",
        }
        report["gdn_qk_normalization"] = {
            "requested_backend": gdn_qk_norms.pop(), "layers": len(gdn_modules),
            "scope": "grouped_prefill_whole_l2", "candidate_reduction": "triton_fp32_sum",
            "cross_backend_comparison": "diagnostic_reduction_order_differs",
            "fallback": "native_for_unsupported_operand_metadata",
            "decode": "unchanged", "execution_observed": False,
        }
    if mlps:
        report["postnorm_fp8"] = {
            "requested_backend": postnorm_backends.pop(), "layers": len(mlps),
            "scope": "m1_post_attention_gate_up", "norm_intermediate_dtype": "bfloat16",
            "quantizer": "fi_block128_fp32_scales", "scale_backing": [40, 4],
            "fallback": "native_for_unsupported_operand_metadata", "execution_observed": False,
        }
        report["mlp_activation"] = {
            "backend": activation_backends.pop(), "layers": len(mlps),
            "silu_intermediate_dtype": "bfloat16",
        }
    if full_attention:
        report["full_attention_prefill"] = {
            "backend": prefill_backends.pop(), "layers": len(full_attention),
            "scope": "fresh_prefill", "cached_continuation": "native",
        }
        report["full_attention_rope"] = {
            "backend": rope_backends.pop(), "layers": len(full_attention),
            "scope": "postnorm_partial_rope", "norm_backend": "native",
            "product_intermediate_dtype": "bfloat16",
        }
    if fp8_modules:
        report.update(fp8_loading_report)
        report["projection_layout"] = (
            "merged" if any(isinstance(module, FP8MergedLinear) for module in fp8_modules.values())
            else "separate"
        )
        report["gdn_ba_projection_layout"] = (
            "merged" if any(isinstance(module, FusedProjection)
                            and module.names == ("in_proj_b", "in_proj_a")
                            for _, module in fused)
            else "separate"
        )
    return report


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
    projection_layout = os.environ.get("NANOVLLM_FP8_PROJECTIONS", "separate")
    if projection_layout not in ("separate", "merged"):
        raise ValueError("Unsupported NANOVLLM_FP8_PROJECTIONS")
    if projection_layout == "merged" and config.weight_format != "fp8":
        raise ValueError("Merged FP8 projections require an FP8 checkpoint")
    ba_layout = os.environ.get("NANOVLLM_GDN_BA_PROJECTIONS", "separate")
    if ba_layout not in ("separate", "merged"):
        raise ValueError("Unsupported NANOVLLM_GDN_BA_PROJECTIONS")
    if ba_layout == "merged" and config.weight_format != "fp8":
        raise ValueError("Merged BF16 GDN b/a projections require an FP8 checkpoint")
    with torch.device("meta"):
        model = Qwen38ForCausalLM(config).to(dtype=torch.bfloat16)
        if config.weight_format == "fp8":
            from .quant import convert_fp8_linears

            if torch.cuda.get_device_capability(0) != (9, 0):
                raise ValueError("Dedicated FP8 kernel requires SM90")
            # Do not call model.to(dtype=...) after this: FP8 weights and FP32
            # multipliers must retain their distinct representation.
            convert_fp8_linears(model, config.modules_to_not_convert)
            if projection_layout == "merged":
                from .quant import fuse_fp8_projections

                fuse_fp8_projections(model)
            if ba_layout == "merged":
                from .quant import fuse_bf16_gdn_ba_projections

                fuse_bf16_gdn_ba_projections(model)
        else:
            model.fuse_projections()
    model.to_empty(device=device)
    report = load_weights(model, root)
    model.eval()
    return model, report
