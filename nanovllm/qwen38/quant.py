"""Strict native W8A8 linear modules; no weight-dequantization fallback."""

from __future__ import annotations

import os

import torch
from torch import nn

from .config import FP8_QUANTIZATION


class FP8Linear(nn.Module):
    """E4M3 weights, FP32 block multipliers and dynamic per-token activations.

    The checkpoint calls its dequantization multiplier ``weight_scale_inv``.
    It is used directly, never reciprocated. BF16 checkpoint multipliers are
    represented exactly in FP32. Only the GEMM operands are FP8; output is BF16.
    """

    backend = "triton_block_fp8"
    kernel_name = "qwen38_block_fp8"

    def __init__(self, in_features: int, out_features: int, *, device=None):
        super().__init__()
        requested = os.environ.get("NANOVLLM_FP8_GEMM", "triton")
        if requested not in ("triton", self.backend):
            raise ValueError(
                f"Unsupported NANOVLLM_FP8_GEMM={requested!r}; this build implements "
                f"only {self.backend}. No fallback was selected."
            )
        if in_features <= 0 or out_features <= 0:
            raise ValueError("FP8 linear dimensions must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(
            out_features, in_features, device=device, dtype=torch.float8_e4m3fn,
        ), requires_grad=False)
        self.weight_scale_inv = nn.Parameter(torch.empty(
            (out_features + 127) // 128, (in_features + 127) // 128,
            device=device, dtype=torch.float32,
        ), requires_grad=False)
        self.native_fp8_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim < 2 or inputs.shape[-1] != self.in_features or inputs.numel() == 0:
            raise ValueError("FP8 linear requires nonempty [..., K] activations")
        if inputs.dtype != torch.bfloat16 or self.weight.dtype != torch.float8_e4m3fn or self.weight_scale_inv.dtype != torch.float32:
            raise ValueError("FP8 linear dtype contract was changed")
        if inputs.device.type != "cuda" or inputs.device != self.weight.device or inputs.device != self.weight_scale_inv.device:
            raise ValueError("Native FP8 execution requires colocated CUDA tensors")
        if torch.cuda.get_device_capability(inputs.device) != (9, 0):
            raise ValueError("Dedicated FP8 kernel requires SM90")
        from ._triton_fp8 import matmul

        result = matmul(inputs.contiguous(), self.weight, self.weight_scale_inv)
        self.native_fp8_calls += 1
        return result


def convert_fp8_linears(model: nn.Module, exclusions: tuple[str, ...]) -> int:
    """Replace precisely the supported text projections, checking HF exclusions.

    Call after allocating ordinary parameters as BF16 on meta, before to_empty.
    Conversion is structural: it does not quantize or dequantize model weights.
    """
    excluded = tuple(name.replace("model.language_model.", "model.") for name in exclusions)
    selected = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        skip = name == "lm_head" or name.endswith((".linear_attn.in_proj_a", ".linear_attn.in_proj_b"))
        declared_skip = any(name == key or name.startswith(key + ".") for key in excluded)
        if skip != declared_skip:
            raise ValueError("FP8 exclusions disagree with supported text projections")
        if skip:
            continue
        suffixes = (
            ".linear_attn.in_proj_qkv", ".linear_attn.in_proj_z", ".linear_attn.out_proj",
            ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj",
            ".mlp.gate_proj", ".mlp.up_proj", ".mlp.down_proj",
        )
        if not name.startswith("model.layers.") or not name.endswith(suffixes) or module.bias is not None:
            raise ValueError("Unexpected text linear in FP8 conversion")
        selected.append((name, module))
    if not selected:
        raise ValueError("No supported FP8 text projections")
    for name, module in selected:
        parent_name, attribute = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), attribute, FP8Linear(
            module.in_features, module.out_features, device=module.weight.device,
        ))
    return len(selected)


def fp8_report(count: int) -> dict:
    return {
        "dtype": "fp8", "activation_dtype": "bfloat16",
        "weight_dtype": "float8_e4m3fn", "scale_dtype": "float32",
        "checkpoint_scale_dtype": "bfloat16", "quantization": dict(FP8_QUANTIZATION),
        "execution": "native_fp8", "fp8_linear_modules": count,
        "dequantized": False, "backend": FP8Linear.backend,
        "kernel_name": FP8Linear.kernel_name,
    }
