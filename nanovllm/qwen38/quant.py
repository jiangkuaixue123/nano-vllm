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
        if requested not in ("triton", self.backend, "flashinfer", "flashinfer_deepgemm"):
            raise ValueError(
                f"Unsupported NANOVLLM_FP8_GEMM={requested!r}; this build implements "
                "triton, flashinfer and flashinfer_deepgemm. No fallback was selected."
            )
        self.use_flashinfer = requested in ("flashinfer", "flashinfer_deepgemm")
        self.use_deepgemm = requested == "flashinfer_deepgemm"
        if self.use_flashinfer:
            self.backend = "flashinfer_small_m_triton_prefill"
            self.kernel_name = "fp8_blockscale_gemm_sm90+qwen38_block_fp8"
        if self.use_deepgemm:
            self.backend = "flashinfer_small_m_deepgemm_prefill"
            self.kernel_name = "fp8_blockscale_gemm_sm90+deep_gemm.fp8_gemm_nt"
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
        self.kernel_calls = {"flashinfer": 0, "triton": 0, "deepgemm": 0}

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim < 2 or inputs.shape[-1] != self.in_features or inputs.numel() == 0:
            raise ValueError("FP8 linear requires nonempty [..., K] activations")
        if inputs.dtype != torch.bfloat16 or self.weight.dtype != torch.float8_e4m3fn or self.weight_scale_inv.dtype != torch.float32:
            raise ValueError("FP8 linear dtype contract was changed")
        if inputs.device.type != "cuda" or inputs.device != self.weight.device or inputs.device != self.weight_scale_inv.device:
            raise ValueError("Native FP8 execution requires colocated CUDA tensors")
        if torch.cuda.get_device_capability(inputs.device) != (9, 0):
            raise ValueError("Dedicated FP8 kernel requires SM90")
        result = self._matmul(inputs)
        self.native_fp8_calls += 1
        return result

    def _matmul(self, inputs: torch.Tensor) -> torch.Tensor:
        # Match vLLM's small-M SM90 choice; prefill follows the selected backend.
        # FlashInfer's BF16 input path quantizes activations internally.
        rows = inputs.numel() // self.in_features
        if self.use_flashinfer and rows < 32:
            if self.in_features % 128 or self.out_features % 64:
                raise ValueError("FlashInfer FP8 requires K divisible by 128 and N by 64")
            from flashinfer.gemm import fp8_blockscale_gemm_sm90

            result = fp8_blockscale_gemm_sm90(
                inputs.reshape(rows, self.in_features).contiguous(), self.weight,
                weight_scale=self.weight_scale_inv, out_dtype=torch.bfloat16,
            ).reshape(*inputs.shape[:-1], self.out_features)
            self.kernel_calls["flashinfer"] += 1
        elif self.use_deepgemm:
            from ._deepgemm_fp8 import matmul

            result = matmul(inputs.reshape(rows, self.in_features).contiguous(),
                            self.weight, self.weight_scale_inv)
            result = result.reshape(*inputs.shape[:-1], self.out_features)
            self.kernel_calls["deepgemm"] += 1
        else:
            from ._triton_fp8 import matmul

            result = matmul(inputs.contiguous(), self.weight, self.weight_scale_inv)
            self.kernel_calls["triton"] += 1
        return result


class FP8MergedLinear(FP8Linear):
    """Concatenate independent output rows without changing FP8 blocks.

    Every original projection starts at a 128-row scale boundary. Checkpoint
    weights and scales load directly into slices of the single allocation.
    """

    def __init__(self, in_features, out_sizes, names, *, device=None):
        self.out_sizes = tuple(out_sizes)
        self.names = tuple(names)
        if (not self.out_sizes or len(self.out_sizes) != len(self.names)
                or len(set(self.names)) != len(self.names)
                or in_features % 128
                or any(size <= 0 or size % 128 for size in self.out_sizes)):
            raise ValueError("Merged FP8 projections require distinct names and 128-aligned blocks")
        super().__init__(in_features, sum(self.out_sizes), device=device)

    def part_ranges(self):
        start = 0
        ranges = {}
        for name, size in zip(self.names, self.out_sizes):
            ranges[name] = (start, start + size)
            start += size
        return ranges

    def forward(self, inputs):
        return super().forward(inputs).split(self.out_sizes, dim=-1)


def fuse_fp8_projections(model):
    """Structural fusion on meta only, before allocating or loading weights."""
    from .model import MLP, DeltaNet, FullAttention

    groups = []
    for parent in model.modules():
        if isinstance(parent, MLP):
            target, names = "gate_up_proj", ("gate_proj", "up_proj")
        elif isinstance(parent, FullAttention):
            target, names = "qkv_proj", ("q_proj", "k_proj", "v_proj")
        elif isinstance(parent, DeltaNet):
            target, names = "in_proj_qkvz", ("in_proj_qkv", "in_proj_z")
        else:
            continue
        parts = [getattr(parent, name, None) for name in names]
        if (not all(type(part) is FP8Linear for part in parts)
                or not all(part.weight.is_meta and part.weight_scale_inv.is_meta for part in parts)
                or len({part.in_features for part in parts}) != 1
                or len({part.backend for part in parts}) != 1):
            raise ValueError("FP8 projection fusion requires unallocated compatible FP8 linears")
        first = parts[0]
        merged = FP8MergedLinear(
            first.in_features, [part.out_features for part in parts], names, device="meta",
        )
        # Backend selection belongs to the original constructed modules.
        for attribute in ("use_flashinfer", "use_deepgemm", "backend", "kernel_name"):
            setattr(merged, attribute, getattr(first, attribute))
        groups.append((parent, target, names, merged))
    if not groups:
        raise ValueError("No supported FP8 projection groups")
    for parent, target, names, merged in groups:
        for name in names:
            delattr(parent, name)
        setattr(parent, target, merged)
    return len(groups)


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


def fp8_report(modules) -> dict:
    modules = tuple(modules)
    backends = {(module.backend, module.kernel_name) for module in modules}
    if len(backends) != 1:
        raise ValueError("FP8 loading requires one consistent backend selection")
    backend, kernel_name = next(iter(backends))
    return {
        "dtype": "fp8", "activation_dtype": "bfloat16",
        "weight_dtype": "float8_e4m3fn", "scale_dtype": "float32",
        "checkpoint_scale_dtype": "bfloat16", "quantization": dict(FP8_QUANTIZATION),
        "execution": "native_fp8", "fp8_linear_modules": len(modules),
        "dequantized": False, "backend": backend,
        "kernel_name": kernel_name,
    }
