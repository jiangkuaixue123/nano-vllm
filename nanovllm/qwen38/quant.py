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
        self.deepgemm_quantization = os.environ.get("NANOVLLM_DEEPGEMM_QUANTIZATION", "native")
        if self.deepgemm_quantization not in ("native", "row4"):
            raise ValueError(f"Unsupported NANOVLLM_DEEPGEMM_QUANTIZATION={self.deepgemm_quantization!r}")
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
        self.deepgemm_quantization_calls = {"native": 0, "row4": 0}

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

    def _prepare_silu_fp8(self, device):
        """Source-private M1 down projection; warm the full bridge before capture."""
        from ._silu_fp8 import projection_reason

        reason = projection_reason(self, device)
        if reason:
            raise ValueError("SwiGLU FP8 projection: " + reason)
        with torch.cuda.device(device):
            if (torch.cuda.is_current_stream_capturing()
                    and getattr(self, "_silu_fp8_ready_device", None) != device):
                raise RuntimeError("Warm the SwiGLU FP8 path eagerly before CUDA Graph capture")
            from flashinfer.gemm import fp8_blockscale_gemm_sm90

        return fp8_blockscale_gemm_sm90

    def _forward_silu_prequantized(self, values, scales, input_shape):
        from ._silu_fp8 import validate_prequantized

        validate_prequantized(self, values, scales, input_shape)
        with torch.cuda.device(values.device):
            gemm = self._prepare_silu_fp8(values.device)
            result = gemm(values, self.weight, input_scale=scales,
                          weight_scale=self.weight_scale_inv, out_dtype=torch.bfloat16)
        self.kernel_calls["flashinfer"] += 1
        self.native_fp8_calls += 1
        return result.reshape(*input_shape[:-1], self.out_features)

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
            if self.deepgemm_quantization == "row4":
                from ._triton_quant4 import matmul
            else:
                from ._deepgemm_fp8 import matmul

            result = matmul(inputs.reshape(rows, self.in_features).contiguous(),
                            self.weight, self.weight_scale_inv)
            result = result.reshape(*inputs.shape[:-1], self.out_features)
            self.kernel_calls["deepgemm"] += 1
            self.deepgemm_quantization_calls[self.deepgemm_quantization] += 1
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

    def _prepare_postnorm_fp8(self, device):
        """Import the public entry; a complete eager call must precede capture."""
        from ._postnorm_fp8 import projection_reason

        reason = projection_reason(self, device)
        if reason:
            raise ValueError("Post-attention FP8 projection: " + reason)
        with torch.cuda.device(device):
            if (torch.cuda.is_current_stream_capturing()
                    and getattr(self, "_postnorm_fp8_ready_device", None) != device):
                raise RuntimeError("Warm the post-attention FP8 path eagerly before CUDA Graph capture")
            from flashinfer.gemm import fp8_blockscale_gemm_sm90

        return fp8_blockscale_gemm_sm90

    def forward_prequantized(self, values, scales, input_shape):
        """Source-private postnorm consumer; ordinary forward still accepts BF16."""
        from ._postnorm_fp8 import validate_prequantized

        validate_prequantized(self, values, scales, input_shape)
        with torch.cuda.device(values.device):
            gemm = self._prepare_postnorm_fp8(values.device)
            result = gemm(values, self.weight, input_scale=scales,
                          weight_scale=self.weight_scale_inv, out_dtype=torch.bfloat16)
        self.kernel_calls["flashinfer"] += 1
        self.native_fp8_calls += 1
        return result.reshape(*input_shape[:-1], self.out_features).split(self.out_sizes, dim=-1)

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
                or len({part.deepgemm_quantization for part in parts}) != 1
                or len({part.backend for part in parts}) != 1):
            raise ValueError("FP8 projection fusion requires unallocated compatible FP8 linears")
        first = parts[0]
        merged = FP8MergedLinear(
            first.in_features, [part.out_features for part in parts], names, device="meta",
        )
        # Backend selection belongs to the original constructed modules.
        for attribute in ("use_flashinfer", "use_deepgemm", "backend", "kernel_name", "deepgemm_quantization"):
            setattr(merged, attribute, getattr(first, attribute))
        groups.append((parent, target, names, merged))
    if not groups:
        raise ValueError("No supported FP8 projection groups")
    for parent, target, names, merged in groups:
        for name in names:
            delattr(parent, name)
        setattr(parent, target, merged)
    return len(groups)


def fuse_bf16_gdn_ba_projections(model):
    """Merge excluded BF16 b/a rows after FP8 conversion, before allocation."""
    from .model import DeltaNet, FusedProjection

    groups = []
    names = ("in_proj_b", "in_proj_a")
    for parent in model.modules():
        if not isinstance(parent, DeltaNet):
            continue
        qkv = getattr(parent, "in_proj_qkvz", None)
        if qkv is None:
            qkv = getattr(parent, "in_proj_qkv", None)
        parts = [getattr(parent, name, None) for name in names]
        if (not isinstance(qkv, FP8Linear) or not qkv.weight.is_meta
                or not qkv.weight_scale_inv.is_meta
                or not all(type(part) is nn.Linear for part in parts)
                or not all(part.weight.is_meta and part.weight.dtype == torch.bfloat16
                           and part.bias is None
                           and part.in_features == parent.config.hidden_size
                           and part.out_features == parent.config.linear_num_value_heads
                           for part in parts)):
            raise ValueError("BF16 GDN b/a fusion requires unallocated excluded linears after FP8 conversion")
        first = parts[0]
        merged = FusedProjection(
            first.in_features, [part.out_features for part in parts], names,
            device="meta", dtype=torch.bfloat16,
        )
        groups.append((parent, merged))
    if not groups:
        raise ValueError("No supported BF16 GDN b/a projections")
    # Validate every layer before discarding any original parameter.
    for parent, merged in groups:
        for name in names:
            delattr(parent, name)
        parent.in_proj_ba = merged
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
    quantizers = {(module.deepgemm_quantization, module.use_deepgemm) for module in modules}
    if len(backends) != 1 or len(quantizers) != 1:
        raise ValueError("FP8 loading requires one consistent backend selection")
    backend, kernel_name = next(iter(backends))
    quantizer, active = next(iter(quantizers))
    return {
        "dtype": "fp8", "activation_dtype": "bfloat16",
        "weight_dtype": "float8_e4m3fn", "scale_dtype": "float32",
        "checkpoint_scale_dtype": "bfloat16", "quantization": dict(FP8_QUANTIZATION),
        "execution": "native_fp8", "fp8_linear_modules": len(modules),
        "dequantized": False, "backend": backend,
        "kernel_name": kernel_name,
        "deepgemm_quantization": {
            "backend": quantizer, "active": active, "group_size": 128,
            "rows_per_program": 4 if quantizer == "row4" else 1,
            "scale_dtype": "float32", "scale_layout": "column_major_padded4",
        },
    }
