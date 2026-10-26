"""Decode input RMS/FP8 bridge for the original QKV or QKVZ consumer pair."""

import math

import torch

from .quant import FP8MergedLinear


def projection_reason(projection, device, keep_bf16):
    if type(keep_bf16) is not bool:
        return "requires_explicit_bf16_fanout"
    if type(projection) is not FP8MergedLinear:
        return "requires_merged_fp8_input_projection"
    names, sizes = (
        (("in_proj_qkv", "in_proj_z"), (10240, 6144))
        if keep_bf16
        else (("q_proj", "k_proj", "v_proj"), (12288, 1024, 1024))
    )
    if (
        projection.names != names
        or projection.out_sizes != sizes
        or projection.in_features != 5120
        or projection.out_features != sum(sizes)
    ):
        return "unsupported_input_projection"
    if not projection.use_flashinfer:
        return "requires_flashinfer_small_m"
    weight, scales = projection.weight, projection.weight_scale_inv
    if weight.dtype != torch.float8_e4m3fn or scales.dtype != torch.float32:
        return "invalid_checkpoint_dtypes"
    if weight.device != device or scales.device != device:
        return "checkpoint_device_mismatch"
    if (
        weight.shape != (sum(sizes), 5120)
        or scales.shape != (sum(sizes) // 128, 40)
        or not weight.is_contiguous()
        or not scales.is_contiguous()
        or weight.requires_grad
        or scales.requires_grad
    ):
        return "invalid_checkpoint_layout"
    return None


def execution(x, norm, projection, backend, *, keep_bf16, fresh_prefill=False):
    """Read metadata only; do not populate the original effective-weight cache."""
    from .model import RMSNorm

    if (
        backend not in ("native", "triton")
        or type(keep_bf16) is not bool
        or type(fresh_prefill) is not bool
    ):
        raise ValueError("Invalid input RMS/FP8 selection")
    reason = None
    if backend == "native":
        reason = "configured_native"
    elif fresh_prefill:
        reason = "prefill_native"
    elif x.device.type != "cuda" or getattr(torch.version, "hip", None):
        reason = "requires_nvidia_cuda"
    elif x.shape not in ((1, 5120), (1, 1, 5120)):
        reason = "requires_m1_k5120"
    elif x.dtype != torch.bfloat16 or not x.is_contiguous() or x.requires_grad:
        reason = "requires_contiguous_inference_bf16"
    elif torch.cuda.get_device_capability(x.device) != (9, 0):
        reason = "requires_sm90"
    elif (
        type(norm) is not RMSNorm
        or norm.weight.shape != (5120,)
        or norm.weight.dtype != torch.bfloat16
        or norm.weight.device != x.device
        or not norm.weight.is_contiguous()
        or not isinstance(norm.eps, (float, int))
        or not math.isfinite(norm.eps)
        or norm.eps <= 0
    ):
        reason = "unsupported_input_norm"
    else:
        reason = projection_reason(projection, x.device, keep_bf16)
    return {
        "requested_backend": backend,
        "selected_backend": "native" if reason else "triton",
        "fallback_reason": reason,
        "execution_observed": False,
        "scope": "m1_input_rms_fp8",
        "keep_bf16": keep_bf16,
        "quantizer": "fi_block128_fp32_scales",
    }


def validate_producer(x, gamma, eps, keep_bf16):
    if (
        type(keep_bf16) is not bool
        or x.device.type != "cuda"
        or getattr(torch.version, "hip", None)
        or torch.cuda.get_device_capability(x.device) != (9, 0)
        or x.shape not in ((1, 5120), (1, 1, 5120))
        or gamma.shape != (5120,)
        or not isinstance(eps, (float, int))
        or not math.isfinite(eps)
        or eps <= 0
    ):
        raise ValueError("Input RMS/FP8 requires SM90 M1/K5120 and positive epsilon")
    if any(
        t.device != x.device
        or t.dtype != torch.bfloat16
        or not t.is_contiguous()
        or t.requires_grad
        for t in (x, gamma)
    ):
        raise ValueError("Input RMS/FP8 requires contiguous colocated inference BF16")


def validate_prequantized(projection, values, scales, input_shape, keep_bf16):
    if type(keep_bf16) is not bool:
        raise ValueError("Input projection requires an explicit BF16 fanout mode")
    reason = projection_reason(projection, values.device, keep_bf16)
    if reason:
        raise ValueError("Input RMS/FP8 projection: " + reason)
    if (
        values.device.type != "cuda"
        or getattr(torch.version, "hip", None)
        or torch.cuda.get_device_capability(values.device) != (9, 0)
        or tuple(input_shape) not in ((1, 5120), (1, 1, 5120))
        or values.shape != (1, 5120)
        or values.dtype != torch.float8_e4m3fn
        or not values.is_contiguous()
        or values.requires_grad
    ):
        raise ValueError("Input projection requires contiguous SM90 FP8 [1,5120]")
    if (
        scales.device != values.device
        or scales.dtype != torch.float32
        or scales.shape != (1, 40)
        or scales.stride() != (1, 4)
        or scales.requires_grad
        or scales.untyped_storage().nbytes() - scales.storage_offset() * 4 < 640
    ):
        raise ValueError(
            "Input scales require live [1,40] stride (1,4), full [40,4] backing"
        )


def project(x, norm, projection, *, keep_bf16):
    """Selected complete producer/consumer; failures propagate, input is read-only."""
    route = execution(x, norm, projection, "triton", keep_bf16=keep_bf16)
    if route["selected_backend"] != "triton":
        raise ValueError(
            "Ineligible selected input RMS/FP8 pair: " + route["fallback_reason"]
        )
    with torch.cuda.device(x.device):
        projection._prepare_input_rms_fp8(x.device, keep_bf16)
        gamma = norm._fused_weight(x.dtype)
        validate_producer(x, gamma, norm.eps, keep_bf16)
        from ._triton_input_rms_fp8 import rms_quant

        normalized, values, scales = rms_quant(x, gamma, norm.eps, keep_bf16=keep_bf16)
        outputs = projection._forward_input_prequantized(
            values, scales, x.shape, keep_bf16
        )
        # The complete eager pair, not a metadata query or import, establishes readiness.
        projection._input_rms_fp8_ready_device = x.device
    return normalized, outputs
