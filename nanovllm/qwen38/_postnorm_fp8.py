"""One-token post-attention RMS/FP8 bridge; native fallback is explicit."""

import math

import torch

from .quant import FP8MergedLinear


def _overlap(left, right):
    """Byte intervals for already-validated contiguous tensors; no data reads."""
    a, b = left.data_ptr(), right.data_ptr()
    return a < b + right.numel() * right.element_size() and b < a + left.numel() * left.element_size()


def projection_reason(projection, device):
    if type(projection) is not FP8MergedLinear:
        return "requires_merged_fp8_gate_up"
    if (projection.names != ("gate_proj", "up_proj") or projection.out_sizes != (17408, 17408)
            or projection.in_features != 5120 or projection.out_features != 34816):
        return "unsupported_projection_shape"
    if not projection.use_flashinfer:
        return "requires_flashinfer_small_m"
    weight, scales = projection.weight, projection.weight_scale_inv
    if weight.dtype != torch.float8_e4m3fn or scales.dtype != torch.float32:
        return "invalid_checkpoint_dtypes"
    if weight.device != device or scales.device != device:
        return "checkpoint_device_mismatch"
    if (weight.shape != (34816, 5120) or scales.shape != (272, 40)
            or not weight.is_contiguous() or not scales.is_contiguous()
            or weight.requires_grad or scales.requires_grad):
        return "invalid_checkpoint_layout"
    return None


def execution(residual, delta, projection, backend):
    """Pure metadata query; neither prepares the norm cache nor claims execution."""
    if backend not in ("native", "triton"):
        raise ValueError("Unsupported post-attention FP8 backend")
    reason = None
    if backend == "native":
        reason = "configured_native"
    elif residual.device.type != "cuda" or getattr(torch.version, "hip", None):
        reason = "requires_nvidia_cuda"
    elif residual.device != delta.device:
        reason = "device_mismatch"
    elif residual.shape not in ((1, 5120), (1, 1, 5120)) or delta.shape != residual.shape:
        reason = "requires_m1_k5120"
    elif residual.dtype != torch.bfloat16 or delta.dtype != torch.bfloat16:
        reason = "requires_bf16_operands"
    elif (not residual.is_contiguous() or not delta.is_contiguous()
          or residual.requires_grad or delta.requires_grad):
        reason = "unsupported_operand_layout"
    elif torch.cuda.get_device_capability(residual.device) != (9, 0):
        reason = "requires_sm90"
    else:
        reason = projection_reason(projection, residual.device)
        if reason is None and any(_overlap(residual, other) for other in
                                  (delta, projection.weight, projection.weight_scale_inv)):
            reason = "residual_alias"
    return {"requested_backend": backend, "selected_backend": "native" if reason else "triton",
            "fallback_reason": reason, "execution_observed": False,
            "scope": "m1_post_attention_gate_up", "quantizer": "fi_block128_fp32_scales"}


def validate_producer(residual, delta, gamma, eps):
    # This runs before the in-place residual write, including standalone calls.
    if (residual.device.type != "cuda" or getattr(torch.version, "hip", None)
            or torch.cuda.get_device_capability(residual.device) != (9, 0)
            or residual.shape not in ((1, 5120), (1, 1, 5120))
            or delta.shape != residual.shape or gamma.shape != (5120,)
            or not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps <= 0):
        raise ValueError("Post-attention FP8 producer requires SM90 M1/K5120 and positive epsilon")
    operands = (residual, delta, gamma)
    if any(t.device != residual.device or t.dtype != torch.bfloat16
           or not t.is_contiguous() or t.requires_grad for t in operands):
        raise ValueError("Post-attention FP8 producer requires contiguous colocated inference BF16")
    if _overlap(residual, delta) or _overlap(residual, gamma):
        raise ValueError("Post-attention FP8 residual must not alias a read operand")


def validate_prequantized(projection, values, scales, input_shape):
    reason = projection_reason(projection, values.device)
    if reason:
        raise ValueError("Post-attention FP8 projection: " + reason)
    if (values.device.type != "cuda" or getattr(torch.version, "hip", None)
            or torch.cuda.get_device_capability(values.device) != (9, 0)
            or tuple(input_shape) not in ((1, 5120), (1, 1, 5120))
            or values.shape != (1, 5120) or values.dtype != torch.float8_e4m3fn
            or not values.is_contiguous() or values.requires_grad):
        raise ValueError("Prequantized gate/up requires contiguous SM90 FP8 [1,5120]")
    if (scales.device != values.device or scales.dtype != torch.float32
            or scales.shape != (1, 40) or scales.stride() != (1, 4)
            or scales.requires_grad
            or scales.untyped_storage().nbytes() - scales.storage_offset() * 4 < 40 * 4 * 4):
        raise ValueError("Prequantized scales require live [1,40] stride (1,4) and full [40,4] backing")


def postnorm_gate_up(residual, delta, norm, projection, backend):
    """Return residual, gate, up, selected backend; only residual is mutated.

    Unsupported metadata uses the original BF16 producer and projection. Once
    selected, compilation/launch failures propagate without retry or fallback.
    """
    route = execution(residual, delta, projection, backend)
    if route["selected_backend"] == "native":
        if residual.is_cuda and residual.dtype == torch.bfloat16 and residual.shape[-1] == norm.weight.shape[0]:
            from ._triton_rms import fused_add_rms

            residual, normed = fused_add_rms(residual, delta, norm._fused_weight(residual.dtype), norm.eps)
        else:
            residual = residual + delta
            normed = norm(residual)
        gate, up = projection(normed)
        return residual, gate, up, "native"

    with torch.cuda.device(residual.device):
        # Import the public FI entry and reject cold capture before mutation.
        projection._prepare_postnorm_fp8(residual.device)
        gamma = norm._fused_weight(residual.dtype)
        validate_producer(residual, delta, gamma, norm.eps)
        from ._triton_rms import fused_add_rms_quant

        values, scales = fused_add_rms_quant(residual, delta, gamma, norm.eps)
        gate, up = projection.forward_prequantized(values, scales, residual.shape)
        # Only the complete eager producer/consumer path establishes readiness.
        projection._postnorm_fp8_ready_device = residual.device
    return residual, gate, up, "triton"
