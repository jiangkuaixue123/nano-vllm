"""Default-off M1 GDN output norm/FP8 bridge; fallback is metadata-only."""

import math

import torch

from .quant import FP8Linear


def projection_reason(projection, device):
    if type(projection) is not FP8Linear:
        return "requires_plain_fp8_output"
    if projection.in_features != 6144 or projection.out_features != 5120:
        return "unsupported_projection_shape"
    if not projection.use_flashinfer:
        return "requires_flashinfer_small_m"
    weight, scales = projection.weight, projection.weight_scale_inv
    if weight.dtype != torch.float8_e4m3fn or scales.dtype != torch.float32:
        return "invalid_checkpoint_dtypes"
    if weight.device != device or scales.device != device:
        return "checkpoint_device_mismatch"
    if (weight.shape != (5120, 6144) or scales.shape != (40, 48)
            or not weight.is_contiguous() or not scales.is_contiguous()
            or weight.requires_grad or scales.requires_grad):
        return "invalid_checkpoint_layout"
    return None


def operand_reason(x, gate):
    if x.device.type != "cuda" or getattr(torch.version, "hip", None):
        return "requires_nvidia_cuda"
    if x.device != gate.device:
        return "device_mismatch"
    if x.shape != (1, 1, 48, 128) or gate.shape != x.shape:
        return "requires_m1_48x128"
    if x.dtype != torch.bfloat16 or gate.dtype != torch.bfloat16:
        return "requires_bf16_operands"
    if (not x.is_contiguous() or x.stride() != (6144, 6144, 128, 1)
            or x.data_ptr() % 8 or x.requires_grad or gate.requires_grad
            or any(s < 0 for s in gate.stride())):
        return "unsupported_operand_layout"
    if torch.cuda.get_device_capability(x.device) != (9, 0):
        return "requires_sm90"
    return None


def weight_reason(weight, eps, device, *, require_detached=True):
    if (weight.shape != (128,) or weight.dtype != torch.bfloat16
            or weight.device != device or not weight.is_contiguous()
            or weight.data_ptr() % 8 or (require_detached and weight.requires_grad)):
        return "requires_bf16_norm_weight"
    if not isinstance(eps, (float, int)) or not math.isfinite(eps) or eps <= 0:
        return "requires_positive_epsilon"
    return None


def execution(x, gate, norm, projection, backend):
    """Read metadata only; do not create the existing norm cache or run kernels."""
    if backend not in ("native", "triton"):
        raise ValueError("Unsupported GDN output norm FP8 backend")
    if backend == "native":
        reason = "configured_native"
    elif norm.outnorm_backend != "triton":
        reason = "requires_triton_gdn_outnorm"
    else:
        reason = (operand_reason(x, gate) or weight_reason(norm.weight, norm.eps, x.device, require_detached=False)
                  or projection_reason(projection, x.device))
    return {"requested_backend": backend, "selected_backend": "native" if reason else "triton",
            "fallback_reason": reason, "execution_observed": False,
            "scope": "m1_gdn_outnorm_fp8_output", "quantizer": "fi_block128_fp32_scales"}


def validate_producer(x, gate, weight, eps):
    reason = operand_reason(x, gate) or weight_reason(weight, eps, x.device)
    if reason:
        raise ValueError("GDN output norm FP8 producer: " + reason)


def validate_prequantized(projection, values, scales, input_shape):
    reason = projection_reason(projection, values.device)
    if reason:
        raise ValueError("GDN output FP8 projection: " + reason)
    if (values.device.type != "cuda" or getattr(torch.version, "hip", None)
            or torch.cuda.get_device_capability(values.device) != (9, 0)
            or tuple(input_shape) != (1, 1, 6144)
            or values.shape != (1, 6144) or values.dtype != torch.float8_e4m3fn
            or not values.is_contiguous() or values.requires_grad):
        raise ValueError("Prequantized GDN input requires contiguous SM90 FP8 [1,6144]")
    if (scales.device != values.device or scales.dtype != torch.float32
            or scales.shape != (1, 48) or scales.stride() != (1, 4)
            or scales.requires_grad
            or scales.untyped_storage().nbytes() - scales.storage_offset() * 4 < 48 * 4 * 4):
        raise ValueError("Prequantized scales require live [1,48] stride (1,4) and full [48,4] backing")


def outnorm_fp8_output(x, gate, norm, projection):
    """Execute the selected bridge; complete eager execution prepares capture."""
    with torch.cuda.device(x.device):
        projection._prepare_gdn_outnorm_fp8(x.device)
        weight = norm._fused_weight(x.dtype)
        validate_producer(x, gate, weight, norm.eps)
        from ._triton_gdn_outnorm_fp8 import gated_rms_quant

        values, scales = gated_rms_quant(x, gate, weight, norm.eps)
        result = projection._forward_gdn_outnorm_prequantized(values, scales, (1, 1, 6144))
        projection._gdn_outnorm_fp8_ready_device = x.device
    return result
