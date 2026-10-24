"""Default-off M1 SwiGLU/FP8 bridge; unsupported metadata keeps the old path."""

import torch

from .quant import FP8Linear


def projection_reason(projection, device):
    if type(projection) is not FP8Linear:
        return "requires_plain_fp8_down"
    if projection.in_features != 17408 or projection.out_features != 5120:
        return "unsupported_projection_shape"
    if not projection.use_flashinfer:
        return "requires_flashinfer_small_m"
    weight, scales = projection.weight, projection.weight_scale_inv
    if weight.dtype != torch.float8_e4m3fn or scales.dtype != torch.float32:
        return "invalid_checkpoint_dtypes"
    if weight.device != device or scales.device != device:
        return "checkpoint_device_mismatch"
    if (weight.shape != (5120, 17408) or scales.shape != (40, 136)
            or not weight.is_contiguous() or not scales.is_contiguous()
            or weight.requires_grad or scales.requires_grad):
        return "invalid_checkpoint_layout"
    return None


def operand_reason(gate, up):
    if gate.device.type != "cuda" or getattr(torch.version, "hip", None):
        return "requires_nvidia_cuda"
    if gate.device != up.device:
        return "device_mismatch"
    if gate.shape not in ((1, 17408), (1, 1, 17408)) or up.shape != gate.shape:
        return "requires_m1_k17408"
    if gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16:
        return "requires_bf16_operands"
    if (gate.requires_grad or up.requires_grad
            or any(s < 0 for tensor in (gate, up) for s in tensor.stride())):
        return "unsupported_operand_layout"
    if torch.cuda.get_device_capability(gate.device) != (9, 0):
        return "requires_sm90"
    return None


def execution(gate, up, projection, backend, activation_backend="triton"):
    """Read metadata only; this observer does not prepare or execute kernels."""
    if backend not in ("native", "triton"):
        raise ValueError("Unsupported SwiGLU FP8 backend")
    if backend == "native":
        reason = "configured_native"
    elif activation_backend != "triton":
        reason = "requires_triton_mlp_activation"
    else:
        reason = operand_reason(gate, up) or projection_reason(projection, gate.device)
    return {"requested_backend": backend, "selected_backend": "native" if reason else "triton",
            "fallback_reason": reason, "execution_observed": False,
            "scope": "m1_silu_fp8_down", "quantizer": "fi_block128_fp32_scales"}


def validate_producer(gate, up):
    reason = operand_reason(gate, up)
    if reason:
        raise ValueError("SwiGLU FP8 producer: " + reason)


def validate_prequantized(projection, values, scales, input_shape):
    reason = projection_reason(projection, values.device)
    if reason:
        raise ValueError("SwiGLU FP8 projection: " + reason)
    if (values.device.type != "cuda" or getattr(torch.version, "hip", None)
            or torch.cuda.get_device_capability(values.device) != (9, 0)
            or tuple(input_shape) not in ((1, 17408), (1, 1, 17408))
            or values.shape != (1, 17408) or values.dtype != torch.float8_e4m3fn
            or not values.is_contiguous() or values.requires_grad):
        raise ValueError("Prequantized down input requires contiguous SM90 FP8 [1,17408]")
    if (scales.device != values.device or scales.dtype != torch.float32
            or scales.shape != (1, 136) or scales.stride() != (1, 4)
            or scales.requires_grad
            or scales.untyped_storage().nbytes() - scales.storage_offset() * 4 < 136 * 4 * 4):
        raise ValueError("Prequantized scales require live [1,136] stride (1,4) and full [136,4] backing")


def silu_fp8_down(gate, up, projection):
    """Execute the selected producer/consumer; exceptions never select fallback."""
    validate_producer(gate, up)
    with torch.cuda.device(gate.device):
        projection._prepare_silu_fp8(gate.device)
        from ._triton_silu_fp8 import silu_mul_quant

        values, scales = silu_mul_quant(gate, up)
        result = projection._forward_silu_prequantized(values, scales, gate.shape)
        # A complete eager producer/consumer call, not import alone, warms capture.
        projection._silu_fp8_ready_device = gate.device
    return result
