"""Opt-in whole Q/K L2 normalization; reduction order differs from Torch."""

import torch


def _read_layout(tensor):
    extent = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape)):
        if size <= 1:
            continue
        if stride < extent:
            return False
        extent += (size - 1) * stride
    return extent <= 2**31 - 1


def execution(query, key, backend):
    """Metadata eligibility only, not an execution or numerical receipt."""
    if backend not in ("native", "triton"):
        raise ValueError("Unsupported GDN Q/K normalization backend")
    reason = None
    if backend == "native":
        reason = "configured_native"
    elif query.device.type != "cuda" or getattr(torch.version, "hip", None):
        reason = "requires_nvidia_cuda"
    elif query.device != key.device:
        reason = "device_mismatch"
    elif torch.cuda.get_device_capability(query.device) != (9, 0):
        reason = "requires_sm90"
    elif query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
        reason = "requires_bf16_inputs"
    elif (query.ndim != 4 or query.shape[0] != 1 or query.shape[2:] != (16, 128)
          or not 1 <= query.shape[1] <= 8192 or key.shape != query.shape):
        reason = "unsupported_grouped_prefill_shape"
    elif query.requires_grad or key.requires_grad:
        reason = "requires_inference_operands"
    elif not _read_layout(query) or not _read_layout(key):
        reason = "unsupported_read_layout"
    return {"requested_backend": backend, "selected_backend": "native" if reason else "triton",
            "fallback_reason": reason, "scope": "whole_qk_l2", "executed": False,
            "cross_backend_comparison": "diagnostic_reduction_order_differs"}


def normalize_pair(query, key, backend):
    if execution(query, key, backend)["selected_backend"] == "native":
        def normalized(x):
            work = x.float()
            return (work * torch.rsqrt((work * work).sum(-1, keepdim=True) + 1e-6)).to(x.dtype)

        return normalized(query).contiguous(), normalized(key).contiguous()
    from ._triton_gdn_qk_norm import normalize_pair as fused

    return fused(query, key)
