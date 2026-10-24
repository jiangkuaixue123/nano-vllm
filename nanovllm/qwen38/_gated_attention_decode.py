"""Metadata dispatch for the experimental fixed-shape gated decode kernel.

This module imports no Triton. Unsupported metadata preserves native attention;
once selected, preparation/compilation/launch errors propagate to the caller.
"""

import torch


def execution(query, key, value, gate, positions, backend, *, capacity, fresh_prefill):
    if backend not in ("native", "triton_split"):
        raise ValueError("Unsupported gated decode attention backend")
    reason = None
    tensors = (query, key, value, gate, positions)
    if backend == "native":
        reason = "configured_native"
    elif fresh_prefill:
        reason = "fresh_prefill"
    elif capacity != 10240 or key is None or value is None:
        reason = "requires_fixed_capacity_10240"
    elif query.device.type != "cuda" or getattr(torch.version, "hip", None):
        reason = "requires_nvidia_cuda"
    elif any(t.device != query.device for t in tensors):
        reason = "device_mismatch"
    elif (query.shape != (1, 24, 1, 256) or key.shape != (1, 4, 10240, 256)
          or value.shape != key.shape or gate.shape != (1, 1, 6144)
          or positions.shape != (1,)):
        reason = "unsupported_shape"
    elif any(t.dtype != torch.bfloat16 for t in tensors[:-1]) or positions.dtype != torch.int64:
        reason = "unsupported_dtype"
    elif any(not t.is_contiguous() or t.requires_grad for t in tensors):
        reason = "unsupported_layout_or_grad"
    elif torch.cuda.get_device_capability(query.device) != (9, 0):
        reason = "requires_sm90"
    return {
        "requested_backend": backend,
        "selected_backend": "native" if reason else "triton_split",
        "fallback_reason": reason,
        "execution_observed": False,
        "scope": "single_token_gqa24_kv4_d256_capacity10240",
    }


def prepare(module, query):
    """Reject cold capture before the consumer writes its request-owned KV.

Scratch/output tensors are invocation-owned eager tensors or allocations in the
existing model-owned CUDA Graph pool. No process-global scratch is introduced.
Readiness records a completed eager launch, not compilation or GPU validation.
"""
    with torch.cuda.device(query.device):
        if (torch.cuda.is_current_stream_capturing()
                and getattr(module, "_gated_decode_ready_device", None) != query.device):
            raise RuntimeError("Warm gated decode attention eagerly before CUDA Graph capture")
        from ._triton_gated_attention_decode import gated_attention_decode

    return gated_attention_decode
