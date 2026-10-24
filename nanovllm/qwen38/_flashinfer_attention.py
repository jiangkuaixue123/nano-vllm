"""Explicit standalone FA3 for fresh production-shape BF16 attention only."""

import torch


def flashinfer_fa3_prefill(query, key, value):
    """Read HND operands through NHD views; preserve the caller's HND shape.

    This backend changes attention reduction order. It does not promise native
    or FlashSDPA bit equality. Cached continuation and decode never use it.
    """
    tensors = (query, key, value)
    if any(t.device.type != "cuda" for t in tensors) or torch.version.hip is not None:
        raise ValueError("FlashInfer FA3 prefill requires NVIDIA CUDA")
    if any(t.device != query.device or t.dtype != torch.bfloat16 for t in tensors):
        raise ValueError("FlashInfer FA3 prefill requires same-device BF16 operands")
    if any(t.ndim != 4 for t in tensors):
        raise ValueError("FlashInfer FA3 prefill requires four-dimensional HND operands")
    count = query.shape[2]
    if not 0 < count < 2**31 or query.shape != (1, 24, count, 256):
        raise ValueError("FlashInfer FA3 prefill requires batch1, 24 Q heads and head dimension256")
    if key.shape != (1, 4, count, 256) or value.shape != key.shape:
        raise ValueError("FlashInfer FA3 prefill requires fresh equal-length 4-head K/V")
    # The SM90 kernel builds TMA descriptors with a fixed unit D stride.
    # N/H strides and the data pointer must retain 16-byte alignment. The
    # production merged V view has token stride14336 and a nonzero offset.
    if any(t.stride(-1) != 1 or any(s <= 0 or s * 2 % 16 for s in t.stride()[1:3])
           or t.data_ptr() % 16 for t in tensors):
        raise ValueError("FlashInfer FA3 prefill requires unit last stride and aligned positive H/N strides")
    if torch.cuda.get_device_capability(query.device)[0] != 9:
        raise ValueError("FlashInfer FA3 prefill requires SM90")
    version = torch.version.cuda
    if version is None or tuple(int(part) for part in version.split(".")[:2]) < (12, 3):
        raise ValueError("FlashInfer FA3 prefill requires CUDA12.3 or newer")

    # Import only on the explicitly selected, supported initial-prefill path.
    from flashinfer.prefill import single_prefill_with_kv_cache

    # Always use NHD metadata: the installed single-FA3 HND binding reads
    # kv_len/head-count from dimensions0/1 even when HND is requested.
    q, k, v = (t.squeeze(0).transpose(0, 1) for t in tensors)
    output = single_prefill_with_kv_cache(
        q, k, v, o_dtype=torch.bfloat16, causal=True, kv_layout="NHD",
        pos_encoding_mode="NONE", use_fp16_qk_reduction=False,
        sm_scale=0.0625, window_left=-1, logits_soft_cap=0.0,
        backend="fa3", return_lse=False,
    )
    if (output.shape != q.shape or output.dtype != query.dtype
            or output.device != query.device or not output.is_contiguous()):
        raise RuntimeError("FlashInfer FA3 returned an unsupported output layout or dtype")
    return output.transpose(0, 1).unsqueeze(0)
