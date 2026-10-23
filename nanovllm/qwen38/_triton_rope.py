"""Optional paired postnorm partial RoPE with the native BF16 boundaries.

The two BF16 products and the BF16 sum are separate rounding steps. This
kernel removes temporary storage, not those steps. Compiler/numerical parity
and performance require separate GPU qualification; no norm is fused here.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paired_rope_kernel(
    Q, K, C, S, OQ, OK, TOKENS: tl.constexpr, QN: tl.constexpr, N: tl.constexpr,
    QH: tl.constexpr, QT: tl.constexpr, QD: tl.constexpr,
    KH: tl.constexpr, KT: tl.constexpr, KD: tl.constexpr,
    CT: tl.constexpr, CD: tl.constexpr, ST: tl.constexpr, SD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < N
    is_q = offset < QN
    local = tl.where(is_q, offset, offset - QN)
    dim = local % 256
    token = local // 256 % TOKENS
    head = local // (256 * TOKENS)
    source = tl.where(is_q, Q + head * QH + token * QT + dim * QD,
                      K + head * KH + token * KT + dim * KD)
    value = tl.load(source, valid, other=0)
    rotate = valid & (dim < 64)
    partner_dim = tl.where(dim < 32, dim + 32, dim - 32)
    partner_source = tl.where(
        is_q, Q + head * QH + token * QT + partner_dim * QD,
        K + head * KH + token * KT + partner_dim * KD,
    )
    partner = tl.load(partner_source, rotate, other=0).to(tl.float32)
    # Native constructs [-second, first] before multiplying. Preserve signed
    # zero and the sign operation rather than rewriting this as subtraction.
    signed_partner = tl.where(dim < 32, -partner, partner)
    cosine = tl.load(C + token * CT + dim * CD, rotate, other=0).to(tl.float32)
    sine = tl.load(S + token * ST + dim * SD, rotate, other=0).to(tl.float32)
    product1 = (value.to(tl.float32) * cosine).to(tl.bfloat16, fp_downcast_rounding="rtne")
    product2 = (signed_partner * sine).to(tl.bfloat16, fp_downcast_rounding="rtne")
    rotated = (product1.to(tl.float32) + product2.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne",
    )
    # Keep the tail in BF16, including its original storage bits. No FP32
    # round-trip is needed for the unrotated 192 values in each head.
    result = tl.where(dim < 64, rotated, value)
    destination = tl.where(is_q, OQ + local, OK + local)
    tl.store(destination, result, valid)


def paired_partial_rope(query, key, cos, sin):
    """Read BF16 strided views; return two fresh contiguous production outputs."""
    tensors = (query, key, cos, sin)
    if (query.device.type != "cuda" or torch.version.hip is not None
            or any(t.device != query.device for t in tensors)):
        raise ValueError("Paired partial RoPE requires one NVIDIA CUDA device")
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise ValueError("Paired partial RoPE requires BF16 operands")
    if any(t.ndim != 4 for t in tensors):
        raise ValueError("Paired partial RoPE requires rank-four operands")
    tokens = query.shape[2]
    if (tokens <= 0 or query.shape != (1, 24, tokens, 256)
            or key.shape != (1, 4, tokens, 256)
            or cos.shape != (1, 1, tokens, 64)
            or sin.shape != cos.shape):
        raise ValueError("Paired partial RoPE requires the production Q/K/table shapes")
    if any(t.requires_grad for t in tensors):
        raise ValueError("Paired partial RoPE supports inference tensors only")
    if any(s < 0 for t in tensors for s in t.stride()):
        raise ValueError("Paired partial RoPE requires nonnegative strides")
    out_q = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    out_k = torch.empty(key.shape, dtype=key.dtype, device=key.device)
    total = query.numel() + key.numel()
    with torch.cuda.device(query.device):
        _paired_rope_kernel[(triton.cdiv(total, 1024),)](
            query, key, cos, sin, out_q, out_k, tokens, query.numel(), total,
            *query.stride()[1:], *key.stride()[1:],
            *cos.stride()[2:], *sin.stride()[2:], 1024,
            num_warps=4, enable_fp_fusion=False,
        )
    return out_q, out_k
