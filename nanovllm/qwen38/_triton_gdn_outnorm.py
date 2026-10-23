"""Opt-in BF16 GDN output RMSNorm/SiLU: target CUDA validation is required."""

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _outnorm_kernel(X, G, W, O, ROWS: tl.constexpr, HEADS: tl.constexpr,
                    GT: tl.constexpr, GH: tl.constexpr, GC: tl.constexpr,
                    EPS: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4)
    lanes = tl.arange(0, 32)
    cols = lanes[None, :] * 4
    offset = rows[:, None] * 128 + cols
    mask = rows[:, None] < ROWS
    x0 = tl.load(X + offset, mask, other=0).to(tl.float32)
    x1 = tl.load(X + offset + 1, mask, other=0).to(tl.float32)
    x2 = tl.load(X + offset + 2, mask, other=0).to(tl.float32)
    x3 = tl.load(X + offset + 3, mask, other=0).to(tl.float32)
    # Torch's aligned vectorized width-128 RMSNorm visits four adjacent values
    # per lane, followed by shuffle-down offsets 16,8,4,2,1. Do not use a plain
    # reduction over 128 squares: that changes the FP32 addition tree. Explicit
    # FMA targets CUDA's contracted square/add; compiled parity is still unproven.
    sums = tl.fma(x0, x0, 0.0)
    sums = tl.fma(x1, x1, sums)
    sums = tl.fma(x2, x2, sums)
    sums = tl.fma(x3, x3, sums)
    for delta in tl.static_range(5):
        other = tl.broadcast_to((lanes[None, :] + (16 >> delta)) % 32, (4, 32))
        sums = sums + tl.gather(sums, other, axis=1)
    # Only lane zero follows the CUDA shuffle-down tree; other lanes are unused.
    variance = tl.sum(tl.where(lanes[None, :] == 0, sums, 0.0), axis=1) * (1.0 / 128)
    inv = libdevice.rsqrt(variance + EPS)
    gate_offset = (rows[:, None] // HEADS * GT
                   + rows[:, None] % HEADS * GH + cols * GC)
    for component in tl.static_range(4):
        x = tl.load(X + offset + component, mask, other=0).to(tl.float32)
        weight = tl.load(W + cols + component).to(tl.float32)
        gate = tl.load(G + gate_offset + component * GC, mask, other=0).to(tl.float32)
        norm = (weight * (inv[:, None] * x)).to(tl.bfloat16, fp_downcast_rounding="rtne")
        silu = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
        rounded_gate = silu.to(tl.bfloat16, fp_downcast_rounding="rtne")
        result = (norm.to(tl.float32) * rounded_gate.to(tl.float32)).to(
            tl.bfloat16, fp_downcast_rounding="rtne")
        tl.store(O + offset + component, result, mask)


def gated_rms_norm(x, gate, weight, eps):
    """Read [1,tokens,heads,128] BF16 operands; never mutate input or weight."""
    operands = (x, gate, weight)
    if (x.device.type != "cuda" or any(t.device != x.device for t in operands)
            or torch.version.hip is not None):
        raise ValueError("Fused GDN output norm requires one NVIDIA CUDA device")
    if any(t.dtype != torch.bfloat16 for t in operands):
        raise ValueError("Fused GDN output norm requires BF16 operands")
    if (x.ndim != 4 or x.shape[0] != 1 or x.shape[-1] != 128
            or gate.shape != x.shape or any(n <= 0 for n in x.shape)
            or weight.shape != (128,)):
        raise ValueError("Fused GDN output norm requires [1,tokens,heads,128] and weight [128]")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Fused GDN output norm requires contiguous input and weight")
    # Singleton token/head strides can still suggest channels-last to Torch,
    # which selects its composite RMSNorm with a different reduction tree.
    canonical = (x.shape[1] * x.shape[2] * 128, x.shape[2] * 128, 128, 1)
    if x.stride() != canonical:
        raise ValueError("Fused GDN output norm requires canonical contiguous input strides")
    if x.data_ptr() % 8 or weight.data_ptr() % 8:
        raise ValueError("Fused GDN output norm requires 8-byte aligned input and weight")
    if any(t.requires_grad for t in operands):
        raise ValueError("Fused GDN output norm supports inference tensors only")
    if any(s < 0 for s in gate.stride()):
        raise ValueError("Fused GDN output norm requires nonnegative gate strides")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("Fused GDN output norm requires finite positive epsilon")
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rows = x.shape[1] * x.shape[2]
    with torch.cuda.device(x.device):
        _outnorm_kernel[(triton.cdiv(rows, 4),)](
            x, gate, weight, output, rows, x.shape[2], *gate.stride()[1:], eps,
            num_warps=4, enable_fp_fusion=False,
        )
    return output
