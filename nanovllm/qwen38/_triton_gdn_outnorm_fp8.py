"""Append FI group-128 quantization to the current BF16 GDN output norm/gate.

Reduction and rounding come from this engine's _triton_gdn_outnorm; quantization
comes from its H2 bridge. Upstream fusion precedents are documented separately.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _outnorm_quant_kernel(X, G, W, Q, S, ROWS: tl.constexpr, HEADS: tl.constexpr,
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
    # Keep the active kernel's four-value FMA chain and lane-zero shuffle tree.
    sums = tl.fma(x0, x0, 0.0)
    sums = tl.fma(x1, x1, sums)
    sums = tl.fma(x2, x2, sums)
    sums = tl.fma(x3, x3, sums)
    for delta in tl.static_range(5):
        other = tl.broadcast_to((lanes[None, :] + (16 >> delta)) % 32, (4, 32))
        sums = sums + tl.gather(sums, other, axis=1)
    variance = tl.sum(tl.where(lanes[None, :] == 0, sums, 0.0), axis=1) * (1.0 / 128)
    inv = libdevice.rsqrt(variance + EPS)
    # Each width-128 head is exactly one FP8 group, in original flattened order.
    cols = tl.arange(0, 128)[None, :]
    offset = rows[:, None] * 128 + cols
    gate_offset = (rows[:, None] // HEADS * GT
                   + rows[:, None] % HEADS * GH + cols * GC)
    x = tl.load(X + offset, mask, other=0).to(tl.float32)
    weight = tl.load(W + cols).to(tl.float32)
    gate = tl.load(G + gate_offset, mask, other=0).to(tl.float32)
    norm = (weight * (inv[:, None] * x)).to(tl.bfloat16, fp_downcast_rounding="rtne")
    silu = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
    rounded_gate = silu.to(tl.bfloat16, fp_downcast_rounding="rtne")
    result = (norm.to(tl.float32) * rounded_gate.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne")
    blocks = result.to(tl.float32)
    # Literal H2/FlashInfer semantics, including NaN/zero handling and division.
    magnitudes = tl.where(blocks == blocks, tl.abs(blocks), 0.0)  # noqa: PLR0124
    amax = tl.max(magnitudes, axis=1)
    scale = tl.where(amax != 0.0, tl.div_rn(448.0, amax), 1.0)
    inverse_scale = tl.div_rn(1.0, scale)
    quantized = (blocks * scale[:, None]).to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(Q + offset, quantized, mask)
    tl.store(S + rows * 4, inverse_scale, rows < ROWS)


def gated_rms_quant(x, gate, weight, eps):
    """Return fresh FP8 [1,6144] and live FP32 [1,48] with full [48,4] backing.

    Scale padding is not output. Captured allocations belong to CUDA Graph;
    no input, parameter, recurrent state or global scratch tensor is modified.
    """
    from ._gdn_outnorm_fp8 import validate_producer

    validate_producer(x, gate, weight, eps)
    with torch.cuda.device(x.device):
        values = torch.empty((1, 6144), dtype=torch.float8_e4m3fn, device=x.device)
        scale_storage = torch.empty((48, 4), dtype=torch.float32, device=x.device)
        scales = scale_storage.as_strided((1, 48), (1, 4))
        _outnorm_quant_kernel[(12,)](
            x, gate, weight, values, scale_storage, 48, 48, *gate.stride()[1:], eps,
            num_warps=4, enable_fp_fusion=False,
        )
    return values, scales
