"""Optional CUDA BF16 SiLU/multiply with an explicit BF16 intermediate.

This removes the materialized SiLU tensor, not its rounding. Exact agreement
with a particular Torch/CUDA build still requires GPU validation of exp/div,
subnormals and compiler lowering; this module makes no parity or speed claim.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _silu_mul_kernel(G, U, O, N: tl.constexpr, COLS: tl.constexpr,
                     ROWS: tl.constexpr, GB: tl.constexpr, GR: tl.constexpr,
                     GC: tl.constexpr, UB: tl.constexpr, UR: tl.constexpr,
                     UC: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < N
    row, col = offset // COLS, offset % COLS
    batch, row = row // ROWS, row % ROWS
    gate = tl.load(G + batch * GB + row * GR + col * GC, mask, other=0).to(tl.float32)
    up = tl.load(U + batch * UB + row * UR + col * UC, mask, other=0).to(tl.float32)
    # Torch CUDA SiLU returns BF16 before its separate multiply. Do not replace
    # this with sigmoid(gate)*gate*up or erase the explicit rounding boundary.
    silu = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
    rounded = silu.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    result = (rounded * up).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(O + offset, result, mask)


def silu_and_mul(gate, up):
    """Read 2D/3D BF16 views with independent strides; return fresh dense output."""
    if (gate.device.type != "cuda" or gate.device != up.device
            or torch.version.hip is not None):
        raise ValueError("Fused MLP activation requires one NVIDIA CUDA device")
    if gate.dtype != torch.bfloat16 or up.dtype != torch.bfloat16:
        raise ValueError("Fused MLP activation requires BF16 gate and up")
    if gate.shape != up.shape or gate.ndim not in (2, 3) or any(n <= 0 for n in gate.shape):
        raise ValueError("Fused MLP activation requires equal nonempty 2D/3D shapes")
    if gate.requires_grad or up.requires_grad:
        raise ValueError("Fused MLP activation supports inference tensors only")
    if any(s < 0 for tensor in (gate, up) for s in tensor.stride()):
        raise ValueError("Fused MLP activation requires nonnegative strides")
    output = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
    gate_strides = (0, *gate.stride()) if gate.ndim == 2 else gate.stride()
    up_strides = (0, *up.stride()) if up.ndim == 2 else up.stride()
    # Merged FP8 projections return split views with row stride 2*COLS. Address
    # those views directly, including different column strides and masked tails.
    with torch.cuda.device(gate.device):
        _silu_mul_kernel[(triton.cdiv(gate.numel(), 1024),)](
            gate, up, output, gate.numel(), gate.shape[-1], gate.shape[-2],
            *gate_strides, *up_strides, 1024, num_warps=4,
            enable_fp_fusion=False,
        )
    return output
