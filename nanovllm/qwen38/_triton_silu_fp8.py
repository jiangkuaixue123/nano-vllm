"""Fuse the existing BF16 SwiGLU stages with the existing FI block quantizer.

The arithmetic is adapted from this engine's _triton_silu and _triton_rms.
See docs/qwen38-silu-fp8.md for upstream fusion-design references and limits.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _silu_mul_quant_kernel(G, U, Q, S, GC: tl.constexpr, UC: tl.constexpr,
                           BLOCK: tl.constexpr):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(G + offset * GC).to(tl.float32)
    up = tl.load(U + offset * UC).to(tl.float32)
    # Keep both BF16 rounding boundaries of the active _triton_silu kernel.
    silu = tl.div_rn(gate, 1.0 + libdevice.exp(-gate))
    rounded = silu.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    result = (rounded * up).to(tl.bfloat16, fp_downcast_rounding="rtne")
    blocks = tl.reshape(result.to(tl.float32), (BLOCK // 128, 128))
    # Literal H2/FlashInfer quantization semantics, including NaN/zero handling.
    magnitudes = tl.where(blocks == blocks, tl.abs(blocks), 0.0)  # noqa: PLR0124
    amax = tl.max(magnitudes, axis=1)
    scale = tl.where(amax != 0.0, tl.div_rn(448.0, amax), 1.0)
    inverse_scale = tl.div_rn(1.0, scale)
    quantized = (blocks * scale[:, None]).to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(Q + offset, tl.reshape(quantized, (BLOCK,)))
    groups = tl.program_id(0) * (BLOCK // 128) + tl.arange(0, BLOCK // 128)
    tl.store(S + groups * 4, inverse_scale)


def silu_mul_quant(gate, up):
    """Read singleton strided BF16 views; return fresh FP8 and live padded scales.

    Only column zero of the scale backing is output. Its padding is neither
    initialized nor inspected. CUDA Graph owns captured allocations; there is
    no module/global scratch tensor or request-state mutation.
    """
    from ._silu_fp8 import validate_producer

    validate_producer(gate, up)
    with torch.cuda.device(gate.device):
        values = torch.empty((1, 17408), dtype=torch.float8_e4m3fn, device=gate.device)
        scale_storage = torch.empty((136, 4), dtype=torch.float32, device=gate.device)
        scales = scale_storage.as_strided((1, 136), (1, 4))
        _silu_mul_quant_kernel[(17,)](
            gate, up, values, scale_storage, gate.stride()[-1], up.stride()[-1],
            BLOCK=1024, num_warps=4, enable_fp_fusion=False,
        )
    return values, scales
