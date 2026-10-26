"""M1 input RMS + original FI block128 quantization, optionally retaining BF16."""

import torch
import triton
import triton.language as tl


@triton.jit
def _input_rms_quant_kernel(
    x_ptr,
    w_ptr,
    bf16_ptr,
    q_ptr,
    scale_ptr,
    eps,
    BLOCK: tl.constexpr,
    KEEP_BF16: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    mask = cols < 5120
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, 0) / 5120
    normalized = x * tl.rsqrt(variance + eps)
    weight = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Preserve the original BF16 norm result before the group amax and quantization.
    rounded = (normalized * weight).to(tl.bfloat16)
    if KEEP_BF16:
        tl.store(bf16_ptr + cols, rounded, mask=mask)
    blocks = tl.reshape(rounded.to(tl.float32), (BLOCK // 128, 128))
    magnitudes = tl.where(blocks == blocks, tl.abs(blocks), 0.0)  # noqa: PLR0124 - FI ignores NaNs in amax
    amax = tl.max(magnitudes, axis=1)
    scale = tl.where(amax != 0.0, tl.div_rn(448.0, amax), 1.0)
    inverse_scale = tl.div_rn(1.0, scale)
    quantized = (blocks * scale[:, None]).to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(q_ptr + cols, tl.reshape(quantized, (BLOCK,)), mask=mask)
    groups = tl.arange(0, BLOCK // 128)
    tl.store(scale_ptr + groups * 4, inverse_scale, mask=groups < 40)


def rms_quant(x, gamma, eps, *, keep_bf16):
    """Fresh outputs; the live scale view retains full padded backing through capture."""
    from ._input_rms_fp8 import validate_producer

    validate_producer(x, gamma, eps, keep_bf16)
    with torch.cuda.device(x.device):
        normalized = torch.empty_like(x) if keep_bf16 else None
        values = torch.empty((1, 5120), device=x.device, dtype=torch.float8_e4m3fn)
        backing = torch.empty((40, 4), device=x.device, dtype=torch.float32)
        scales = backing.as_strided((1, 40), (1, 4))
        # The unused BF16 pointer is never accessed in the no-materialization variant.
        _input_rms_quant_kernel[(1,)](
            x,
            gamma,
            normalized if keep_bf16 else x,
            values,
            backing,
            eps,
            BLOCK=8192,
            KEEP_BF16=keep_bf16,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return normalized, values, scales
