"""融合 add + RMSNorm：一次读写完成残差相加与归一化，只减物化次数。

数值语义与「x = x + mixed（舍入为 bf16）→ rms_norm(x)」一致：var 与归一化都基于
舍入后的 bf16 值，fp32 累加，再乘调用方传入的权重。剩余差异只应来自 fp32 归约顺序。
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _add_rms_kernel(res_ptr, delta_ptr, w_ptr, out_ptr, n_cols, eps,
                    BLOCK: tl.constexpr, STORE_RESIDUAL: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    base = row * n_cols
    r = tl.load(res_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(delta_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    h = r + d
    # 原路径是「先把残差舍入成 bf16，再对这份 bf16 做 rms_norm」；
    # 融合版必须吃同一份舍入后的值，否则每层的归一化输入都比原路径多一份 fp32 精度。
    h_rounded = h.to(tl.bfloat16).to(tl.float32)
    if STORE_RESIDUAL:
        tl.store(res_ptr + base + cols, h.to(tl.bfloat16), mask=mask)
    var = tl.sum(h_rounded * h_rounded, 0) / n_cols
    n = h_rounded * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + base + cols, (n * w).to(tl.bfloat16), mask=mask)


def fused_add_rms(residual, delta, weight, eps, block=8192):
    """residual ← residual + delta（原地），返回归一化结果（新张量）。

    只支持二维 [rows, hidden] 与三维 [1, rows, hidden]（后者按行展平处理）。
    """
    if residual.dtype != torch.bfloat16 or delta.dtype != torch.bfloat16:
        raise ValueError("fused_add_rms requires bfloat16 operands")
    if residual.shape != delta.shape or residual.shape[-1] > block:
        raise ValueError("fused_add_rms shape contract violated")
    rows = residual.numel() // residual.shape[-1]
    n_cols = residual.shape[-1]
    out = torch.empty_like(residual)
    _add_rms_kernel[(rows,)](residual, delta, weight, out, n_cols, eps,
                             BLOCK=block, STORE_RESIDUAL=True)
    return residual, out


@triton.jit
def _add_rms_quant_kernel(res_ptr, delta_ptr, w_ptr, q_ptr, scale_ptr, n_cols, eps,
                          BLOCK: tl.constexpr):
    # Keep the original full-row RMS sequence before forming block maxima.
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    r = tl.load(res_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    d = tl.load(delta_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    h = r + d
    h_rounded = h.to(tl.bfloat16).to(tl.float32)
    tl.store(res_ptr + cols, h.to(tl.bfloat16), mask=mask)
    var = tl.sum(h_rounded * h_rounded, 0) / n_cols
    n = h_rounded * tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # This BF16 boundary is semantic even though no normed tensor is stored.
    normalized = (n * w).to(tl.bfloat16).to(tl.float32)
    blocks = tl.reshape(normalized, (BLOCK // 128, 128))
    # CUDA __hmax starts from +0 and ignores a NaN operand. The group maximum
    # stays BF16-representable; NaNs in values still reach the FP8 conversion.
    magnitudes = tl.where(blocks == blocks, tl.abs(blocks), 0.0)
    amax = tl.max(magnitudes, axis=1)
    scale = tl.where(amax != 0.0, tl.div_rn(448.0, amax), 1.0)
    inverse_scale = tl.div_rn(1.0, scale)
    quantized = (blocks * scale[:, None]).to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(q_ptr + cols, tl.reshape(quantized, (BLOCK,)), mask=mask)
    groups = tl.arange(0, BLOCK // 128)
    tl.store(scale_ptr + groups * 4, inverse_scale, mask=groups < n_cols // 128)


def fused_add_rms_quant(residual, delta, weight, eps):
    """M1/K5120 producer; fresh FP8 values and live view of full padded scales.

    The returned scale view owns its [40,4] backing through PyTorch. Only column
    zero is written; padding bytes are not outputs. CUDA Graph owns allocations
    from its capture pool, without a process-global or module scratch buffer.
    """
    from ._postnorm_fp8 import validate_producer

    validate_producer(residual, delta, weight, eps)
    with torch.cuda.device(residual.device):
        values = torch.empty((1, 5120), dtype=torch.float8_e4m3fn, device=residual.device)
        scale_storage = torch.empty((40, 4), dtype=torch.float32, device=residual.device)
        scales = scale_storage.as_strided((1, 40), (1, 4))
        _add_rms_quant_kernel[(1,)](residual, delta, weight, values, scale_storage, 5120, eps,
                                    BLOCK=8192, num_warps=4)
    return values, scales
