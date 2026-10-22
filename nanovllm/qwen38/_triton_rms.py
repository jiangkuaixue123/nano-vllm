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
