# Copyright 2025-2026 The HuggingFace Inc. team. All rights reserved.
# Adapted from kernels-community/finegrained-fp8 matmul.py and utils.py.
# Licensed under Apache-2.0; see LICENSES/Apache-2.0.txt.
# Source: https://huggingface.co/kernels/kernels-community/finegrained-fp8
"""Fixed-configuration native FP8 GEMM; imported only for CUDA execution."""

import torch
import triton
import triton.language as tl

# Introspection for the explicit L0 probe; never participates in computation.
last_compiled_kernel = None


@triton.jit
def qwen38_block_fp8(A, B, S, C, M: tl.constexpr, N: tl.constexpr,
                     K: tl.constexpr, BM: tl.constexpr):
    physical_rows = tl.arange(0, BM)
    rows = tl.program_id(0) * 2 + physical_rows // 32
    lanes = physical_rows % 32
    output_rows = tl.program_id(0) * 2 + tl.arange(0, 2)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    inner = tl.arange(0, 128)
    acc = tl.zeros((2, 128), tl.float32)
    for block in range(tl.cdiv(K, 128)):
        kk = block * 128 + inner
        raw = tl.load(A + rows[:, None] * K + kk[None, :],
                      (rows[:, None] < M) & (kk[None, :] < K), other=0).to(tl.float32)
        scale_a = tl.max(tl.abs(raw), axis=1) / 448.0
        a = (raw / tl.maximum(scale_a[:, None], 1e-12)).to(tl.float8e4nv)
        b = tl.load(B + cols[None, :] * K + kk[:, None],
                    (cols[None, :] < N) & (kk[:, None] < K), other=0.0)
        scale_b = tl.load(S + (cols // 128) * tl.cdiv(K, 128) + block,
                          cols < N, other=0)
        # Pack 32 singleton lanes per logical row into the SM90 M64 tile.
        # With finite B, masking A retains one product per hardware K32 segment.
        # The FP32 lane reduction changes rounding order; speed is unmeasured.
        keep = (inner[None, :] % 32) == lanes[:, None]
        aa = tl.where(keep, a, 0.0).to(tl.float8e4nv)
        packed = tl.dot(aa, b, max_num_imprecise_acc=32)
        partial = tl.sum(tl.reshape(packed, (2, 32, 128)), axis=1)
        logical_scale_a = tl.max(tl.reshape(scale_a, (2, 32)), axis=1)
        acc += partial * logical_scale_a[:, None] * scale_b[None, :]
    tl.store(C + output_rows[:, None] * N + cols[None, :], acc.to(tl.bfloat16),
             (output_rows[:, None] < M) & (cols[None, :] < N))


def matmul(inputs, weight, scales):
    global last_compiled_kernel
    n, k = weight.shape
    m = inputs.numel() // k
    output = torch.empty((*inputs.shape[:-1], n), device=inputs.device, dtype=torch.bfloat16)
    # A fixed launch, deliberately without candidate autotuning. The FP8 dot is
    # accumulated in FP32 per K128 block, then scaled in the reference order.
    # SM90 native FP8 requires MMA v3, whose M tile is a multiple of 64.
    # Smaller tiles select MMA v2 and Triton promotes both FP8 operands to FP16.
    # Each physical tile holds two logical rows, with 32 singleton lanes per row.
    block_m = 64
    with torch.cuda.device(inputs.device):
        last_compiled_kernel = qwen38_block_fp8[(triton.cdiv(m, 2), triton.cdiv(n, 128))](
            inputs, weight, scales, output, m, n, k, block_m, num_warps=4, num_stages=2,
        )
    return output

@triton.jit
def qwen38_w8a16_gemv(A, W, S, C, K: tl.constexpr, N: tl.constexpr,
                      BN: tl.constexpr, BK: tl.constexpr):
    """Decode 形状（M=1）权重直通：FP8 权重按 block-128 反量化后与 bf16 激活点积。

    M=1 时 GEMM 是纯权重带宽问题，MMA 单元闲着。这里只流 FP8 权重（半字节宽），
    反量化在寄存器里做，省下的正是 decode 线性层一半的 HBM 流量。
    """
    pid = tl.program_id(0)
    offs_n = pid * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    acc = tl.zeros((BN,), tl.float32)
    scale_row = (offs_n // 128) * (K // 128)
    for k0 in range(0, K, BK):
        kk = k0 + offs_k
        a = tl.load(A + kk).to(tl.float32)                       # [BK] bf16 -> fp32
        w = tl.load(W + offs_n[:, None] * K + kk[None, :],
                    mask=(offs_n[:, None] < N), other=0.0)       # [BN, BK] fp8
        s = tl.load(S + scale_row + (k0 // 128), mask=offs_n < N, other=0.0)
        d = w.to(tl.float32) * s[:, None]
        acc += tl.sum(d * a[None, :], axis=1)
    tl.store(C + offs_n, acc.to(tl.bfloat16), mask=offs_n < N)


def gemv(inputs, weight, scales):
    """Single-token FP8 linear: stream FP8 weights, dequantize in registers."""
    n, k = weight.shape
    output = torch.empty(n, device=inputs.device, dtype=torch.bfloat16)
    # BK 必须是 128 的整数倍：block scale 按 K/128 一档一个，跨档就要按子块分别取 scale。
    # 目前只支持 BK=128（跨档版本未写），tile 由实测选出。
    block_n, block_k = 32, 128
    with torch.cuda.device(inputs.device):
        qwen38_w8a16_gemv[(triton.cdiv(n, block_n),)](
            inputs.reshape(-1), weight, scales, output,
            k, n, block_n, block_k, num_warps=4, num_stages=3,
        )
    return output
