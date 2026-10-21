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
