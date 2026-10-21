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
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * 128 + tl.arange(0, 128)
    inner = tl.arange(0, 128)
    acc = tl.zeros((BM, 128), tl.float32)
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
        acc += tl.dot(a, b) * scale_a[:, None] * scale_b[None, :]
    tl.store(C + rows[:, None] * N + cols[None, :], acc.to(tl.bfloat16),
             (rows[:, None] < M) & (cols[None, :] < N))


def matmul(inputs, weight, scales):
    global last_compiled_kernel
    n, k = weight.shape
    m = inputs.numel() // k
    output = torch.empty((*inputs.shape[:-1], n), device=inputs.device, dtype=torch.bfloat16)
    # A fixed launch, deliberately without candidate autotuning. The FP8 dot is
    # accumulated in FP32 per K128 block, then scaled in the reference order.
    with torch.cuda.device(inputs.device):
        last_compiled_kernel = qwen38_block_fp8[(triton.cdiv(m, 16), triton.cdiv(n, 128))](
            inputs, weight, scales, output, m, n, k, 16, num_warps=4, num_stages=2,
        )
    return output
