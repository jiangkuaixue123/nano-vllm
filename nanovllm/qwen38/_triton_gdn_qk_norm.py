"""Paired Q/K whole L2 candidate; FP32 reduction tree is intentionally different.

The 16-token/four-warp schedule follows the vLLM preparation reference, but
input strides and native rsqrt semantics are retained explicitly. No gates,
query scaling, V or state work belongs here. GPU qualification is separate.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _l2(values):
    work = values.to(tl.float32)
    squared = work * work
    # FP fusion is disabled at launch: square and sum remain separate FP32 ops.
    total = tl.sum(squared, axis=1)
    epsilon_added = total + 1.0e-6
    inverse = libdevice.rsqrt(epsilon_added)
    product = work * inverse[:, None]
    return product.to(tl.bfloat16, fp_downcast_rounding="rtne")


@triton.jit
def _paired_l2(Query, Key, QueryOut, KeyOut, T: tl.constexpr,
               QT: tl.constexpr, QH: tl.constexpr, QF: tl.constexpr,
               KT: tl.constexpr, KH: tl.constexpr, KF: tl.constexpr,
               BT: tl.constexpr):
    token = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    feature = tl.arange(0, 128).to(tl.int64)
    output_offset = (token[:, None] * 16 + head) * 128 + feature[None, :]
    mask = token[:, None] < T
    query = tl.load(Query + token[:, None] * QT + head * QH + feature[None, :] * QF,
                    mask, other=0.0)
    tl.store(QueryOut + output_offset, _l2(query), mask)
    key = tl.load(Key + token[:, None] * KT + head * KH + feature[None, :] * KF,
                  mask, other=0.0)
    tl.store(KeyOut + output_offset, _l2(key), mask)


def normalize_pair(query, key):
    query_out = torch.empty(query.shape, device=query.device, dtype=torch.bfloat16)
    key_out = torch.empty(key.shape, device=key.device, dtype=torch.bfloat16)
    with torch.cuda.device(query.device):
        _paired_l2[(triton.cdiv(query.shape[1], 16), 16)](
            query, key, query_out, key_out, query.shape[1],
            *query.stride()[1:], *key.stride()[1:], 16,
            num_warps=4, num_stages=2, enable_fp_fusion=False,
        )
    return query_out, key_out
