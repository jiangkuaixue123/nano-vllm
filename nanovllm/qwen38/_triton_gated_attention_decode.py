# Copyright 2023-2024 SGLang Team
# Modifications Copyright 2026 InferMatrixLab contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. See LICENSES/Apache-2.0.txt.
#
# Adapted from SGLang's grouped stage1 and stage2 decode attention:
# https://github.com/sgl-project/sglang/blob/406786fd910dd82d03d1ce2c2e4c08b547786bc2/python/sglang/kernels/ops/attention/decode_attention.py
# Upstream declares adaptation from LightLLM:
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py
"""Unqualified SM90 C1 split attention, direct BF16 KV and a BF16 gate epilogue.

One fixed configuration; no autotuning or scheduler. This computes dense
attention over every live token. Online softmax/PV changes rounding relative to
native globally normalized BF16 probabilities; native-bit-exactness is NOT a
contract. See docs/qwen38-gated-attention-decode.md before any qualification.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gated_decode_partials(Q, K, V, Position, Partial, Lse):
    kv_head = tl.program_id(0)
    split = tl.program_id(1)
    # 32 splits * 320 tokens == the fixed 10240 capacity. A split is never
    # inferred from a captured Python length; short resets mask every stale tail.
    live = tl.load(Position) + 1
    start = split * 320
    end = tl.minimum(start + 320, live)
    h = tl.arange(0, 16)
    d = tl.arange(0, 256)
    heads = kv_head * 6 + h
    valid_h = h < 6
    q = tl.load(Q + heads[:, None] * 256 + d[None, :], valid_h[:, None], other=0)
    maximum = tl.full((16,), -float("inf"), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    acc = tl.zeros((16, 256), tl.float32)
    if start < end:
        for tile in tl.range(start, end, 32):
            n = tile + tl.arange(0, 32)
            valid_n = (n < end) & (n < 10240)
            k = tl.load(K + kv_head * 2621440 + n[None, :] * 256 + d[:, None],
                        valid_n[None, :], other=0)
            scores = tl.dot(q, k)
            # Native QK materializes BF16 before its exact-power-of-two scale.
            scores = scores.to(tl.bfloat16).to(tl.float32)
            scores = (scores * 0.0625).to(tl.bfloat16).to(tl.float32)
            # Padded query rows stay finite to avoid inf-inf in their unused math.
            scores = tl.where(valid_n[None, :], scores, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(scores, 1))
            rescale = tl.exp(maximum - next_maximum)
            p = tl.exp(scores - next_maximum[:, None])
            v = tl.load(V + kv_head * 2621440 + n[:, None] * 256 + d[None, :],
                        valid_n[:, None], other=0)
            acc = acc * rescale[:, None] + tl.dot(p.to(tl.bfloat16), v)
            denominator = denominator * rescale + tl.sum(p, 1)
            maximum = next_maximum
        acc = acc / denominator[:, None]
        maximum = maximum + tl.log(denominator)
    # Empty splits are overwritten too. No uninitialized or previous-request
    # partial can survive a shorter request reset.
    tl.store(Partial + (heads[:, None] * 32 + split) * 256 + d[None, :],
             acc, valid_h[:, None])
    tl.store(Lse + heads * 32 + split, maximum, valid_h)


@triton.jit
def _gated_decode_merge(Partial, Lse, Gate, Position, Output):
    head = tl.program_id(0)
    d = tl.arange(0, 256)
    live = tl.load(Position) + 1
    maximum = -float("inf")
    denominator = 0.0
    acc = tl.zeros((256,), tl.float32)
    # Retain upstream's stable sequential split merge. The split count and
    # order are identical in eager and Graph, including shorter live lengths.
    for split in tl.range(0, 32, num_stages=2):
        if split * 320 < live:
            partial = tl.load(Partial + (head * 32 + split) * 256 + d)
            lse = tl.load(Lse + head * 32 + split)
            next_maximum = tl.maximum(maximum, lse)
            old_scale = tl.exp(maximum - next_maximum)
            new_scale = tl.exp(lse - next_maximum)
            acc = acc * old_scale + partial * new_scale
            denominator = denominator * old_scale + new_scale
            maximum = next_maximum
    # Preserve the consumer's three BF16 boundaries: attention output,
    # sigmoid output, and final product. Do not fold sigmoid into FP32 PV.
    result = (acc / denominator).to(tl.bfloat16).to(tl.float32)
    gate = tl.load(Gate + head * 256 + d).to(tl.float32)
    sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    sigmoid = sigmoid.to(tl.bfloat16).to(tl.float32)
    tl.store(Output + head * 256 + d, (result * sigmoid).to(tl.bfloat16))


def gated_attention_decode(query, key, value, gate, positions):
    """Launch only after metadata dispatch and the existing in-bounds KV write.

Position is the same device int64 buffer used for that write, in [0,10239].
The request runner owns validity; no device scalar is read back to the host.
"""
    from ._gated_attention_decode import execution

    route = execution(query, key, value, gate, positions, "triton_split",
                      capacity=10240, fresh_prefill=False)
    if route["selected_backend"] != "triton_split":
        raise ValueError("Gated decode attention: " + route["fallback_reason"])
    with torch.cuda.device(query.device):
        partial = torch.empty((24, 32, 256), dtype=torch.float32, device=query.device)
        lse = torch.empty((24, 32), dtype=torch.float32, device=query.device)
        output = torch.empty((1, 1, 6144), dtype=query.dtype, device=query.device)
        _gated_decode_partials[(4, 32)](
            query, key, value, positions, partial, lse,
            num_warps=4, num_stages=2, enable_fp_fusion=False,
        )
        _gated_decode_merge[(24,)](
            partial, lse, gate, positions, output,
            num_warps=4, enable_fp_fusion=False,
        )
    return output
