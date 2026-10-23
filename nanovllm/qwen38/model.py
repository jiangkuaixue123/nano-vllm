# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Modifications Copyright 2026 InferMatrixLab contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. See LICENSES/Apache-2.0.txt.
#
# Derived mathematical paths: Transformers 5.20.2 Qwen3.5 modeling_qwen3_5.py
# https://github.com/huggingface/transformers/blob/v5.20.2/src/transformers/models/qwen3_5/modeling_qwen3_5.py
# This standalone implementation owns its execution and request state; it does
# not import Transformers. Vision, MTP, batching, TP and quantization are absent.
"""Eager native single-request Qwen3.8-27B text core with hybrid request state."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .config import Qwen38TextConfig


@dataclass
class LayerState:
    capacity: int = 0
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    conv: torch.Tensor | None = None
    recurrent: torch.Tensor | None = None
    fresh: bool = True          # True 表示该请求还没写过卷积/递推态


@dataclass
class HybridState:
    layers: list[LayerState]
    length: int = 0
    capacity: int = 0            # >0 表示 KV 预分配、原地写（CUDA Graph 解码用）
    positions: torch.Tensor | None = None   # 设备端位置缓冲（capacity>0 时使用）
    _owner: object = field(default=None, repr=False)


class FusedProjection(nn.Module):
    """把共享输入维的多个投影合成一份权重：(sum(out_i), in)。

    只保留一份参数，原 nn.Linear 在 fuse 时被删除（不是并存），避免多占权重显存。
    forward 一次 matmul 后按输出维切分，kernel 数由 3 降到 1。
    """

    def __init__(self, in_features, out_sizes, names, *, device=None, dtype=None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_sizes = tuple(int(x) for x in out_sizes)
        self.names = tuple(names)
        assert len(self.names) == len(self.out_sizes)
        self.weight = nn.Parameter(
            torch.empty(sum(self.out_sizes), self.in_features, device=device, dtype=dtype)
        )

    def part_ranges(self):
        out, ranges = 0, {}
        for name, size in zip(self.names, self.out_sizes):
            ranges[name] = (out, out + size)
            out += size
        return ranges

    def forward(self, x):
        return F.linear(x, self.weight).split(self.out_sizes, dim=-1)


class RMSNorm(nn.Module):
    """Qwen zero-centered weights; preserve the FP32 multiplication order."""

    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def _fused_weight(self, dtype):
        # 零中心权重 1+w 只算一次，并按 dtype 缓存。用 non-persistent buffer 而不是
        # object.__setattr__：buffer 会跟随 .to()/.cuda() 迁移，否则先 forward 再换设备
        # 会拿到旧 device 的缓存张量。
        cached = getattr(self, "_fused", None)
        if cached is None or cached.dtype is not dtype:
            self.register_buffer(
                "_fused", (1.0 + self.weight.detach().float()).to(dtype), persistent=False,
            )
            cached = self._fused
        return cached

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self._fused_weight(x.dtype), self.eps)


class GatedRMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.outnorm_backend = os.environ.get("NANOVLLM_GDN_OUTNORM", "native")
        if self.outnorm_backend not in ("native", "triton"):
            raise ValueError(f"Unsupported NANOVLLM_GDN_OUTNORM: {self.outnorm_backend}")
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def _fused_weight(self, dtype):
        cached = getattr(self, "_fused", None)
        if cached is None or cached.dtype is not dtype:
            self.register_buffer(
                "_fused", self.weight.detach().to(dtype), persistent=False,
            )
            cached = self._fused
        return cached

    def forward(self, x, gate):
        if self.outnorm_backend == "triton":
            if x.device.type != "cuda" or torch.version.hip is not None:
                raise ValueError("Triton GDN output norm requires NVIDIA CUDA")
            from ._triton_gdn_outnorm import gated_rms_norm

            return gated_rms_norm(x, gate, self._fused_weight(x.dtype), self.eps)
        # 数值路径变更：silu 由 fp32 改为按 x.dtype 计算（省两次 cast 与一次回写），
        # 与 v7 之前的行为不同，已在 PR 说明里单独标注。
        return F.rms_norm(x, (x.shape[-1],), self._fused_weight(x.dtype), self.eps) * F.silu(gate)


def rotary_tables(config, positions, dtype):
    # Raw text uses identical temporal/height/width coordinates, so the reference
    # MRoPE recomposition reduces to these ordinary partial-RoPE frequencies.
    # Recompute on device: no uninitialized buffer after meta -> to_empty loading.
    dim = int(config.head_dim * config.partial_rotary_factor)
    frequency = 1.0 / (
        config.rope_theta
        ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim)
    )
    angles = positions.float().unsqueeze(-1) * frequency
    return (
        torch.cat((angles.cos(), angles.cos()), -1).to(dtype),
        torch.cat((angles.sin(), angles.sin()), -1).to(dtype),
    )


def apply_rope(x, cos, sin):
    dim = cos.shape[-1]
    rotated, tail = x[..., :dim], x[..., dim:]
    first, second = rotated.chunk(2, dim=-1)
    half_rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotated * cos + half_rotated * sin, tail), dim=-1)


class FullAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rope_backend = os.environ.get("NANOVLLM_FULL_ROPE", "native")
        if self.rope_backend not in ("native", "triton"):
            raise ValueError(f"Unsupported NANOVLLM_FULL_ROPE: {self.rope_backend}")
        self.prefill_attention = os.environ.get("NANOVLLM_PREFILL_ATTENTION", "native")
        if self.prefill_attention not in ("native", "sdpa_flash"):
            raise ValueError(f"Unsupported NANOVLLM_PREFILL_ATTENTION: {self.prefill_attention}")
        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_size * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, x, state, positions, cos, sin, *, fresh_prefill=False):
        config = self.config
        count = x.shape[1]
        # Q and gate are interleaved per head, not two contiguous projection halves.
        if hasattr(self, "qkv_proj"):
            q_lin, k_lin, v_lin = self.qkv_proj(x)
        else:
            q_lin, k_lin, v_lin = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q, gate = q_lin.view(1, count, -1, config.head_dim * 2).chunk(2, -1)
        gate = gate.reshape(1, count, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k_lin.view(1, count, -1, config.head_dim)).transpose(1, 2)
        v = v_lin.view(1, count, -1, config.head_dim).transpose(1, 2)
        if self.rope_backend == "triton":
            if q.device.type != "cuda" or torch.version.hip is not None:
                raise ValueError("Triton partial RoPE requires NVIDIA CUDA")
            from ._triton_rope import paired_partial_rope

            q, k = paired_partial_rope(q, k, cos, sin)
        else:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if state.capacity:
            index = positions.reshape(-1)
            state.key.index_copy_(2, index, k)
            state.value.index_copy_(2, index, v)
        else:
            if state.key is not None:
                k = torch.cat((state.key, k), dim=2)
                v = torch.cat((state.value, v), dim=2)
            state.key, state.value = k, v
        if fresh_prefill and self.prefill_attention == "sdpa_flash":
            result = flash_prefill(q, k, v).transpose(1, 2).reshape(1, count, -1)
            return self.o_proj(result * gate.sigmoid())
        kv_heads = config.num_key_value_heads
        repeats = config.num_attention_heads // kv_heads
        # 分组 bmm，且**不做 broadcast**：q 按 (kv_head, repeat*count) 压成 batch=kv_heads，
        # KV 只转置不展开。写成 [1,4,6,count,d] @ [1,4,1,d,T] 会让 PyTorch 把 KV 的
        # repeat 维物化（实测 8K 单步 8.7ms / 16 层，形状 1x4x6x256x8195），这正是那
        # 14ms 非 GEMM 里最大的一块。改为 batch=4 的等形 bmm 后不再产生该拷贝。
        q_grouped = q.reshape(kv_heads, repeats * count, config.head_dim)
        keys, values = (
            (state.key, state.value) if state.capacity and not fresh_prefill else (k, v)
        )
        # Initial prefill has no cached prefix. Keep its original K/V operands
        # and reduction shape even when reserving storage for later graph decode.
        # Reading the padded cache here changes BF16 rounding before capture and
        # can change subsequent greedy tokens. The writes above still populate
        # the fixed addresses consumed by replay.
        keys_bt = keys.reshape(kv_heads, -1, config.head_dim).transpose(1, 2)
        scores = torch.matmul(q_grouped, keys_bt).reshape(
            1, kv_heads, repeats, count, keys.shape[2]
        ) * config.head_dim**-0.5
        if state.capacity or count > 1:
            # decode 单 token 时所有 key 都有效，mask 只在 prefill 建。
            allowed = torch.arange(keys.shape[2], device=x.device).unsqueeze(
                0
            ) <= positions.unsqueeze(1)
            scores = scores.masked_fill(
                ~allowed[None, None, None], torch.finfo(q.dtype).min
            )
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        grouped = torch.matmul(
            probabilities.reshape(kv_heads, repeats * count, keys.shape[2]),
            values.reshape(kv_heads, -1, config.head_dim),
        ).reshape(1, kv_heads, repeats, count, config.head_dim)
        result = (
            grouped.reshape(1, config.num_attention_heads, count, config.head_dim)
            .transpose(1, 2)
            .contiguous()
            .reshape(1, count, -1)
        )
        # The inspected reference uses sigmoid here; its output_gate_type
        # checkpoint field does not select an alternative activation.
        return self.o_proj(result * gate.sigmoid())


def flash_prefill(query, key, value):
    """Initial causal prefill only; require the actual FlashAttention backend."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not query.is_cuda:
        raise ValueError("sdpa_flash prefill requires CUDA")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True, enable_gqa=True
        )


def normalize_qk(query, key):
    def normalize(x):
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    return normalize(query), normalize(key)


def recurrent_delta(query, key, value, decay, beta, initial):
    dtype = query.dtype
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    query, key = normalize_qk(query, key)
    query = query / (query.shape[-1] ** 0.5)
    state = initial.to(value)
    outputs = []
    for step in range(query.shape[2]):
        q, k, v = query[:, :, step], key[:, :, step], value[:, :, step]
        state = state * decay[:, :, step].exp()[..., None, None]
        prediction = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - prediction) * beta[:, :, step].unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state * q.unsqueeze(-1)).sum(dim=-2))
    return torch.stack(outputs, dim=2).transpose(1, 2).contiguous().to(dtype), state


def chunk_delta(query, key, value, decay, beta, initial):
    """Reference-order 64-token triangular DeltaNet prefill; accumulation FP32."""
    dtype, length = query.dtype, query.shape[1]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, decay)
    ]
    query, key = normalize_qk(query, key)
    query = query * query.shape[-1] ** -0.5
    chunk = 64
    padding = (chunk - length % chunk) % chunk
    query, key, value = [F.pad(x, (0, 0, 0, padding)) for x in (query, key, value)]
    beta, decay = F.pad(beta, (0, padding)), F.pad(decay, (0, padding))
    weighted_v, weighted_k = value * beta.unsqueeze(-1), key * beta.unsqueeze(-1)
    query, key, weighted_k, weighted_v = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk, x.shape[-1])
        for x in (query, key, weighted_k, weighted_v)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk)
    cumulative = decay.cumsum(dim=3)
    future = torch.ones(chunk, chunk, dtype=torch.bool, device=query.device).triu(1)
    pairwise = (
        (cumulative.unsqueeze(4) - cumulative.unsqueeze(3))
        .masked_fill(future, float("-inf"))
        .exp()
    )
    system = (weighted_k @ key.transpose(-1, -2)) * pairwise
    attention = (query @ key.transpose(-1, -2)) * pairwise
    weighted_decayed_k = weighted_k * cumulative.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(
        system, weighted_v, upper=False, unitriangular=True
    )
    decayed_keys = torch.linalg.solve_triangular(
        system, weighted_decayed_k, upper=False, unitriangular=True
    )
    if initial is None:
        state = torch.zeros(
            (1, value.shape[1], key.shape[-1], value.shape[-1]),
            dtype=value.dtype,
            device=value.device,
        )
    else:
        state = initial.to(new_values)
    query = query * cumulative.exp().unsqueeze(-1)
    key = key * (cumulative[..., -1:] - cumulative).exp().unsqueeze(-1)
    chunk_decay = cumulative[..., -1].exp()[..., None, None]
    outputs = []
    for index in range(query.shape[2]):
        update = new_values[:, :, index] - decayed_keys[:, :, index] @ state
        outputs.append(query[:, :, index] @ state + attention[:, :, index] @ update)
        state = (
            state * chunk_decay[:, :, index]
            + key[:, :, index].transpose(-1, -2) @ update
        )
    result = torch.stack(outputs, dim=2).reshape(1, value.shape[1], -1, value.shape[-1])
    return result[:, :, :length].transpose(1, 2).to(dtype).contiguous(), state


class DeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gdn_backend = os.environ.get("NANOVLLM_GDN", "native")
        if self.gdn_backend not in ("native", "flashinfer"):
            raise ValueError(f"Unsupported NANOVLLM_GDN: {self.gdn_backend}")
        self.gdn_conv_backend = os.environ.get("NANOVLLM_GDN_CONV", "native")
        if self.gdn_conv_backend not in ("native", "triton"):
            raise ValueError("Unsupported NANOVLLM_GDN_CONV")
        if self.gdn_conv_backend == "triton" and config.linear_conv_kernel_dim != 4:
            raise ValueError("Triton GDN convolution requires a four-entry window")
        self.gdn_prefill_qk_layout = os.environ.get("NANOVLLM_GDN_PREFILL_QK_LAYOUT", "expanded")
        if self.gdn_prefill_qk_layout not in ("expanded", "grouped"):
            raise ValueError("Unsupported NANOVLLM_GDN_PREFILL_QK_LAYOUT")
        if self.gdn_prefill_qk_layout == "grouped" and self.gdn_backend != "flashinfer":
            raise ValueError("Grouped GDN prefill Q/K requires FlashInfer")
        if self.gdn_backend == "flashinfer" and (
            config.linear_key_head_dim != 128 or config.linear_value_head_dim != 128
        ):
            raise ValueError("FlashInfer GDN requires the dedicated 128-dimensional heads")
        self.key_dim = config.linear_num_key_heads * config.linear_key_head_dim
        self.value_dim = config.linear_num_value_heads * config.linear_value_head_dim
        conv_dim = 2 * self.key_dim + self.value_dim
        self.in_proj_qkv = nn.Linear(config.hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(
            config.hidden_size, config.linear_num_value_heads, bias=False
        )
        self.in_proj_a = nn.Linear(
            config.hidden_size, config.linear_num_value_heads, bias=False
        )
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            config.linear_conv_kernel_dim,
            groups=conv_dim,
            bias=False,
            padding=config.linear_conv_kernel_dim - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(config.linear_num_value_heads))
        self.A_log = nn.Parameter(torch.zeros(config.linear_num_value_heads))
        self.norm = GatedRMSNorm(config.linear_value_head_dim, config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

    def forward(self, x, state):
        config, count = self.config, x.shape[1]
        if hasattr(self, "in_proj_fused"):
            qkv_lin, z_lin, b_lin, a_lin = self.in_proj_fused(x)
        else:
            if hasattr(self, "in_proj_qkvz"):
                qkv_lin, z_lin = self.in_proj_qkvz(x)
            else:
                qkv_lin, z_lin = self.in_proj_qkv(x), self.in_proj_z(x)
            b_lin, a_lin = (
                self.in_proj_ba(x) if hasattr(self, "in_proj_ba")
                else (self.in_proj_b(x), self.in_proj_a(x))
            )
        raw = qkv_lin.transpose(1, 2)
        gate = z_lin.reshape(1, count, -1, config.linear_value_head_dim)
        buffered = state.capacity > 0 and state.conv is not None
        previous = state.conv is not None and not state.fresh
        if self.gdn_conv_backend == "triton" and (not previous or count == 1):
            from ._triton_conv import causal_conv_silu

            convolved, state.conv = causal_conv_silu(
                raw, self.conv1d.weight, state.conv if previous else None,
                state_out=state.conv if buffered else None,
            )
            mixed = convolved.transpose(1, 2)
        else:
            # Default and cached multi-token continuation retain the native path.
            mixed = torch.cat((state.conv, raw), dim=-1) if previous else raw
            kernel = config.linear_conv_kernel_dim
            window = F.pad(mixed, (max(0, kernel - mixed.shape[-1]), 0))[:, :, -kernel:]
            if buffered:
                state.conv.copy_(window)
            else:
                state.conv = window.clone()
            padding = 0 if previous and count == 1 else kernel - 1
            convolved = F.conv1d(
                mixed, self.conv1d.weight, groups=mixed.shape[1], padding=padding
            )
            if padding:
                convolved = convolved[:, :, : mixed.shape[-1]]
            mixed = F.silu(convolved[:, :, -count:]).transpose(1, 2)
        query, key, value = mixed.split(
            (self.key_dim, self.key_dim, self.value_dim), dim=-1
        )
        query = query.reshape(1, count, -1, config.linear_key_head_dim)
        key = key.reshape(1, count, -1, config.linear_key_head_dim)
        value = value.reshape(1, count, -1, config.linear_value_head_dim)
        repeats = config.linear_num_value_heads // config.linear_num_key_heads
        # FlashInfer prefill accepts grouped-value attention: output head h uses
        # Q/K head h // repeats. Normalize each original head once in the bridge.
        # Single-token decode and the default/native paths retain expanded heads.
        if self.gdn_prefill_qk_layout == "expanded" or (previous and count == 1):
            query, key = (
                query.repeat_interleave(repeats, 2),
                key.repeat_interleave(repeats, 2),
            )
        current = None if state.fresh else state.recurrent
        if self.gdn_backend == "flashinfer":
            from . import _flashinfer_gdn

            if previous and count == 1:
                output, updated = _flashinfer_gdn.decode(
                    query, key, value, current if buffered else current.clone(),
                    self.A_log, a_lin, self.dt_bias, b_lin
                )
            else:
                beta = b_lin.sigmoid()
                decay = -self.A_log.float().exp() * F.softplus(a_lin.float() + self.dt_bias)
                output, updated = _flashinfer_gdn.prefill(
                    query, key, value, decay, beta, current
                )
        else:
            beta = b_lin.sigmoid()
            decay = -self.A_log.float().exp() * F.softplus(a_lin.float() + self.dt_bias)
            operation = recurrent_delta if previous and count == 1 else chunk_delta
            output, updated = operation(query, key, value, decay, beta, current)
        if buffered:
            state.recurrent.copy_(updated)
        else:
            state.recurrent = updated
        state.fresh = False
        output = self.norm(output, gate).reshape(1, count, self.value_dim)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_backend = os.environ.get("NANOVLLM_MLP_ACTIVATION", "native")
        if self.activation_backend not in ("native", "triton"):
            raise ValueError(f"Unsupported NANOVLLM_MLP_ACTIVATION: {self.activation_backend}")
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def _fuse(self):
        fused = FusedProjection(
            self.gate_proj.in_features,
            (self.gate_proj.weight.shape[0], self.up_proj.weight.shape[0]),
            ("gate_proj", "up_proj"),
            device=self.gate_proj.weight.device, dtype=self.gate_proj.weight.dtype,
        )
        del self.gate_proj, self.up_proj
        self.gate_up_proj = fused
        return fused

    def forward(self, x):
        if hasattr(self, "gate_up_proj"):
            gate, up = self.gate_up_proj(x)
        else:
            gate, up = self.gate_proj(x), self.up_proj(x)
        if self.activation_backend == "triton":
            if gate.device.type != "cuda" or torch.version.hip is not None:
                raise ValueError("Triton MLP activation requires NVIDIA CUDA")
            from ._triton_silu import silu_and_mul

            return self.down_proj(silu_and_mul(gate, up))
        return self.down_proj(F.silu(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.kind = kind
        if kind == "linear_attention":
            self.linear_attn = DeltaNet(config)
        else:
            self.self_attn = FullAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, state, positions, cos, sin, *, fresh_prefill=False):
        normalized = self.input_layernorm(x)
        mixed = (
            self.linear_attn(normalized, state)
            if self.kind == "linear_attention"
            else self.self_attn(normalized, state, positions, cos, sin,
                                fresh_prefill=fresh_prefill)
        )
        # 残差相加与 post-attention norm 折成一个 kernel：x ← x + mixed（原地），
        # 同时产出归一化结果，少一次物化。数值语义与分开写相同。
        norm = self.post_attention_layernorm
        if x.is_cuda and x.dtype == torch.bfloat16 and x.shape[-1] == norm.weight.shape[0]:
            from ._triton_rms import fused_add_rms

            x, normed = fused_add_rms(x, mixed, norm._fused_weight(x.dtype), norm.eps)
        else:
            x = x + mixed
            normed = norm(x)
        return x + self.mlp(normed)


class TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config, kind) for kind in config.layer_types]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)


class Qwen38ForCausalLM(nn.Module):
    def __init__(self, config: Qwen38TextConfig):
        super().__init__()
        self.prefill_logits = os.environ.get("NANOVLLM_PREFILL_LOGITS", "all")
        if self.prefill_logits not in ("all", "last"):
            raise ValueError("NANOVLLM_PREFILL_LOGITS must be all or last")
        self.config = config
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._state_owner = object()

    def fuse_projections(self):
        """BF16 路径把 q/k/v、in_proj_*、gate/up 各合成一份权重（删除原 Linear）。"""
        count = 0
        for module in self.modules():
            if isinstance(module, MLP):
                module._fuse()
                count += 1
            elif isinstance(module, FullAttention):
                fused = FusedProjection(
                    module.q_proj.in_features,
                    (module.q_proj.weight.shape[0], module.k_proj.weight.shape[0],
                     module.v_proj.weight.shape[0]),
                    ("q_proj", "k_proj", "v_proj"),
                    device=module.q_proj.weight.device, dtype=module.q_proj.weight.dtype,
                )
                del module.q_proj, module.k_proj, module.v_proj
                module.qkv_proj = fused
                count += 1
            elif isinstance(module, DeltaNet):
                fused = FusedProjection(
                    module.in_proj_qkv.in_features,
                    (module.in_proj_qkv.weight.shape[0], module.in_proj_z.weight.shape[0],
                     module.in_proj_b.weight.shape[0], module.in_proj_a.weight.shape[0]),
                    ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
                    device=module.in_proj_qkv.weight.device,
                    dtype=module.in_proj_qkv.weight.dtype,
                )
                del module.in_proj_qkv, module.in_proj_z, module.in_proj_b, module.in_proj_a
                module.in_proj_fused = fused
                count += 1
        return count

    def new_state(self, capacity: int | None = None) -> HybridState:
        layers = [LayerState() for _ in self.config.layer_types]
        if capacity:
            kv = self.config.num_key_value_heads
            hd = self.config.head_dim
            dtype = self.model.embed_tokens.weight.dtype
            device = self.model.embed_tokens.weight.device
            for layer, kind in zip(layers, self.config.layer_types):
                layer.capacity = capacity
                if kind == "full_attention":
                    layer.key = torch.zeros(1, kv, capacity, hd, dtype=dtype, device=device)
                    layer.value = torch.zeros(1, kv, capacity, hd, dtype=dtype, device=device)
                else:
                    # CUDA Graph 抓的是固定地址：卷积窗与递推态必须原地写，否则图内每步
                    # 都读到 capture 时那一份，DeltaNet 的状态永远不累积。
                    key_dim = self.config.linear_num_key_heads * self.config.linear_key_head_dim
                    value_dim = (self.config.linear_num_value_heads
                                 * self.config.linear_value_head_dim)
                    layer.conv = torch.zeros(
                        1, 2 * key_dim + value_dim, self.config.linear_conv_kernel_dim,
                        dtype=dtype, device=device,
                    )
                    layer.recurrent = torch.zeros(
                        1, self.config.linear_num_value_heads,
                        self.config.linear_key_head_dim, self.config.linear_value_head_dim,
                        dtype=torch.float32, device=device,
                    )
        return HybridState(layers, capacity=capacity or 0, _owner=self._state_owner)

    @torch.inference_mode()
    def forward(
        self, input_ids: torch.Tensor, state: HybridState | None = None
    ) -> torch.Tensor:
        if (
            input_ids.ndim != 2
            or input_ids.shape[0] != 1
            or input_ids.shape[1] < 1
            or input_ids.dtype != torch.long
        ):
            raise ValueError("Expected nonempty int64 token IDs with shape [1, tokens]")
        if input_ids.device != self.model.embed_tokens.weight.device:
            raise ValueError("Token IDs and model must be on the same device")
        if (state is None or (state.length == 0 and state.positions is None)) and (
            input_ids.min().item() < 0
            or input_ids.max().item() >= self.config.vocab_size
        ):
            raise ValueError("Token ID outside model vocabulary")
        state = self.new_state() if state is None else state
        if (
            not isinstance(state, HybridState)
            or state._owner is not self._state_owner
            or len(state.layers) != self.config.num_hidden_layers
        ):
            raise ValueError("State belongs to a different model")
        if type(state.length) is not int or state.length < 0:
            raise ValueError("Invalid state position")
        end = state.length + input_ids.shape[1]
        if state.positions is None and end > self.config.max_position_embeddings:
            raise ValueError("Context exceeds model position limit")
        # Stage updates transactionally; functions never mutate old tensor storage.
        pending = [
            LayerState(capacity=layer.capacity, key=layer.key, value=layer.value,
                       conv=layer.conv, recurrent=layer.recurrent, fresh=layer.fresh)
            for layer in state.layers
        ]
        hidden = self.model.embed_tokens(input_ids)
        positions = (
            state.positions
            if state.positions is not None
            else torch.arange(state.length, end, device=input_ids.device)
        )
        cos, sin = rotary_tables(self.config, positions, hidden.dtype)
        cos, sin = cos[None, None], sin[None, None]
        for layer, layer_state in zip(self.model.layers, pending):
            hidden = layer(hidden, layer_state, positions, cos, sin,
                           fresh_prefill=state.length == 0 and state.positions is None)
        # Every input token still updates attention/recurrent state. Generation
        # needs only the final row; select it before the vocabulary projection.
        # The opt-in also covers cached multi-token continuations. Default mode
        # retains the complete [1, tokens, vocabulary] model API.
        if self.prefill_logits == "last":
            hidden = hidden[:, -1:, :]
        logits = self.lm_head(self.model.norm(hidden))
        state.layers = pending
        if state.positions is None:
            state.length = end
        return logits
