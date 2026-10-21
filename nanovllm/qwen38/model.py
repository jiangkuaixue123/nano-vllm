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

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .config import Qwen38TextConfig


@dataclass
class LayerState:
    key: torch.Tensor | None = None
    value: torch.Tensor | None = None
    conv: torch.Tensor | None = None
    recurrent: torch.Tensor | None = None


@dataclass
class HybridState:
    layers: list[LayerState]
    length: int = 0
    _owner: object = field(default=None, repr=False)


class RMSNorm(nn.Module):
    """Qwen zero-centered weights; preserve the FP32 multiplication order."""

    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        return (normalized * (1.0 + self.weight.float())).to(x.dtype)


class GatedRMSNorm(nn.Module):
    def __init__(self, size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x, gate):
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps
        )
        weighted = self.weight * normalized.to(x.dtype)
        return (weighted * F.silu(gate.float())).to(x.dtype)


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
        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_key_value_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_size * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(self, x, state, positions, cos, sin):
        config = self.config
        count = x.shape[1]
        # Q and gate are interleaved per head, not two contiguous projection halves.
        q, gate = self.q_proj(x).view(1, count, -1, config.head_dim * 2).chunk(2, -1)
        gate = gate.reshape(1, count, -1)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(1, count, -1, config.head_dim)).transpose(
            1, 2
        )
        v = self.v_proj(x).view(1, count, -1, config.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if state.key is not None:
            k = torch.cat((state.key, k), dim=2)
            v = torch.cat((state.value, v), dim=2)
        state.key, state.value = k, v
        repeats = config.num_attention_heads // config.num_key_value_heads
        keys, values = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
        scores = (q @ keys.transpose(-1, -2)) * config.head_dim**-0.5
        allowed = torch.arange(k.shape[2], device=x.device).unsqueeze(
            0
        ) <= positions.unsqueeze(1)
        scores = scores.masked_fill(~allowed[None, None], torch.finfo(q.dtype).min)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        result = (
            (probabilities @ values).transpose(1, 2).contiguous().reshape(1, count, -1)
        )
        # The inspected reference uses sigmoid here; its output_gate_type
        # checkpoint field does not select an alternative activation.
        return self.o_proj(result * gate.sigmoid())


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
        raw = self.in_proj_qkv(x).transpose(1, 2)
        gate = self.in_proj_z(x).reshape(1, count, -1, config.linear_value_head_dim)
        beta = self.in_proj_b(x).sigmoid()
        decay = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(x).float() + self.dt_bias
        )
        previous = state.conv is not None
        mixed = torch.cat((state.conv, raw), dim=-1) if previous else raw
        kernel = config.linear_conv_kernel_dim
        state.conv = F.pad(mixed, (max(0, kernel - mixed.shape[-1]), 0))[
            :, :, -kernel:
        ].clone()
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
        query, key = (
            query.repeat_interleave(repeats, 2),
            key.repeat_interleave(repeats, 2),
        )
        operation = recurrent_delta if previous and count == 1 else chunk_delta
        output, state.recurrent = operation(
            query, key, value, decay, beta, state.recurrent
        )
        output = self.norm(output, gate).reshape(1, count, self.value_dim)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


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

    def forward(self, x, state, positions, cos, sin):
        normalized = self.input_layernorm(x)
        mixed = (
            self.linear_attn(normalized, state)
            if self.kind == "linear_attention"
            else self.self_attn(normalized, state, positions, cos, sin)
        )
        x = x + mixed
        return x + self.mlp(self.post_attention_layernorm(x))


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
        self.config = config
        self.model = TextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self._state_owner = object()

    def new_state(self) -> HybridState:
        return HybridState(
            [LayerState() for _ in self.config.layer_types], _owner=self._state_owner
        )

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
        if (
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
        if end > self.config.max_position_embeddings:
            raise ValueError("Context exceeds model position limit")
        # Stage updates transactionally; functions never mutate old tensor storage.
        pending = [
            LayerState(layer.key, layer.value, layer.conv, layer.recurrent)
            for layer in state.layers
        ]
        hidden = self.model.embed_tokens(input_ids)
        positions = torch.arange(state.length, end, device=input_ids.device)
        cos, sin = rotary_tables(self.config, positions, hidden.dtype)
        cos, sin = cos[None, None], sin[None, None]
        for layer, layer_state in zip(self.model.layers, pending):
            hidden = layer(hidden, layer_state, positions, cos, sin)
        logits = self.lm_head(self.model.norm(hidden))
        state.layers, state.length = pending, end
        return logits
