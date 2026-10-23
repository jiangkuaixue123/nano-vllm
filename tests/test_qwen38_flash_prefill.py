"""Initial-prefill routing/cache tests; CPU mocks are not Flash GPU evidence."""

import pytest
import torch

from nanovllm.qwen38 import model as model_module
from nanovllm.qwen38.config import Qwen38TextConfig
from nanovllm.qwen38.model import FullAttention, LayerState, rotary_tables


@pytest.mark.parametrize("capacity", [0, 8])
def test_initial_flash_prefill_keeps_cache_writes_and_compact_operands(monkeypatch, capacity):
    monkeypatch.setenv("NANOVLLM_PREFILL_ATTENTION", "sdpa_flash")
    cfg = Qwen38TextConfig.for_testing()
    layer = FullAttention(cfg)
    state = LayerState(capacity=capacity)
    if capacity:
        state.key = torch.zeros(1, cfg.num_key_value_heads, capacity, cfg.head_dim)
        state.value = torch.zeros_like(state.key)
    positions = torch.arange(3)
    cos, sin = rotary_tables(cfg, positions, torch.float32)
    calls = []

    def fused(q, k, v):
        assert q.shape[2] == k.shape[2] == v.shape[2] == 3
        assert q.shape[1] == cfg.num_attention_heads and k.shape[1] == cfg.num_key_value_heads
        torch.testing.assert_close(state.key[:, :, :3], k)
        torch.testing.assert_close(state.value[:, :, :3], v)
        calls.append(True)
        return torch.zeros_like(q)

    monkeypatch.setattr(model_module, "flash_prefill", fused)
    result = layer(torch.randn(1, 3, cfg.hidden_size), state, positions,
                   cos[None, None], sin[None, None], fresh_prefill=True)
    assert calls == [True] and torch.count_nonzero(result) == 0


def test_cached_continuation_stays_native(monkeypatch):
    monkeypatch.setenv("NANOVLLM_PREFILL_ATTENTION", "sdpa_flash")
    cfg = Qwen38TextConfig.for_testing()
    layer = FullAttention(cfg)
    state = LayerState()
    positions = torch.arange(2)
    cos, sin = rotary_tables(cfg, positions, torch.float32)

    def forbidden(*args):
        raise AssertionError("cached continuation must use native positional masking")

    monkeypatch.setattr(model_module, "flash_prefill", forbidden)
    layer(torch.randn(1, 2, cfg.hidden_size), state, positions,
          cos[None, None], sin[None, None], fresh_prefill=False)
    assert state.key.shape[2] == 2
    positions = torch.arange(2, 4)
    cos, sin = rotary_tables(cfg, positions, torch.float32)
    layer(torch.randn(1, 2, cfg.hidden_size), state, positions,
          cos[None, None], sin[None, None], fresh_prefill=False)
    assert state.key.shape[2] == 4


def test_unknown_backend_and_cpu_fail_closed(monkeypatch):
    monkeypatch.setenv("NANOVLLM_PREFILL_ATTENTION", "typo")
    with pytest.raises(ValueError, match="Unsupported NANOVLLM_PREFILL_ATTENTION"):
        FullAttention(Qwen38TextConfig.for_testing())
    q = torch.zeros(1, 1, 1, 8)
    with pytest.raises(ValueError, match="requires CUDA"):
        model_module.flash_prefill(q, q, q)
