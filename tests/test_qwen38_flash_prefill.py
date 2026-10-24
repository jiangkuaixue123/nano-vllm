"""Initial-prefill routing/cache tests; CPU mocks are not Flash GPU evidence."""

import pytest

torch = pytest.importorskip("torch")

from nanovllm.qwen38 import model as model_module
from nanovllm.qwen38.config import Qwen38TextConfig
from nanovllm.qwen38.model import FullAttention, LayerState, rotary_tables


def patch_prefill(monkeypatch, backend, function):
    if backend == "sdpa_flash":
        monkeypatch.setattr(model_module, "flash_prefill", function)
    else:
        from nanovllm.qwen38 import _flashinfer_attention

        monkeypatch.setattr(_flashinfer_attention, "flashinfer_fa3_prefill", function)


@pytest.mark.parametrize("backend", ["sdpa_flash", "flashinfer_fa3"])
@pytest.mark.parametrize("capacity", [0, 8])
def test_initial_flash_prefill_keeps_cache_writes_and_compact_operands(monkeypatch, capacity, backend):
    monkeypatch.setenv("NANOVLLM_PREFILL_ATTENTION", backend)
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

    patch_prefill(monkeypatch, backend, fused)
    result = layer(torch.randn(1, 3, cfg.hidden_size), state, positions,
                   cos[None, None], sin[None, None], fresh_prefill=True)
    assert calls == [True] and torch.count_nonzero(result) == 0


@pytest.mark.parametrize("backend", ["sdpa_flash", "flashinfer_fa3"])
def test_cached_continuation_stays_native(monkeypatch, backend):
    monkeypatch.setenv("NANOVLLM_PREFILL_ATTENTION", backend)
    cfg = Qwen38TextConfig.for_testing()
    layer = FullAttention(cfg)
    state = LayerState()
    positions = torch.arange(2)
    cos, sin = rotary_tables(cfg, positions, torch.float32)

    def forbidden(*args):
        raise AssertionError("cached continuation must use native positional masking")

    patch_prefill(monkeypatch, backend, forbidden)
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


@pytest.mark.parametrize("backend", ["sdpa_flash", "flashinfer_fa3"])
def test_merged_flash_prefill_preserves_projection_view_layout(monkeypatch, backend):
    monkeypatch.setenv("NANOVLLM_PREFILL_ATTENTION", backend)
    cfg = Qwen38TextConfig.for_testing()
    layer = FullAttention(cfg)
    q_size = cfg.num_attention_heads * cfg.head_dim * 2
    kv_size = cfg.num_key_value_heads * cfg.head_dim
    total = q_size + 2 * kv_size

    class MergedProjection(torch.nn.Module):
        def forward(self, inputs):
            packed = torch.arange(inputs.shape[1] * total, dtype=inputs.dtype)
            return packed.reshape(1, inputs.shape[1], total).split(
                (q_size, kv_size, kv_size), dim=-1
            )

    layer.qkv_proj = MergedProjection()
    state = LayerState()
    positions = torch.arange(3)
    cos, sin = rotary_tables(cfg, positions, torch.float32)
    calls = []

    def fused(q, k, v):
        assert v.stride(2) == total
        assert not v.is_contiguous()
        assert v.shape == (1, cfg.num_key_value_heads, 3, cfg.head_dim)
        assert torch.equal(state.value, v)
        calls.append(True)
        return torch.zeros_like(q)

    patch_prefill(monkeypatch, backend, fused)
    output = layer(torch.randn(1, 3, cfg.hidden_size), state, positions,
                   cos[None, None], sin[None, None], fresh_prefill=True)
    assert calls == [True] and torch.count_nonzero(output) == 0


@pytest.mark.parametrize("backend", ["native", "sdpa_flash", "flashinfer_fa3"])
def test_loader_reports_constructed_prefill_backend(tmp_path, monkeypatch, backend):
    from test_qwen38_fp8_projections import loaded_models

    from nanovllm.qwen38.loader import load_weights

    monkeypatch.setenv("NANOVLLM_PREFILL_ATTENTION", backend)
    _, model, _, report = loaded_models(tmp_path)
    monkeypatch.setenv("NANOVLLM_PREFILL_ATTENTION", "ignored_after_construction")
    actual = load_weights(model, tmp_path)
    assert report["full_attention_prefill"] == actual["full_attention_prefill"] == {
        "backend": backend, "layers": 1,
        "scope": "fresh_prefill", "cached_continuation": "native",
    }


def test_loader_rejects_mixed_prefill_backends_before_copy(tmp_path, monkeypatch):
    from test_qwen38_fp8_projections import loaded_models

    from nanovllm.qwen38.loader import load_weights

    monkeypatch.delenv("NANOVLLM_PREFILL_ATTENTION", raising=False)
    _, model, _, _ = loaded_models(tmp_path)
    extra = FullAttention(model.config).bfloat16()
    extra.prefill_attention = "flashinfer_fa3"
    model.add_module("extra_attention", extra)
    versions = {name: value._version for name, value in model.named_parameters()}
    with pytest.raises(ValueError, match="Mixed full-attention prefill"):
        load_weights(model, tmp_path)
    assert {name: value._version for name, value in model.named_parameters()} == versions
