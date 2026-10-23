"""CPU consumer contracts; CUDA arithmetic is an explicit separate probe."""

import sys

import pytest

torch = pytest.importorskip("torch")

from nanovllm.qwen38.config import Qwen38TextConfig
from nanovllm.qwen38.loader import load_weights
from nanovllm.qwen38.model import FullAttention, LayerState, apply_rope, rotary_tables


@pytest.mark.parametrize("tokens", [1, 3])
def test_native_partial_rope_keeps_materialized_products_and_tail(monkeypatch, tokens):
    monkeypatch.delenv("NANOVLLM_FULL_ROPE", raising=False)
    config = Qwen38TextConfig.for_testing()
    module = FullAttention(config).bfloat16()
    assert module.rope_backend == "native"
    x = torch.randn(1, 4, tokens, 8).bfloat16()
    cos, sin = rotary_tables(config, torch.arange(tokens), torch.bfloat16)
    cos, sin = cos[None, None], sin[None, None]
    first, second = x[..., :4].chunk(2, -1)
    expected = torch.cat((x[..., :4] * cos + torch.cat((-second, first), -1) * sin,
                          x[..., 4:]), -1)
    assert torch.equal(apply_rope(x, cos, sin).view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("option", ["", "cuda", "fused", "TRITON"])
def test_unknown_rope_option_fails_at_construction(monkeypatch, option):
    monkeypatch.setenv("NANOVLLM_FULL_ROPE", option)
    with pytest.raises(ValueError, match="NANOVLLM_FULL_ROPE"):
        FullAttention(Qwen38TextConfig.for_testing())


def test_constructed_triton_mode_rejects_cpu_before_import(monkeypatch):
    monkeypatch.setenv("NANOVLLM_FULL_ROPE", "triton")
    config = Qwen38TextConfig.for_testing()
    module = FullAttention(config).bfloat16()
    monkeypatch.setenv("NANOVLLM_FULL_ROPE", "native")
    assert module.rope_backend == "triton"
    positions = torch.arange(1)
    cos, sin = rotary_tables(config, positions, torch.bfloat16)
    before = set(sys.modules)
    with torch.inference_mode(), pytest.raises(ValueError, match="NVIDIA CUDA"):
        module(torch.ones(1, 1, config.hidden_size).bfloat16(), LayerState(), positions,
               cos[None, None], sin[None, None])
    assert "nanovllm.qwen38._triton_rope" not in set(sys.modules) - before


@pytest.mark.parametrize("backend", ["native", "triton"])
def test_loader_reports_constructed_mode_not_current_environment(tmp_path, monkeypatch, backend):
    from test_qwen38_fp8_projections import loaded_models

    monkeypatch.setenv("NANOVLLM_FULL_ROPE", backend)
    _, model, _, report = loaded_models(tmp_path)
    monkeypatch.setenv("NANOVLLM_FULL_ROPE", "triton" if backend == "native" else "native")
    actual = load_weights(model, tmp_path)
    assert report["full_attention_rope"] == actual["full_attention_rope"] == {
        "backend": backend, "layers": 1, "scope": "postnorm_partial_rope",
        "norm_backend": "native", "product_intermediate_dtype": "bfloat16",
    }


def test_loader_rejects_mixed_modes_before_copy(tmp_path, monkeypatch):
    from test_qwen38_fp8_projections import loaded_models

    monkeypatch.delenv("NANOVLLM_FULL_ROPE", raising=False)
    _, model, _, _ = loaded_models(tmp_path)
    # A second full-attention module is enough to reject before checkpoint
    # coverage/copy; this test does not pretend it is a valid architecture.
    extra = FullAttention(model.config).bfloat16()
    extra.rope_backend = "triton"
    model.add_module("extra_attention", extra)
    versions = {name: value._version for name, value in model.named_parameters()}
    with pytest.raises(ValueError, match="Mixed full-attention RoPE"):
        load_weights(model, tmp_path)
    assert {name: value._version for name, value in model.named_parameters()} == versions
