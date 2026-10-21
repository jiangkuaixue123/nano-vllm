"""Tiny CPU parity only: no downloaded weights or production quality claims.

Transformers is an independently executed oracle in these tests, never part of
candidate model execution. Test tolerances apply only to synthetic unit inputs.
"""

import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from nanovllm.qwen38.config import Qwen38TextConfig
from nanovllm.qwen38.model import Qwen38ForCausalLM


def make_models(dtype, seed=7, **overrides):
    torch.manual_seed(seed)
    tiny = Qwen38TextConfig.for_testing(**overrides)
    reference_config = Qwen3_5TextConfig(
        hidden_size=tiny.hidden_size,
        intermediate_size=tiny.intermediate_size,
        vocab_size=tiny.vocab_size,
        num_hidden_layers=tiny.num_hidden_layers,
        num_attention_heads=tiny.num_attention_heads,
        num_key_value_heads=tiny.num_key_value_heads,
        head_dim=tiny.head_dim,
        linear_num_key_heads=tiny.linear_num_key_heads,
        linear_num_value_heads=tiny.linear_num_value_heads,
        linear_key_head_dim=tiny.linear_key_head_dim,
        linear_value_head_dim=tiny.linear_value_head_dim,
        linear_conv_kernel_dim=tiny.linear_conv_kernel_dim,
        layer_types=list(tiny.layer_types),
        rms_norm_eps=tiny.rms_norm_eps,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=None,
        max_position_embeddings=tiny.max_position_embeddings,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": tiny.rope_theta,
            "partial_rotary_factor": tiny.partial_rotary_factor,
            "mrope_section": [1, 1, 0],
            "mrope_interleaved": True,
        },
    )
    reference_config._attn_implementation = "eager"
    reference = Qwen3_5ForCausalLM(reference_config).eval()
    # Match from_pretrained(dtype=...): weights use the selected dtype while
    # computed rotary frequencies remain FP32. A whole-module .to(dtype) would
    # also round inv_freq, creating an oracle that differs from checkpoint load.
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.data = parameter.data.to(dtype=dtype)
    native = Qwen38ForCausalLM(tiny).to(dtype=dtype).eval()
    # A strict load proves the entire native parameter surface matches the oracle.
    native.load_state_dict(reference.state_dict(), strict=True)
    return native, reference


def assert_logits(actual, expected, dtype):
    if dtype == torch.float32:
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    else:
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=1e-2, atol=2e-3
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length", [1, 7, 65])
def test_full_prefill_matches_reference(dtype, length):
    native, reference = make_models(dtype)
    ids = torch.randint(0, native.config.vocab_size, (1, length))
    with torch.inference_mode():
        expected = reference(ids, use_cache=False).logits
    actual = native(ids)
    assert actual.shape == expected.shape == (1, length, native.config.vocab_size)
    assert_logits(actual, expected, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("chunks", [(3, 1, 1, 1), (1, 2, 4), (63, 2, 1)])
def test_cached_and_chunked_prefill_match_reference(dtype, chunks):
    native, reference = make_models(dtype, seed=11)
    tokens = torch.randint(0, native.config.vocab_size, (1, sum(chunks)))
    state = native.new_state()
    cache, position = None, 0
    for count in chunks:
        ids = tokens[:, position : position + count]
        with torch.inference_mode():
            output = reference(ids, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        actual = native(ids, state=state)
        assert_logits(actual, output.logits, dtype)
        position += count
        assert state.length == position
        assert state.layers[0].recurrent.dtype == torch.float32
        assert state.layers[0].conv.shape[-1] == native.config.linear_conv_kernel_dim
        assert state.layers[1].key.shape[2] == position
        # End state as well as logits must agree; stale state can hide for one token.
        torch.testing.assert_close(
            state.layers[0].recurrent,
            cache.layers[0].recurrent_states[0],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(state.layers[0].conv, cache.layers[0].conv_states[0])


def test_request_interleaving_reset_and_recompute_are_independent():
    native, _reference = make_models(torch.float32)
    a, b = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[5, 6, 7]])
    state_a, state_b = native.new_state(), native.new_state()
    native(a[:, :2], state=state_a)
    native(b[:, :1], state=state_b)
    out_a = native(a[:, 2:], state=state_a)
    out_b = native(b[:, 1:], state=state_b)
    assert_logits(out_a, native(a)[:, 2:], torch.float32)
    assert_logits(out_b, native(b)[:, 1:], torch.float32)
    assert_logits(native(a, native.new_state()), native(a), torch.float32)
    assert (
        state_a.layers[0].recurrent.data_ptr() != state_b.layers[0].recurrent.data_ptr()
    )


def test_forward_failure_does_not_partially_commit_state(monkeypatch):
    native, _reference = make_models(torch.float32)
    state = native.new_state()
    native(torch.tensor([[1, 2]]), state=state)
    old_layers = state.layers
    snapshots = [
        (
            layer.key.clone() if layer.key is not None else None,
            layer.recurrent.clone() if layer.recurrent is not None else None,
        )
        for layer in old_layers
    ]

    def fail(_hidden):
        raise RuntimeError("synthetic projection failure")

    monkeypatch.setattr(native.lm_head, "forward", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        native(torch.tensor([[3]]), state=state)
    assert state.length == 2 and state.layers is old_layers
    for layer, (key, recurrent) in zip(state.layers, snapshots):
        if key is not None:
            torch.testing.assert_close(layer.key, key, rtol=0, atol=0)
        if recurrent is not None:
            torch.testing.assert_close(layer.recurrent, recurrent, rtol=0, atol=0)


def test_reject_invalid_inputs_context_and_foreign_state():
    native, _reference = make_models(torch.float32, max_position_embeddings=3)
    other, _reference = make_models(torch.float32)
    for ids in (
        torch.empty((1, 0), dtype=torch.long),
        torch.ones((2, 1), dtype=torch.long),
        torch.tensor([[-1]]),
        torch.tensor([[native.config.vocab_size]]),
        torch.ones((1, 1), dtype=torch.float32),
    ):
        with pytest.raises(ValueError):
            native(ids)
    with pytest.raises(ValueError, match="different model"):
        native(torch.tensor([[1]]), state=other.new_state())
    state = native.new_state()
    native(torch.tensor([[1, 2]]), state=state)
    with pytest.raises(ValueError, match="position limit"):
        native(torch.tensor([[3, 4]]), state=state)
    assert state.length == 2


def test_meta_loading_has_no_uninitialized_rotary_buffers():
    config = Qwen38TextConfig.for_testing()
    original = Qwen38ForCausalLM(config)
    with torch.device("meta"):
        restored = Qwen38ForCausalLM(config)
    restored.to_empty(device="cpu")
    restored.load_state_dict(original.state_dict(), strict=True)
    assert list(restored.named_buffers()) == []
    ids = torch.tensor([[1, 2]])
    torch.testing.assert_close(restored(ids), original(ids), rtol=0, atol=0)


def test_native_execution_does_not_import_transformers():
    path = Path(__file__).parents[1] / "nanovllm" / "qwen38" / "model.py"
    syntax = ast.parse(path.read_text())
    for node in ast.walk(syntax):
        if isinstance(node, ast.Import):
            assert all(
                not alias.name.startswith("transformers") for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("transformers")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_production_head_dimensions_and_ratios_with_adjacent_linear_layers(dtype):
    """Exercise production inner shapes without allocating 27B model weights."""
    native, reference = make_models(
        dtype,
        seed=29,
        num_attention_heads=6,
        num_key_value_heads=1,
        head_dim=256,
        linear_num_key_heads=1,
        linear_num_value_heads=3,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        partial_rotary_factor=0.25,
        num_hidden_layers=4,
        layer_types=("linear_attention",) * 3 + ("full_attention",),
    )
    ids = torch.randint(0, native.config.vocab_size, (1, 6))
    with torch.inference_mode():
        expected = reference(ids, use_cache=False).logits
    assert_logits(native(ids), expected, dtype)

    state, cache, position = native.new_state(), None, 0
    for count in (2, 3, 1):
        chunk = ids[:, position : position + count]
        with torch.inference_mode():
            output = reference(chunk, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        assert_logits(native(chunk, state=state), output.logits, dtype)
        position += count
        assert state.length == position
        for index in range(3):
            actual, oracle = state.layers[index], cache.layers[index]
            assert actual.recurrent.shape == (1, 3, 128, 128)
            assert actual.recurrent.dtype == torch.float32
            assert actual.conv.shape == (1, 640, 4)
            torch.testing.assert_close(
                actual.recurrent, oracle.recurrent_states[0], rtol=1e-5, atol=1e-6
            )
            torch.testing.assert_close(actual.conv, oracle.conv_states[0])
        attention = state.layers[3]
        assert attention.key.shape == attention.value.shape == (1, 1, position, 256)
        torch.testing.assert_close(attention.key, cache.layers[3].keys)
        torch.testing.assert_close(attention.value, cache.layers[3].values)


def test_reference_factory_matches_pretrained_buffer_dtype(tmp_path):
    """Guard the oracle itself against accidentally rounding rotary frequencies."""
    _native, reference = make_models(
        torch.bfloat16, head_dim=256, partial_rotary_factor=0.25
    )
    checkpoint = tmp_path / "synthetic-reference"
    reference.save_pretrained(checkpoint)
    loaded = Qwen3_5ForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).eval()
    assert reference.model.embed_tokens.weight.dtype == torch.bfloat16
    assert loaded.model.embed_tokens.weight.dtype == torch.bfloat16
    assert reference.model.rotary_emb.inv_freq.dtype == torch.float32
    assert loaded.model.rotary_emb.inv_freq.dtype == torch.float32
    torch.testing.assert_close(
        reference.model.rotary_emb.inv_freq,
        loaded.model.rotary_emb.inv_freq,
        rtol=0,
        atol=0,
    )
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(
            reference(ids, use_cache=False).logits,
            loaded(ids, use_cache=False).logits,
            rtol=0,
            atol=0,
        )
