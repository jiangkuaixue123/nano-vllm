"""Adapter contract tests only; mocked FlashInfer is not GPU qualification."""

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch

from nanovllm.qwen38 import _flashinfer_gdn
from nanovllm.qwen38.config import Qwen38TextConfig
from nanovllm.qwen38.model import DeltaNet, LayerState


def test_prefill_state_transpose_and_decay_not_log(monkeypatch):
    module = ModuleType("flashinfer.gdn_prefill")
    initial = torch.arange(32, dtype=torch.float32).view(1, 2, 4, 4)
    q = torch.randn(1, 3, 2, 4, dtype=torch.bfloat16)
    decay = torch.full((1, 3, 2), -2.0)
    beta = torch.full((1, 3, 2), .5, dtype=torch.bfloat16)

    def call(**kw):
        torch.testing.assert_close(kw["initial_state"], initial.transpose(-1, -2))
        assert kw["initial_state"].is_contiguous()
        assert kw["cu_seqlens"].tolist() == [0, 3]
        assert kw["g"].dtype == kw["beta"].dtype == torch.float32
        torch.testing.assert_close(kw["g"], decay.squeeze(0).exp())
        assert not kw["use_qk_l2norm_in_kernel"]
        assert kw["backend"] == "flashinfer" and kw["output_final_state"]
        return kw["v"].clone(), kw["initial_state"] + 1

    module.chunk_gated_delta_rule = call
    monkeypatch.setitem(sys.modules, module.__name__, module)
    out, state = _flashinfer_gdn.prefill(q, q, q, decay, beta, initial)
    assert out.shape == q.shape
    torch.testing.assert_close(state, initial + 1)
    assert state.dtype == torch.float32 and state.is_contiguous()


def test_decode_passes_raw_gates_and_mutable_kv_state(monkeypatch):
    module = ModuleType("flashinfer.gdn_decode")
    q = torch.randn(1, 1, 2, 128, dtype=torch.bfloat16)
    state = torch.zeros(1, 2, 128, 128)
    a = b = torch.ones(1, 1, 2, dtype=torch.bfloat16)
    log = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
    bias = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))

    def call(**kw):
        assert kw["state"] is state and kw["a"] is a and kw["b"] is b
        assert kw["A_log"].dtype == torch.float32
        assert not kw["A_log"].requires_grad and not kw["dt_bias"].requires_grad
        assert kw["A_log"].data_ptr() == log.data_ptr()
        assert kw["dt_bias"].data_ptr() == bias.data_ptr()
        assert kw["use_qk_l2norm"] and kw["backend"] == "flashinfer"
        kw["state"].add_(1)
        return kw["v"], kw["state"]

    module.gated_delta_rule_decode = call
    monkeypatch.setitem(sys.modules, module.__name__, module)
    out, updated = _flashinfer_gdn.decode(q, q, q, state, log, a, bias, b)
    assert updated is state and out is q and torch.all(state == 1)


@pytest.mark.parametrize("buffered", [False, True])
def test_decode_clones_only_unbuffered_transactional_state(monkeypatch, buffered):
    monkeypatch.setenv("NANOVLLM_GDN", "flashinfer")
    cfg = Qwen38TextConfig.for_testing(linear_key_head_dim=128, linear_value_head_dim=128)
    model = DeltaNet(cfg)
    conv = torch.zeros(1, 2 * model.key_dim + model.value_dim, cfg.linear_conv_kernel_dim)
    original = torch.zeros(1, cfg.linear_num_value_heads, 128, 128)
    state = LayerState(capacity=8 if buffered else 0, conv=conv, recurrent=original, fresh=False)

    def decode(q, k, v, recurrent, *rest):
        assert (recurrent is original) == buffered
        recurrent.add_(1)
        return v, recurrent

    monkeypatch.setattr(_flashinfer_gdn, "decode", decode)
    model(torch.zeros(1, 1, cfg.hidden_size), state)
    assert torch.all(state.recurrent == 1)
    assert torch.all(original == (1 if buffered else 0))


def test_unknown_backend_and_unsupported_heads_fail_closed(monkeypatch):
    monkeypatch.setenv("NANOVLLM_GDN", "typo")
    with pytest.raises(ValueError, match="Unsupported NANOVLLM_GDN"):
        DeltaNet(Qwen38TextConfig.for_testing())
    monkeypatch.setenv("NANOVLLM_GDN", "flashinfer")
    with pytest.raises(ValueError, match="128-dimensional"):
        DeltaNet(Qwen38TextConfig.for_testing())


def test_graph_prewarms_actual_decode_branch_and_restores_fresh_state(monkeypatch):
    from nanovllm.qwen38 import graph as graph_module

    capturing = False
    decode_warmed = False
    calls = []
    layer = LayerState(capacity=1, conv=torch.zeros(1), recurrent=torch.zeros(1))
    state = SimpleNamespace(layers=[layer], positions=None)

    class Model:
        model = SimpleNamespace(embed_tokens=SimpleNamespace(weight=torch.zeros(1)))

        def __call__(self, ids, *, state):
            nonlocal decode_warmed
            if capturing:
                assert decode_warmed and not state.layers[0].fresh
            elif not state.layers[0].fresh:
                decode_warmed = True
            calls.append((capturing, state.layers[0].fresh))
            state.layers[0].fresh = False
            state.layers[0].recurrent.add_(1)
            return ids.float()

    @contextmanager
    def capture(*args, **kwargs):
        nonlocal capturing
        capturing = True
        yield
        capturing = False

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: object())
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(graph_module, "capture_pool", lambda owner: None)
    graph_module.DecodeGraph(Model(), state, 0, 1).capture()
    assert calls == [(False, True), (False, False), (True, False)]
    assert layer.fresh and torch.count_nonzero(layer.recurrent) == 0
