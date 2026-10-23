"""CPU adapter/layout checks with mocked FlashInfer, not GPU parity evidence."""

import copy
import json
import sys
from types import ModuleType

import pytest
import torch
from safetensors.torch import save_file

from nanovllm.qwen38 import _flashinfer_gdn
from nanovllm.qwen38.config import Qwen38TextConfig
from nanovllm.qwen38.loader import load_weights
from nanovllm.qwen38.model import DeltaNet, LayerState, Qwen38ForCausalLM


def install_prefill(monkeypatch, calls):
    module = ModuleType("flashinfer.gdn_prefill")

    def call(**kw):
        q, k, v = kw["q"], kw["k"], kw["v"]
        heads = v.shape[1]
        # Installed SM90 GVA maps output head h to floor(h * Hq / Hv).
        index = torch.arange(heads) * q.shape[1] // heads
        expanded_q = q.index_select(1, index)
        expanded_k = k.index_select(1, index)
        assert kw["backend"] == "flashinfer" and not kw["use_qk_l2norm_in_kernel"]
        assert kw["output_final_state"]
        assert kw["g"].dtype == kw["beta"].dtype == torch.float32
        assert all(kw[name].is_contiguous() for name in ("q", "k", "v", "g", "beta"))
        assert kw["cu_seqlens"].tolist() == [0, q.shape[0]]
        initial = kw["initial_state"]
        if initial is None:
            initial = torch.zeros(1, heads, v.shape[-1], q.shape[-1])
        assert initial.is_contiguous()
        # Deliberately asymmetric V/K state update checks both transposes.
        update = torch.einsum("thv,thk->hvk", v.float(), expanded_k.float()).unsqueeze(0)
        calls.append((q.shape[1], heads, q.shape[0]))
        return expanded_q + v, initial + update

    module.chunk_gated_delta_rule = call
    monkeypatch.setitem(sys.modules, module.__name__, module)


def test_real_head_ratio_first_last_groups_and_full_state(monkeypatch):
    calls = []
    install_prefill(monkeypatch, calls)
    torch.manual_seed(19)
    q = torch.randn(1, 3, 16, 128, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(1, 3, 48, 128, dtype=torch.bfloat16)
    decay = torch.randn(1, 3, 48).neg().abs().neg()
    beta = torch.randn(1, 3, 48, dtype=torch.bfloat16).sigmoid()
    initial = torch.randn(1, 48, 128, 128)
    saved = initial.clone()
    grouped = _flashinfer_gdn.prefill(q, k, v, decay, beta, initial)
    expanded = _flashinfer_gdn.prefill(q.repeat_interleave(3, 2), k.repeat_interleave(3, 2),
                                     v, decay, beta, initial)
    assert calls == [(16, 48, 3), (48, 48, 3)]
    for actual, expected in zip(grouped, expanded):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(initial, saved, rtol=0, atol=0)
    index = torch.arange(48) * 16 // 48
    assert index[[0, 2, 3, 44, 45, 47]].tolist() == [0, 0, 1, 14, 15, 15]
    assert grouped[0].shape == (1, 3, 48, 128)
    assert grouped[1].shape == (1, 48, 128, 128)
    assert grouped[1].dtype == torch.float32 and grouped[1].is_contiguous()


@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("first_count", [1, 3])
def test_prefill_only_opt_in_preserves_decode_and_continuation(monkeypatch, buffered, first_count):
    calls = []
    install_prefill(monkeypatch, calls)
    monkeypatch.setenv("NANOVLLM_GDN", "flashinfer")
    monkeypatch.setenv("NANOVLLM_GDN_PREFILL_QK_LAYOUT", "grouped")
    cfg = Qwen38TextConfig.for_testing(linear_num_value_heads=6,
                                     linear_key_head_dim=128, linear_value_head_dim=128)
    torch.manual_seed(23)
    grouped = DeltaNet(cfg).bfloat16()
    expanded = copy.deepcopy(grouped)
    expanded.gdn_prefill_qk_layout = "expanded"
    decode_heads = []

    def decode(q, k, v, initial, *gates):
        decode_heads.append((q.shape[2], k.shape[2], v.shape[2]))
        assert q.shape[2] == k.shape[2] == v.shape[2] == 6
        initial.add_(torch.einsum("bthk,bthv->bhkv", k.float(), v.float()))
        return q + v, initial

    monkeypatch.setattr(_flashinfer_gdn, "decode", decode)

    def state():
        if not buffered:
            return LayerState()
        return LayerState(capacity=16,
                          conv=torch.zeros(1, 2 * grouped.key_dim + grouped.value_dim, 4,
                                           dtype=torch.bfloat16),
                          recurrent=torch.zeros(1, 6, 128, 128))

    actual_state, expected_state = state(), state()
    for count in [first_count, 1, 2, 1]:
        x = torch.randn(1, count, cfg.hidden_size, dtype=torch.bfloat16)
        with torch.inference_mode():
            actual = grouped(x, actual_state)
            expected = expanded(x, expected_state)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_state.conv, expected_state.conv, rtol=0, atol=0)
        torch.testing.assert_close(actual_state.recurrent, expected_state.recurrent, rtol=0, atol=0)
        assert actual_state.fresh is expected_state.fresh is False
    assert calls == [(2, 6, first_count), (6, 6, first_count), (2, 6, 2), (6, 6, 2)]
    assert decode_heads == [(6, 6, 6)] * 4


@pytest.mark.parametrize("backend,layout", [("native", "grouped"), ("flashinfer", "typo")])
def test_invalid_layout_or_backend_fails_closed(monkeypatch, backend, layout):
    monkeypatch.setenv("NANOVLLM_GDN", backend)
    monkeypatch.setenv("NANOVLLM_GDN_PREFILL_QK_LAYOUT", layout)
    with pytest.raises(ValueError, match="Grouped GDN|NANOVLLM_GDN_PREFILL_QK_LAYOUT"):
        DeltaNet(Qwen38TextConfig.for_testing(linear_key_head_dim=128, linear_value_head_dim=128))


@pytest.mark.parametrize("layout", [None, "grouped"])
def test_loader_and_health_report_constructed_layout_not_later_environment(tmp_path, monkeypatch, layout):
    monkeypatch.setenv("NANOVLLM_GDN", "flashinfer")
    if layout is None:
        monkeypatch.delenv("NANOVLLM_GDN_PREFILL_QK_LAYOUT", raising=False)
    else:
        monkeypatch.setenv("NANOVLLM_GDN_PREFILL_QK_LAYOUT", layout)
    cfg = Qwen38TextConfig.for_testing(num_hidden_layers=1, layer_types=("linear_attention",),
                                     linear_num_value_heads=6,
                                     linear_key_head_dim=128, linear_value_head_dim=128)
    model = Qwen38ForCausalLM(cfg).bfloat16()
    tensors = {("model.language_model." + name.removeprefix("model.")
                if name.startswith("model.") else name): parameter.detach().contiguous()
               for name, parameter in model.named_parameters()}
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "model.safetensors" for name in tensors},
    }))
    monkeypatch.setenv("NANOVLLM_GDN_PREFILL_QK_LAYOUT", "expanded" if layout else "grouped")
    report = load_weights(model, tmp_path)
    assert report["gdn_prefill"] == {
        "backend": "flashinfer", "qk_layout": layout or "expanded", "layers": 1,
        "query_key_heads": 2 if layout else 6, "value_heads": 6, "decode_qk_layout": "expanded",
    }
    from test_qwen38_service import engine

    service = engine()
    service.loader_report["gdn_prefill"] = report["gdn_prefill"]
    assert service.provenance()["loader_report"]["gdn_prefill"] == report["gdn_prefill"]
