"""CPU contracts only; the explicit CUDA probe is not part of default CI."""

import sys

import pytest

torch = pytest.importorskip("torch")

from nanovllm.qwen38.config import Qwen38TextConfig
from nanovllm.qwen38.loader import load_weights
from nanovllm.qwen38.model import MLP


@pytest.mark.parametrize("merged", [False, True])
def test_default_preserves_bf16_intermediate_and_projection_views(monkeypatch, merged):
    monkeypatch.delenv("NANOVLLM_MLP_ACTIVATION", raising=False)
    model = MLP(Qwen38TextConfig.for_testing()).bfloat16()
    inputs = torch.randn(1, 7, model.gate_proj.in_features).bfloat16()
    with torch.inference_mode():
        gate, up = model.gate_proj(inputs), model.up_proj(inputs)
        expected = model.down_proj(torch.nn.functional.silu(gate) * up)
        if merged:
            weights = torch.cat((model.gate_proj.weight, model.up_proj.weight))
            model._fuse().weight.copy_(weights)
            actual_gate, actual_up = model.gate_up_proj(inputs)
            assert not actual_gate.is_contiguous() and not actual_up.is_contiguous()
            # GEMM row-count changes are outside this test: compare the actual
            # merged views with the original materialized activation expression.
            expected = model.down_proj(torch.nn.functional.silu(actual_gate) * actual_up)
        assert model.activation_backend == "native"
        assert torch.equal(model(inputs), expected)


@pytest.mark.parametrize("option", ["", "cuda", "fused", "TRITON"])
def test_unknown_activation_option_is_not_silently_ignored(monkeypatch, option):
    monkeypatch.setenv("NANOVLLM_MLP_ACTIVATION", option)
    with pytest.raises(ValueError, match="NANOVLLM_MLP_ACTIVATION"):
        MLP(Qwen38TextConfig.for_testing())


def test_triton_cpu_is_rejected_before_adapter_import(monkeypatch):
    monkeypatch.setenv("NANOVLLM_MLP_ACTIVATION", "triton")
    model = MLP(Qwen38TextConfig.for_testing()).bfloat16()
    monkeypatch.setenv("NANOVLLM_MLP_ACTIVATION", "native")
    assert model.activation_backend == "triton"
    before = set(sys.modules)
    with torch.inference_mode(), pytest.raises(ValueError, match="NVIDIA CUDA"):
        model(torch.ones(1, 1, model.gate_proj.in_features).bfloat16())
    assert "nanovllm.qwen38._triton_silu" not in set(sys.modules) - before


@pytest.mark.parametrize("backend", ["native", "triton"])
def test_loader_reports_constructed_backend_not_later_environment(tmp_path, monkeypatch, backend):
    from test_qwen38_fp8_projections import loaded_models

    monkeypatch.setenv("NANOVLLM_MLP_ACTIVATION", backend)
    _, model, _, report = loaded_models(tmp_path)
    monkeypatch.setenv("NANOVLLM_MLP_ACTIVATION", "triton" if backend == "native" else "native")
    actual = load_weights(model, tmp_path)
    assert report["mlp_activation"] == actual["mlp_activation"] == {
        "backend": backend, "layers": 2, "silu_intermediate_dtype": "bfloat16",
    }


def test_mixed_activation_backends_reject_before_parameter_copy(tmp_path, monkeypatch):
    from test_qwen38_fp8_projections import loaded_models

    monkeypatch.delenv("NANOVLLM_MLP_ACTIVATION", raising=False)
    _, model, _, _ = loaded_models(tmp_path)
    model.model.layers[0].mlp.activation_backend = "triton"
    versions = {name: value._version for name, value in model.named_parameters()}
    with pytest.raises(ValueError, match="Mixed MLP activation"):
        load_weights(model, tmp_path)
    assert {name: value._version for name, value in model.named_parameters()} == versions
