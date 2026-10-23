"""CPU selection/cache contracts; no CUDA arithmetic qualification."""

import sys

import pytest

torch = pytest.importorskip("torch")

from nanovllm.qwen38.loader import load_weights
from nanovllm.qwen38.model import GatedRMSNorm


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("merged", [False, True])
def test_native_preserves_weight_and_intermediate_semantics(monkeypatch, dtype, merged):
    monkeypatch.delenv("NANOVLLM_GDN_OUTNORM", raising=False)
    module = GatedRMSNorm(128, 1e-6).to(dtype)
    with torch.inference_mode():
        module.weight.copy_(torch.linspace(-2, 3, 128))
        x = torch.randn(1, 7, 3, 128).to(dtype)
        packed = torch.randn(1, 7, 7 * 128).to(dtype)
        gate = packed[..., 4 * 128:].view_as(x) if merged else torch.randn_like(x)
        expected = torch.nn.functional.rms_norm(x, (128,), module.weight, module.eps)
        expected = expected * torch.nn.functional.silu(gate)
        snapshot = x.clone(), gate.clone(), module.weight.clone()
        assert module.outnorm_backend == "native"
        assert torch.equal(module(x, gate), expected)
        cached = module._fused_weight(dtype)
        assert cached is module._fused_weight(dtype)
        assert torch.equal(cached, module.weight)
        assert "_fused" not in module.state_dict()
        for original, saved in zip((x, gate, module.weight), snapshot):
            assert torch.equal(original, saved)


@pytest.mark.parametrize("option", ["", "cuda", "fused", "TRITON"])
def test_unknown_outnorm_selection_rejects(monkeypatch, option):
    monkeypatch.setenv("NANOVLLM_GDN_OUTNORM", option)
    with pytest.raises(ValueError, match="NANOVLLM_GDN_OUTNORM"):
        GatedRMSNorm(128, 1e-6)


def test_backend_is_construction_bound_and_cpu_rejects_before_import(monkeypatch):
    monkeypatch.setenv("NANOVLLM_GDN_OUTNORM", "triton")
    module = GatedRMSNorm(128, 1e-6).bfloat16()
    monkeypatch.setenv("NANOVLLM_GDN_OUTNORM", "native")
    assert module.outnorm_backend == "triton"
    before = set(sys.modules)
    x = torch.ones(1, 1, 48, 128).bfloat16()
    with torch.inference_mode(), pytest.raises(ValueError, match="NVIDIA CUDA"):
        module(x, x)
    assert "nanovllm.qwen38._triton_gdn_outnorm" not in set(sys.modules) - before


@pytest.mark.parametrize("backend", ["native", "triton"])
def test_health_reports_constructed_norm_backend(tmp_path, monkeypatch, backend):
    from test_qwen38_fp8_projections import loaded_models
    from test_qwen38_service import engine

    monkeypatch.setenv("NANOVLLM_GDN_OUTNORM", backend)
    _, model, _, report = loaded_models(tmp_path)
    monkeypatch.setenv("NANOVLLM_GDN_OUTNORM", "triton" if backend == "native" else "native")
    reloaded = load_weights(model, tmp_path)
    assert report["gdn_outnorm"] == reloaded["gdn_outnorm"] == {
        "backend": backend, "layers": 1,
        "norm_intermediate_dtype": "bfloat16", "silu_intermediate_dtype": "bfloat16",
    }
    service = engine()
    service.loader_report["gdn_outnorm"] = report["gdn_outnorm"]
    assert service.provenance()["loader_report"]["gdn_outnorm"] == report["gdn_outnorm"]


def test_mixed_outnorm_options_reject_before_any_parameter_copy(tmp_path, monkeypatch):
    from test_qwen38_fp8_projections import loaded_models

    monkeypatch.delenv("NANOVLLM_GDN_OUTNORM", raising=False)
    _, model, _, _ = loaded_models(tmp_path)
    # Duplicate a GDN layer to exercise an inconsistent constructed model.
    import copy

    layer = copy.deepcopy(model.model.layers[0])
    layer.linear_attn.norm.outnorm_backend = "triton"
    model.model.layers.append(layer)
    versions = {name: value._version for name, value in model.named_parameters()}
    with pytest.raises(ValueError, match="Mixed GDN output norm"):
        load_weights(model, tmp_path)
    assert {name: value._version for name, value in model.named_parameters()} == versions
