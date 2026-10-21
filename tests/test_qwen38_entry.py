"""Public entry selection without importing legacy Flash Attention or CUDA."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from nanovllm import LLM, SamplingParams
from nanovllm.qwen38 import engine


def test_dedicated_dispatch_is_lazy_and_uses_configuration(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    calls = []
    sentinel = object()

    def constructor(model, **kwargs):
        calls.append((model, kwargs))
        return sentinel

    monkeypatch.setattr(engine, "DedicatedEngine", constructor)
    assert LLM(tmp_path, max_context_tokens=32, max_active_requests=2) is sentinel
    assert calls == [(tmp_path, {"max_context_tokens": 32, "max_active_requests": 2})]
    assert "nanovllm.engine.model_runner" not in sys.modules
    assert "flash_attn" not in sys.modules


def test_legacy_dispatch_preserves_constructor_options(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    module = ModuleType("nanovllm.engine.llm_engine")
    module.LLMEngine = lambda model, **kwargs: (model, kwargs)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert LLM(tmp_path, tensor_parallel_size=2, enforce_eager=True) == (
        tmp_path, {"tensor_parallel_size": 2, "enforce_eager": True},
    )


def test_greedy_sampling_is_explicit_and_unsupported_constructor_options_fail_early(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    assert SamplingParams(temperature=0).temperature == 0
    with pytest.raises(TypeError, match="unexpected keyword"):
        LLM(tmp_path, tensor_parallel_size=2)


def test_module_entrypoint_origin_is_preserved_when_executed_as_main(monkeypatch):
    from test_qwen38_service import engine as fake_engine

    from nanovllm.qwen38 import server

    main = ModuleType("__main__")
    main.__file__ = server.__file__
    main.__spec__ = SimpleNamespace(name="nanovllm.qwen38.server")
    monkeypatch.setitem(sys.modules, "__main__", main)
    monkeypatch.delitem(sys.modules, "nanovllm.qwen38.server", raising=False)
    origins = fake_engine().provenance()["module_origins"]
    assert origins["nanovllm.qwen38.server"]["path"] == server.__file__
