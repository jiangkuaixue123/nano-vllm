"""Execute adapter/constructor metadata with mocks; never GPU arithmetic evidence."""

import ast
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1] / "nanovllm" / "qwen38"


class Tensor:
    def __init__(self, shape, strides=None, *, dtype="bf16", device=None, pointer=4096):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device or SimpleNamespace(type="cuda", index=0)
        self.pointer = pointer
        if strides is None:
            strides = []
            size = 1
            for dimension in reversed(shape):
                strides.insert(0, size)
                size *= dimension
        self.strides = tuple(strides)

    @property
    def ndim(self):
        return len(self.shape)

    def stride(self, index=None):
        return self.strides if index is None else self.strides[index]

    def data_ptr(self):
        return self.pointer

    def squeeze(self, index):
        assert self.shape[index] == 1
        return self._view(self.shape[:index] + self.shape[index + 1:],
                          self.strides[:index] + self.strides[index + 1:])

    def transpose(self, first, second):
        shape, strides = list(self.shape), list(self.strides)
        shape[first], shape[second] = shape[second], shape[first]
        strides[first], strides[second] = strides[second], strides[first]
        return self._view(shape, strides)

    def unsqueeze(self, index):
        shape, strides = list(self.shape), list(self.strides)
        shape.insert(index, 1)
        strides.insert(index, self.strides[index] * self.shape[index])
        return self._view(shape, strides)

    def _view(self, shape, strides):
        return Tensor(shape, strides, dtype=self.dtype, device=self.device, pointer=self.pointer)

    def is_contiguous(self):
        size = 1
        for dimension, stride in zip(reversed(self.shape), reversed(self.strides)):
            if dimension != 1 and stride != size:
                return False
            size *= dimension
        return True


@pytest.fixture
def adapter(monkeypatch):
    tree = ast.parse((ROOT / "_flashinfer_attention.py").read_text())
    tree.body = [node for node in tree.body if not isinstance(node, ast.Import)]
    torch = SimpleNamespace(bfloat16="bf16", version=SimpleNamespace(hip=None, cuda="12.9"),
                            cuda=SimpleNamespace(get_device_capability=lambda _: (9, 0)))
    calls = []
    package = ModuleType("flashinfer")
    module = ModuleType("flashinfer.prefill")

    def run(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))
        return Tensor(q.shape, device=q.device, pointer=8192)

    module.single_prefill_with_kv_cache = run
    monkeypatch.setitem(sys.modules, "flashinfer", package)
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", module)
    namespace = {"torch": torch}
    # Execute this repository's adapter with metadata mocks, without importing Torch.
    exec(compile(tree, "adapter-under-test", "exec"), namespace)  # noqa: S102
    return namespace["flashinfer_fa3_prefill"], torch, calls, module


def operands(count=8192):
    device = SimpleNamespace(type="cuda", index=0)
    return [Tensor((1, 24, count, 256), device=device),
            Tensor((1, 4, count, 256), device=device),
            Tensor((1, 4, count, 256), (count * 14336, 256, 14336, 1),
                   device=device, pointer=4096 + 13312 * 2)]


@pytest.mark.parametrize("count", [1, 129, 8192])
def test_exact_api_parameters_and_views_without_input_copy(adapter, count):
    call, _, calls, _ = adapter
    inputs = operands(count)
    before = [(t.shape, t.strides, t.pointer) for t in inputs]
    output = call(*inputs)
    assert [(t.shape, t.strides, t.pointer) for t in inputs] == before
    q, k, v, kwargs = calls.pop()
    assert q.shape == (count, 24, 256) and k.shape == v.shape == (count, 4, 256)
    assert q.strides == k.strides == (256, count * 256, 1)
    assert v.strides == (14336, 256, 1)
    assert [t.pointer for t in (q, k, v)] == [t.pointer for t in inputs]
    assert kwargs == {"o_dtype": "bf16", "causal": True, "kv_layout": "NHD",
                      "pos_encoding_mode": "NONE", "use_fp16_qk_reduction": False,
                      "sm_scale": 0.0625, "window_left": -1, "logits_soft_cap": 0.0,
                      "backend": "fa3", "return_lse": False}
    assert output.shape == inputs[0].shape and output.pointer == 8192
    assert output.transpose(1, 2).is_contiguous()


@pytest.mark.parametrize("case", ["cpu", "hip", "dtype", "device", "rank", "batch",
                                  "q_heads", "k_heads", "head_dim", "empty", "length",
                                  "last_stride", "zero_stride", "negative_stride",
                                  "unaligned_stride", "pointer", "sm80", "sm100", "cuda_old"])
def test_unsupported_input_rejected_before_flashinfer_import(adapter, monkeypatch, case):
    call, torch, calls, _ = adapter
    inputs = operands()
    if case == "cpu":
        inputs[0].device = SimpleNamespace(type="cpu")
    elif case == "hip":
        torch.version.hip = "6.0"
    elif case == "dtype":
        inputs[1].dtype = "fp16"
    elif case == "device":
        inputs[2].device = SimpleNamespace(type="cuda", index=1)
    elif case in ("rank", "batch", "q_heads", "head_dim", "empty"):
        inputs[0].shape = {"rank": (24, 8192, 256), "batch": (2, 24, 8192, 256),
                           "q_heads": (1, 16, 8192, 256), "head_dim": (1, 24, 8192, 128),
                           "empty": (1, 24, 0, 256)}[case]
    elif case in ("k_heads", "length"):
        inputs[1].shape = (1, 8, 8192, 256) if case == "k_heads" else (1, 4, 8191, 256)
    elif case in ("last_stride", "zero_stride", "negative_stride", "unaligned_stride"):
        inputs[2].strides = {"last_stride": (1, 256, 14336, 2),
                             "zero_stride": (1, 0, 14336, 1),
                             "negative_stride": (1, -256, 14336, 1),
                             "unaligned_stride": (1, 256, 14335, 1)}[case]
    elif case == "pointer":
        inputs[2].pointer += 2
    elif case in ("sm80", "sm100"):
        torch.cuda.get_device_capability = lambda _: (8 if case == "sm80" else 10, 0)
    else:
        torch.version.cuda = "12.2"
    monkeypatch.setitem(sys.modules, "flashinfer.prefill", None)
    with pytest.raises(ValueError):
        call(*inputs)
    assert calls == []


def test_backend_failure_is_not_replaced_by_another_path(adapter):
    call, _, _, module = adapter

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic kernel failure")

    module.single_prefill_with_kv_cache = fail
    with pytest.raises(RuntimeError, match="synthetic kernel failure"):
        call(*operands())


@pytest.mark.parametrize("case", ["shape", "dtype", "device", "stride"])
def test_unexpected_output_is_rejected(adapter, case):
    call, _, _, module = adapter

    def output(q, *args, **kwargs):
        result = Tensor(q.shape, device=q.device)
        if case == "shape":
            result.shape = (8192, 4, 256)
        elif case == "dtype":
            result.dtype = "fp32"
        elif case == "device":
            result.device = SimpleNamespace(type="cuda", index=1)
        else:
            result.strides = (256, 8192 * 256, 1)
        return result

    module.single_prefill_with_kv_cache = output
    with pytest.raises(RuntimeError, match="unsupported output"):
        call(*operands())


@pytest.mark.parametrize("backend", [None, "native", "sdpa_flash", "flashinfer_fa3", "typo"])
def test_selection_is_bound_at_construction(monkeypatch, backend):
    tree = ast.parse((ROOT / "model.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FullAttention")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"]
    namespace = {"os": os, "nn": SimpleNamespace(Module=object, Linear=lambda *a, **kw: None),
                 "RMSNorm": lambda *a: None}
    # This is the repository's constructor, with arithmetic modules replaced by mocks.
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "constructor-under-test", "exec"), namespace)  # noqa: S102
    monkeypatch.delenv("NANOVLLM_PREFILL_ATTENTION", raising=False)
    monkeypatch.delenv("NANOVLLM_FULL_ROPE", raising=False)
    if backend is not None:
        monkeypatch.setenv("NANOVLLM_PREFILL_ATTENTION", backend)
    config = SimpleNamespace(num_attention_heads=24, num_key_value_heads=4, head_dim=256,
                             hidden_size=5120, rms_norm_eps=1e-6)
    if backend == "typo":
        with pytest.raises(ValueError, match="NANOVLLM_PREFILL_ATTENTION"):
            namespace["FullAttention"](config)
    else:
        module = namespace["FullAttention"](config)
        monkeypatch.setenv("NANOVLLM_PREFILL_ATTENTION", "changed_later")
        assert module.prefill_attention == (backend or "native")
