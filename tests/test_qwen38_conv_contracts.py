"""Execute actual constructor/consumer/wrapper AST with metadata-only doubles.

These checks import neither Torch nor Triton and make no numerical claim.
The separately invoked CUDA probe owns numerical and graph observations.
"""

import ast
import contextlib
import copy
import os
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1] / "nanovllm/qwen38"


def nodes(name):
    return ast.parse((ROOT / name).read_text()).body


def execute(body, namespace):
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),  # noqa: S102
                 "actual-source-fragment", "exec"), namespace)


class Tensor:
    def __init__(self, shape=(1, 33, 3), strides=None, dtype="bf16", device="cuda:0"):
        self.shape, self.ndim = shape, len(shape)
        self.strides = strides or (shape[1] * shape[2], shape[2], 1)
        self.dtype, self.device = dtype, NS(type=device.split(":")[0], name=device)

    def stride(self, dim=None):
        return self.strides if dim is None else self.strides[dim]

    def transpose(self, a, b):
        shape = list(self.shape)
        shape[a], shape[b] = shape[b], shape[a]
        return Tensor(tuple(shape))

    def reshape(self, *shape):
        return self

    def __getitem__(self, key):
        shape = tuple(len(range(*item.indices(size))) for size, item in zip(self.shape, key))
        result = Tensor(shape, self.strides, self.dtype, self.device.name)
        if hasattr(self, "clone_factory"):
            result.clone_factory = self.clone_factory
        return result

    def clone(self):
        if hasattr(self, "clone_factory"):
            return self.clone_factory(self)
        return Tensor(self.shape)

    def copy_(self, source):
        self.copied = source


def delta_class():
    cls = copy.deepcopy(next(n for n in nodes("model.py") if getattr(n, "name", None) == "DeltaNet"))
    forward = next(n for n in cls.body if getattr(n, "name", None) == "forward")
    stop = next(i for i, n in enumerate(forward.body)
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
                and [x.id for x in n.targets[0].elts] == ["query", "key", "value"])
    forward.body = forward.body[:stop] + [ast.Return(ast.Name("mixed", ast.Load()))]
    ns = {
        "nn": NS(Module=object, Linear=lambda *a, **k: None,
                 Conv1d=lambda *a, **k: NS(weight=object()), Parameter=lambda x: x),
        "torch": NS(ones=lambda *a: None, zeros=lambda *a: None),
        "os": os, "GatedRMSNorm": lambda *a: None, "__package__": "nanovllm.qwen38",
    }
    execute([cls], ns)
    return ns["DeltaNet"], ns


def config(width=4):
    return NS(hidden_size=8, linear_num_key_heads=2, linear_key_head_dim=128,
              linear_num_value_heads=6, linear_value_head_dim=128,
              linear_conv_kernel_dim=width, rms_norm_eps=1e-6)


@pytest.mark.parametrize("option", [None, "native", "triton", "invalid"])
def test_selection_is_bound_at_construction(monkeypatch, option):
    monkeypatch.delenv("NANOVLLM_GDN", raising=False)
    monkeypatch.delenv("NANOVLLM_GDN_PREFILL_QK_LAYOUT", raising=False)
    monkeypatch.delenv("NANOVLLM_GDN_CONV", raising=False)
    if option is not None:
        monkeypatch.setenv("NANOVLLM_GDN_CONV", option)
    cls, _ = delta_class()
    if option == "invalid":
        with pytest.raises(ValueError, match="NANOVLLM_GDN_CONV"):
            cls(config())
    else:
        module = cls(config())
        monkeypatch.setenv("NANOVLLM_GDN_CONV", "changed-after-construction")
        assert module.gdn_conv_backend == (option or "native")


def test_unsupported_fused_width_fails_before_execution(monkeypatch):
    monkeypatch.setenv("NANOVLLM_GDN_CONV", "triton")
    cls, _ = delta_class()
    with pytest.raises(ValueError, match="four-entry"):
        cls(config(3))


@pytest.mark.parametrize("backend", ["native", "triton"])
@pytest.mark.parametrize("buffered", [False, True])
@pytest.mark.parametrize("previous,count", [(False, 1), (False, 8192), (True, 1), (True, 2)])
def test_consumer_dispatch_and_state_ownership(monkeypatch, backend, buffered, previous, count):
    cls, ns = delta_class()
    module = object.__new__(cls)
    module.config, module.gdn_conv_backend = config(), backend
    module.in_proj_fused = lambda x: (Tensor((1, count, 33)), Tensor(), None, None)
    module.conv1d = NS(weight=object())
    old, updated = Tensor((1, 33, 4)), Tensor((1, 33, 4))
    state = NS(capacity=10240 if buffered else 0, conv=old, fresh=not previous)
    calls = []
    adapter = ModuleType("nanovllm.qwen38._triton_conv")

    def fused(raw, weight, history, *, state_out):
        calls.append(("fused", raw, weight, history, state_out))
        return Tensor((1, 33, count)), state_out if state_out is not None else updated

    adapter.causal_conv_silu = fused
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    ns["torch"].cat = lambda args, dim: Tensor((1, 33, count + 4))
    ns["F"] = NS(pad=lambda x, padding: x,
                 conv1d=lambda x, weight, **kw: calls.append(("native", kw)) or x,
                 silu=lambda x: x)
    module.forward(Tensor((1, count, 8)), state)
    if backend == "triton" and (not previous or count == 1):
        assert calls[0][0] == "fused"
        assert calls[0][2:] == (module.conv1d.weight, old if previous else None,
                                old if buffered else None)
        assert state.conv is (old if buffered else updated)
    else:
        assert calls == [("native", {"groups": 33, "padding": 0 if previous and count == 1 else 3})]
        assert (state.conv is old) is buffered


def clone_strides(tensor):
    """Pinned ATen clone/empty_like Preserve layout rules, without tensor data."""
    shape, strides = tensor.shape, tensor.stride()
    expected = 1
    for dim in sorted(range(len(shape)), key=lambda d: strides[d]):
        if shape[dim] == 1:
            continue
        if strides[dim] != expected:
            break
        expected *= shape[dim]
    else:
        return strides

    # infer_dense_strides: reverse order, then stable stride/size insertion sort.
    order = list(reversed(range(len(shape))))
    for i in range(1, len(order)):
        right = i
        for left in reversed(range(i)):
            a, b = order[left], order[right]
            if strides[a] > strides[b] or (strides[a] == strides[b] and shape[a] > shape[b]):
                order[left], order[right] = order[right], order[left]
                right = left
            elif strides[a] < strides[b]:
                break
    result, expected = [0] * len(shape), 1
    for dim in order:
        result[dim] = expected
        expected *= max(shape[dim], 1)
    return tuple(result)


def wrapper():
    launches = []
    allocations = []

    def allocate(kind, shape, strides=None, *, dtype="bf16", device="cuda:0"):
        name = device if isinstance(device, str) else device.name
        tensor = Tensor(shape, strides, dtype, name)
        tensor.clone_factory = lambda t: allocate("clone", t.shape, clone_strides(t),
                                                  dtype=t.dtype, device=t.device)
        allocations.append((kind, tensor))
        return tensor

    class Kernel:
        def __getitem__(self, grid):
            return lambda *a, **k: launches.append((grid, a, k))

    torch = NS(bfloat16="bf16", version=NS(hip=None), is_grad_enabled=lambda: False,
               cuda=NS(device=lambda _: contextlib.nullcontext()),
               empty=lambda shape, **kw: allocate("empty", shape, **kw),
               empty_strided=lambda shape, strides, **kw: allocate("empty_strided", shape, strides, **kw),
               allocations=allocations)
    ns = {"torch": torch, "triton": NS(cdiv=lambda x, y: (x + y - 1) // y),
          "_conv_silu_kernel": Kernel()}
    execute([n for n in nodes("_triton_conv.py")
             if getattr(n, "name", None) in {"_fresh_state", "causal_conv_silu"}], ns)
    return ns["causal_conv_silu"], launches, torch


@pytest.mark.parametrize("count,decode", [(1, False), (3, False), (4, False), (8192, False), (1, True)])
@pytest.mark.parametrize("buffered", [False, True])
def test_wrapper_passes_split_strides_and_retains_buffer(count, decode, buffered):
    call, launches, torch = wrapper()
    raw = Tensor((1, 33, count), (count * 48, 1, 48))
    weight = Tensor((33, 1, 4), (8, 8, 2))
    history = Tensor((1, 33, 4), (132, 1, 33)) if decode else None
    target = Tensor((1, 33, 4)) if buffered else None
    output, state = call(raw, weight, history, state_out=target)
    assert output is not raw and state is not raw
    assert state.shape == (1, 33, 4)
    assert (state is target) is buffered
    assert (state is history) is False
    grid, args, options = launches[0]
    assert args[:5] == (raw, weight, history if decode else raw, state, output)
    assert args[5:11] == (33, count, 1, 48, 8, 2)
    assert args[11:13] == ((1, 33) if decode else (0, 0))
    assert grid == ((33 + (127 if decode else 31)) // (128 if decode else 32),
                    1 if decode else (count + 31) // 32)
    assert options == {"num_warps": 4, "enable_fp_fusion": False}
    assert sum(t.device.type == "meta" for _, t in torch.allocations) == (
        3 if not buffered and not decode and count >= 4 else 0)


STATE_LAYOUT_CASES = [
    # Merged projection, packed NC, dense NCL, and accepted singleton axes.
    (10240, 4, (65536, 1, 16384), (40960, 1, 10240)),
    (10240, 33, (540672, 1, 16384), (337920, 1, 10240)),
    (10240, 8192, (134217728, 1, 16384), (83886080, 1, 10240)),
    (33, 4, (132, 1, 33), (132, 1, 33)),
    (33, 33, (1089, 1, 33), (1089, 1, 33)),
    (33, 8192, (270336, 1, 33), (270336, 1, 33)),
    (33, 4, (132, 4, 1), (132, 4, 1)),
    (33, 33, (1089, 33, 1), (132, 4, 1)),
    (33, 8192, (270336, 8192, 1), (132, 4, 1)),
    (33, 5, (165, 1, 33), (165, 1, 33)),
    (1, 4, (700, 70, 1), (700, 70, 1)),
    (1, 33, (700, 70, 1), (700, 70, 1)),
    (1, 8192, (16384, 16384, 2), (8192, 8192, 1)),
    (1, 1, (70, 70, 1), (4, 4, 1)),
    (33, 1, (48, 1, 48), (132, 4, 1)),
    (1, 3, (70, 70, 1), (4, 4, 1)),
]


@pytest.mark.parametrize("channels,count,strides,expected", STATE_LAYOUT_CASES)
def test_actual_wrapper_state_layout_uses_only_live_storage(channels, count, strides, expected):
    call, launches, torch = wrapper()
    raw = Tensor((1, channels, count), strides)
    before = (raw.shape, raw.stride(), raw.dtype, copy.deepcopy(raw.device))
    _, state = call(raw, Tensor((channels, 1, 4)))
    assert state.stride() == expected
    assert state.shape == (1, channels, 4)
    # Singleton strides contribute no storage; every live element is allocated.
    assert 1 + sum((size - 1) * stride for size, stride in zip(state.shape, state.stride())) == 4 * channels
    assert launches[0][1][13:15] == expected[1:]
    assert (raw.shape, raw.stride(), raw.dtype, raw.device) == before
    if count >= 4:
        assert [(kind, t.device.type) for kind, t in torch.allocations] == [
            ("empty_strided", "meta"), ("clone", "meta"), ("clone", "meta"),
            ("empty_strided", "cuda"), ("empty", "cuda"),
        ]
        assert torch.allocations[3][1] is state
    else:
        assert [(kind, t.device.type) for kind, t in torch.allocations] == [
            ("empty", "cuda"), ("empty", "cuda"),
        ]


@pytest.mark.parametrize("count", [33, 8192])
def test_regression_distinguishes_direct_tail_allocation(count):
    raw = Tensor((1, 10240, count), (count * 16384, 1, 16384))
    assert clone_strides(raw[:, :, -4:]) == (40960, 1, 10240)
    call, _, _ = wrapper()
    _, state = call(raw, Tensor((10240, 1, 4)))
    assert state.stride() == (count * 10240, 1, 10240)


@pytest.mark.parametrize("channels,count,strides,expected", STATE_LAYOUT_CASES)
def test_optional_torch_meta_state_matches_native_padding_clone(channels, count, strides, expected):
    torch = pytest.importorskip("torch")
    # Pending in the no-Torch control environment; no CUDA or Triton needed.
    raw = torch.empty_strided((1, channels, count), strides, dtype=torch.bfloat16, device="meta")
    reference = torch.nn.functional.pad(raw, (max(0, 4 - count), 0))[:, :, -4:].clone()
    ns = {"torch": torch}
    execute([next(n for n in nodes("_triton_conv.py") if getattr(n, "name", None) == "_fresh_state")], ns)
    actual = (ns["_fresh_state"](raw) if count >= 4 else
              torch.empty((1, channels, 4), dtype=raw.dtype, device=raw.device))
    assert actual.shape == reference.shape == (1, channels, 4)
    assert actual.stride() == reference.stride() == expected
    assert actual.untyped_storage().nbytes() == reference.untyped_storage().nbytes() == 8 * channels


@pytest.mark.parametrize("bad", ["cpu", "dtype", "device", "hip", "grad", "stride", "weight", "history", "target"])
def test_wrapper_rejects_unsupported_contract_before_launch(bad):
    call, launches, torch = wrapper()
    raw, weight, history, target = Tensor(), Tensor((33, 1, 4)), None, None
    if bad == "cpu":
        raw.device.type = "cpu"
    elif bad == "dtype":
        weight.dtype = "fp32"
    elif bad == "device":
        weight.device.name = "cuda:1"
    elif bad == "hip":
        torch.version.hip = "synthetic"
    elif bad == "grad":
        torch.is_grad_enabled = lambda: True
    elif bad == "stride":
        raw.strides = (99, 0, 1)
    elif bad == "weight":
        weight.shape = (33, 1, 3)
    elif bad == "history":
        history = Tensor((1, 33, 4))
    else:
        target = Tensor((1, 33, 3))
    with pytest.raises(ValueError):
        call(raw, weight, history, state_out=target)
    assert launches == []


def test_health_uses_constructed_backend_and_rejects_mixed_layers(monkeypatch):
    load = next(n for n in nodes("loader.py") if getattr(n, "name", None) == "load_weights")
    start = next(i for i, n in enumerate(load.body) if isinstance(n, ast.Assign)
                 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "gdn_convolutions")
    report = next(n for n in load.body if isinstance(n, ast.If)
                  and isinstance(n.test, ast.Name) and n.test.id == "gdn_convolutions")
    module = NS(gdn_conv_backend="triton", config=config())
    ns = {"gdn_modules": [module], "report": {}}
    monkeypatch.setenv("NANOVLLM_GDN_CONV", "native")
    execute(load.body[start:start + 2] + [report], ns)
    assert ns["report"]["gdn_convolution"] == {
        "backend": "triton", "layers": 1, "state_entries": 4,
        "conv_output_dtype": "bfloat16", "fresh_prefill": "triton",
        "single_token_decode": "triton", "cached_multi_token": "native",
    }
    ns["gdn_modules"].append(NS(gdn_conv_backend="native", config=config()))
    with pytest.raises(ValueError, match="disagree"):
        execute(load.body[start:start + 2], ns)
