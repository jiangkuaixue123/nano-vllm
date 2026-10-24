"""CPU metadata and control flow checks; these do not execute CUDA arithmetic."""

import ast
import copy
from types import SimpleNamespace

import pytest
from test_qwen38_postnorm_fp8 import Device, execute, function, tree
from test_qwen38_postnorm_fp8 import Tensor as BaseTensor


class Tensor(BaseTensor):
    def __init__(self, shape=(1, 1, 17408), **kwargs):
        kwargs.setdefault("stride", (34816, 34816, 1) if len(shape) == 3 else (17408, 1))
        super().__init__(shape=shape, **kwargs)


class Projection:
    def __init__(self):
        self.in_features, self.out_features, self.use_flashinfer = 17408, 5120, True
        self.weight = Tensor((5120, 17408), dtype="fp8", pointer=10**8)
        self.weight_scale_inv = Tensor((40, 136), dtype="f32", pointer=10**9)
        self.kernel_calls, self.native_fp8_calls = {"flashinfer": 0}, 0


@pytest.fixture
def production():
    torch = SimpleNamespace(bfloat16="bf16", float8_e4m3fn="fp8", float32="f32",
                            version=SimpleNamespace(hip=None),
                            cuda=SimpleNamespace(get_device_capability=lambda device: (9, 0),
                                                 device=Device, is_current_stream_capturing=lambda: False))
    nodes = [n for n in tree("_silu_fp8.py").body if isinstance(n, ast.FunctionDef)]
    return execute(nodes, {"torch": torch, "FP8Linear": Projection})


def test_observer_accepts_merged_views_without_mutation(production):
    g, u, p = Tensor(contiguous=False), Tensor(pointer=50000, contiguous=False), Projection()
    before = copy.deepcopy([vars(t) for t in (g, u, p.weight, p.weight_scale_inv)])
    assert production["execution"](g, u, p, "native")["fallback_reason"] == "configured_native"
    info = production["execution"](g, u, p, "triton")
    assert info["selected_backend"] == "triton" and info["execution_observed"] is False
    assert [vars(t) for t in (g, u, p.weight, p.weight_scale_inv)] == before


@pytest.mark.parametrize("change,reason", [
    ("m2", "requires_m1_k17408"), ("cpu", "requires_nvidia_cuda"),
    ("sm80", "requires_sm90"), ("dtype", "requires_bf16_operands"),
    ("device", "device_mismatch"), ("stride", "unsupported_operand_layout"),
    ("grad", "unsupported_operand_layout"), ("plain", "requires_plain_fp8_down"),
    ("shape", "unsupported_projection_shape"), ("backend", "requires_flashinfer_small_m"),
    ("weight_dtype", "invalid_checkpoint_dtypes"), ("weight_device", "checkpoint_device_mismatch"),
    ("weight_layout", "invalid_checkpoint_layout"),
])
def test_unsupported_metadata_falls_back_before_execution(production, change, reason):
    g, u, p = Tensor(), Tensor(pointer=50000), Projection()
    if change == "m2":
        g.shape = u.shape = (1, 2, 17408)
    elif change == "cpu":
        g.device.type = "cpu"
    elif change == "sm80":
        production["torch"].cuda.get_device_capability = lambda device: (8, 0)
    elif change == "dtype":
        u.dtype = "f32"
    elif change == "device":
        u.device.index = 1
    elif change == "stride":
        u.strides = (34816, 34816, -1)
    elif change == "grad":
        g.requires_grad = True
    elif change == "plain":
        p = None
    elif change == "shape":
        p.out_features = 64
    elif change == "backend":
        p.use_flashinfer = False
    elif change == "weight_dtype":
        p.weight.dtype = "bf16"
    elif change == "weight_device":
        p.weight.device.index = 1
    elif change == "weight_layout":
        p.weight_scale_inv.contiguous = False
    info = production["execution"](g, u, p, "triton")
    assert (info["selected_backend"], info["fallback_reason"]) == ("native", reason)


def test_activation_configuration_is_required_and_typo_rejected(production):
    args = Tensor(), Tensor(), Projection()
    assert production["execution"](*args, "triton", "native")["fallback_reason"] == "requires_triton_mlp_activation"
    with pytest.raises(ValueError):
        production["execution"](*args, "typo")


@pytest.mark.parametrize("change", [None, "short", "compact", "offset", "dtype"])
def test_consumer_requires_entire_live_scale_backing(production, change):
    q = Tensor((1, 17408), dtype="fp8")
    s = Tensor((1, 136), dtype="f32", stride=(1, 4), capacity=2176)
    if change == "short":
        s.capacity -= 12
    elif change == "compact":
        s.strides = (136, 1)
    elif change == "offset":
        s.offset = 1
    elif change == "dtype":
        s.dtype = "bf16"
    if change:
        with pytest.raises(ValueError, match="scales"):
            production["validate_prequantized"](Projection(), q, s, (1, 1, 17408))
    else:
        production["validate_prequantized"](Projection(), q, s, (1, 1, 17408))


def test_cold_capture_rejected_and_only_complete_bridge_marks_ready(production):
    prepare = function("quant.py", "_prepare_silu_fp8", "FP8Linear")
    ns = execute([prepare], dict(production, fp8_blockscale_gemm_sm90=object()))
    p, g, u = Projection(), Tensor(), Tensor()
    production["torch"].cuda.is_current_stream_capturing = lambda: True
    with pytest.raises(RuntimeError, match="Warm"):
        ns["_prepare_silu_fp8"](p, g.device)
    production["torch"].cuda.is_current_stream_capturing = lambda: False
    events = []
    p._prepare_silu_fp8 = lambda device: events.append("prepare")
    production["silu_mul_quant"] = lambda *args: events.append("producer") or ("q", "s")
    def failed(*args):
        raise RuntimeError("synthetic launch failure")
    p._forward_silu_prequantized = failed
    with pytest.raises(RuntimeError, match="synthetic launch"):
        production["silu_fp8_down"](g, u, p)
    assert not hasattr(p, "_silu_fp8_ready_device")
    p._forward_silu_prequantized = lambda *args: events.append("consumer") or "output"
    assert production["silu_fp8_down"](g, u, p) == "output"
    assert p._silu_fp8_ready_device == g.device
    assert events == ["prepare", "producer", "prepare", "producer", "consumer"]
    production["torch"].cuda.is_current_stream_capturing = lambda: True
    assert ns["_prepare_silu_fp8"](p, g.device) is ns["fp8_blockscale_gemm_sm90"]


def test_selected_mlp_failure_propagates_without_native_retry(production):
    method = function("model.py", "_forward_gate_up", "MLP")
    events = []
    mlp = SimpleNamespace(silu_fp8_backend="triton", activation_backend="triton",
                          silu_fp8_calls={"native": 0, "triton": 0},
                          silu_fp8_execution=lambda *a: {"selected_backend": "triton"},
                          down_proj=lambda *a: events.append("native"))
    def fail(*args):
        raise RuntimeError("synthetic producer failure")
    ns = execute([method], dict(production, silu_fp8_down=fail))
    with pytest.raises(RuntimeError, match="synthetic producer"):
        ns["_forward_gate_up"](mlp, Tensor(), Tensor())
    assert events == [] and mlp.silu_fp8_calls == {"native": 0, "triton": 0}
    ns["silu_fp8_down"] = lambda *args: "candidate"
    assert ns["_forward_gate_up"](mlp, Tensor(), Tensor()) == "candidate"
    assert mlp.silu_fp8_calls == {"native": 0, "triton": 1}
    mlp.silu_fp8_execution = lambda *a: {"selected_backend": "native"}
    ns["silu_and_mul"] = lambda *args: "activation"
    ns["_forward_gate_up"](mlp, Tensor(), Tensor())
    assert events == ["native"] and mlp.silu_fp8_calls == {"native": 1, "triton": 1}


def test_consumer_uses_live_scales_and_existing_gemm_counters(production):
    method = function("quant.py", "_forward_silu_prequantized", "FP8Linear")
    q, s, p = Tensor((1, 17408), dtype="fp8"), Tensor((1, 136), dtype="f32", stride=(1, 4), capacity=2176), Projection()
    def gemm(values, weight, **kwargs):
        assert values is q and weight is p.weight
        assert kwargs == {"input_scale": s, "weight_scale": p.weight_scale_inv, "out_dtype": "bf16"}
        return SimpleNamespace(reshape=lambda *shape: shape)
    p._prepare_silu_fp8 = lambda device: gemm
    ns = execute([method], dict(production))
    assert ns["_forward_silu_prequantized"](p, q, s, (1, 1, 17408)) == (1, 1, 5120)
    assert p.kernel_calls == {"flashinfer": 1} and p.native_fp8_calls == 1
    assert not hasattr(p, "_silu_fp8_ready_device")


@pytest.mark.parametrize("requested", ["native", "triton", "typo"])
def test_constructor_selector_defaults_off(requested):
    init = function("model.py", "__init__", "MLP")
    nodes = [n for n in init.body if "silu_fp8" in ast.unparse(n)]
    reads, obj = [], SimpleNamespace()
    env = SimpleNamespace(get=lambda name, default: reads.append((name, default)) or requested)
    if requested == "typo":
        with pytest.raises(ValueError):
            execute(nodes, {"self": obj, "os": SimpleNamespace(environ=env)})
    else:
        execute(nodes, {"self": obj, "os": SimpleNamespace(environ=env)})
        assert obj.silu_fp8_backend == requested and obj.silu_fp8_calls == {"native": 0, "triton": 0}
    assert reads == [("NANOVLLM_SILU_FP8", "native")]


def test_loader_rejects_mixed_selectors_before_loading_weights():
    load = function("loader.py", "load_weights")
    nodes = [n for n in load.body if ast.unparse(n).startswith(("silu_fp8_backends =", "if len(silu_fp8_backends)"))]
    ns = {"mlps": [SimpleNamespace(silu_fp8_backend=b) for b in ("native", "triton")]}
    with pytest.raises(ValueError, match="Mixed SwiGLU"):
        execute(nodes, ns)
    assert load.body.index(nodes[-1]) < next(i for i, n in enumerate(load.body) if ast.unparse(n).startswith("shards ="))
