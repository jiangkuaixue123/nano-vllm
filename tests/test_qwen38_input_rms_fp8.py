"""Focused production routing/consumer/lifetime controls without Torch imports."""

import ast
import copy
import math
from types import SimpleNamespace

import pytest
from test_qwen38_postnorm_fp8 import Device, Tensor, execute, function, tree


class Projection:
    def __init__(self, keep=False):
        self.names, self.out_sizes = (
            (("in_proj_qkv", "in_proj_z"), (10240, 6144))
            if keep
            else (("q_proj", "k_proj", "v_proj"), (12288, 1024, 1024))
        )
        self.in_features, self.out_features, self.use_flashinfer = (
            5120,
            sum(self.out_sizes),
            True,
        )
        self.weight = Tensor((self.out_features, 5120), "fp8", 10**8)
        self.weight_scale_inv = Tensor((self.out_features // 128, 40), "f32", 10**9)
        self.kernel_calls, self.native_fp8_calls = {"flashinfer": 0}, 0


class Norm:
    def __init__(self):
        self.weight = Tensor((5120,), pointer=50000)
        self.weight.requires_grad = (
            True  # Original nn.Parameter is allowed; consumed cache is detached.
        )
        self.gamma = Tensor((5120,), pointer=70000)
        self.eps, self.cache_calls = 1e-6, 0

    def _fused_weight(self, dtype):
        self.cache_calls += 1
        return self.gamma

    def __call__(self, x):
        return "native_norm"


@pytest.fixture
def production():
    torch = SimpleNamespace(
        bfloat16="bf16",
        float8_e4m3fn="fp8",
        float32="f32",
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(
            device=Device,
            get_device_capability=lambda device: (9, 0),
            is_current_stream_capturing=lambda: False,
        ),
    )
    nodes = [
        n for n in tree("_input_rms_fp8.py").body if isinstance(n, ast.FunctionDef)
    ]
    return execute(
        nodes,
        {"torch": torch, "math": math, "FP8MergedLinear": Projection, "RMSNorm": Norm},
    )


@pytest.mark.parametrize("keep", [False, True])
def test_readonly_route_and_cache(production, keep):
    x, norm, projection = Tensor(), Norm(), Projection(keep)
    before = copy.deepcopy(vars(norm.weight))
    assert (
        production["execution"](x, norm, projection, "triton", keep_bf16=keep)[
            "selected_backend"
        ]
        == "triton"
    )
    assert (
        production["execution"](x, norm, projection, "native", keep_bf16=keep)[
            "fallback_reason"
        ]
        == "configured_native"
    )
    assert vars(norm.weight) == before and norm.cache_calls == 0
    with pytest.raises(ValueError):
        production["execution"](x, norm, projection, "typo", keep_bf16=keep)


def test_loader_report_discovers_only_decoder_instances():
    body = function("loader.py", "load_weights").body
    start = next(
        i
        for i, n in enumerate(body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "input_norm_layers" for t in n.targets
        )
    )

    class Decoder:
        input_rms_fp8_backend = "native"

    for modules, expected in (([object()], []), ([object(), Decoder()], ["native"])):
        ns = execute(
            body[start : start + 3],
            {
                "model": SimpleNamespace(modules=lambda modules=modules: iter(modules)),
                "DecoderLayer": Decoder,
            },
        )
        assert [m.input_rms_fp8_backend for m in ns["input_norm_layers"]] == expected


@pytest.mark.parametrize(
    "change",
    [
        "m2",
        "dtype",
        "layout",
        "grad",
        "cpu",
        "sm80",
        "hip",
        "norm_dtype",
        "norm_shape",
        "epsilon",
        "projection",
        "fanout",
        "checkpoint",
        "scale_device",
    ],
)
def test_metadata_fallback(production, change):
    x, norm, projection = Tensor(), Norm(), Projection()
    if change == "m2":
        x.shape = (1, 2, 5120)
    elif change == "dtype":
        x.dtype = "f32"
    elif change == "layout":
        x.contiguous = False
    elif change == "grad":
        x.requires_grad = True
    elif change == "cpu":
        x.device.type = "cpu"
    elif change == "sm80":
        production["torch"].cuda.get_device_capability = lambda d: (8, 0)
    elif change == "hip":
        production["torch"].version.hip = "test"
    elif change == "norm_dtype":
        norm.weight.dtype = "f32"
    elif change == "norm_shape":
        norm.weight.shape = (128,)
    elif change == "epsilon":
        norm.eps = float("nan")
    elif change == "projection":
        projection.names = ("gate_proj", "up_proj")
    elif change == "fanout":
        projection = Projection(True)
    elif change == "checkpoint":
        projection.weight.requires_grad = True
    elif change == "scale_device":
        projection.weight_scale_inv.device.index = 1
    assert (
        production["execution"](x, norm, projection, "triton", keep_bf16=False)[
            "selected_backend"
        ]
        == "native"
    )
    assert norm.cache_calls == 0


@pytest.mark.parametrize("keep", [False, True])
def test_complete_pair_readiness_and_errors(production, keep):
    x, norm, p = Tensor(), Norm(), Projection(keep)
    events = []
    p._prepare_input_rms_fp8 = lambda *a: events.append("prepare")
    p._forward_input_prequantized = lambda *a: (
        events.append(("consumer", a)) or ("parts",)
    )
    production["rms_quant"] = lambda *a, **k: (
        events.append(("producer", k)) or ("bf16" if keep else None, "q", "s")
    )
    assert production["project"](x, norm, p, keep_bf16=keep) == (
        "bf16" if keep else None,
        ("parts",),
    )
    assert p._input_rms_fp8_ready_device == x.device
    assert events == [
        "prepare",
        ("producer", {"keep_bf16": keep}),
        ("consumer", ("q", "s", x.shape, keep)),
    ]
    del p._input_rms_fp8_ready_device

    def failure(*args, **kwargs):
        raise RuntimeError("selected failure")

    p._forward_input_prequantized = failure
    with pytest.raises(RuntimeError, match="selected failure"):
        production["project"](x, norm, p, keep_bf16=keep)
    assert not hasattr(p, "_input_rms_fp8_ready_device")
    production["rms_quant"] = failure
    with pytest.raises(RuntimeError, match="selected failure"):
        production["project"](x, norm, p, keep_bf16=keep)
    assert not hasattr(p, "_input_rms_fp8_ready_device")
    norm.gamma.requires_grad = True
    with pytest.raises(ValueError, match="inference BF16"):
        production["project"](x, norm, p, keep_bf16=keep)


@pytest.mark.parametrize(
    "change", [None, "compact", "capacity", "offset", "dtype", "input_shape"]
)
def test_consumer_capacity_and_ownership(production, change):
    q = Tensor((1, 5120), "fp8")
    scales = Tensor((1, 40), "f32", stride=(1, 4), capacity=640)
    shape = (1, 1, 5120)
    if change == "compact":
        scales.strides = (40, 1)
    elif change == "capacity":
        scales.capacity = 628
    elif change == "offset":
        scales.offset = 1
    elif change == "dtype":
        scales.dtype = "bf16"
    elif change == "input_shape":
        shape = (1, 2, 5120)
    if change:
        with pytest.raises(ValueError):
            production["validate_prequantized"](Projection(), q, scales, shape, False)
    else:
        production["validate_prequantized"](Projection(), q, scales, shape, False)


@pytest.mark.parametrize("keep", [False, True])
def test_actual_consumer_and_cold_capture(production, keep):
    methods = [
        function("quant.py", n, "FP8MergedLinear")
        for n in ("_prepare_input_rms_fp8", "_forward_input_prequantized")
    ]
    events = []

    class Result:
        def reshape(self, *shape):
            events.append(("shape", shape))
            return self

        def split(self, sizes, dim):
            events.append(("split", sizes, dim))
            return ("outputs",)

    def gemm(*args, **kwargs):
        events.append(("gemm", args, kwargs))
        return Result()

    ns = execute(methods, dict(production, fp8_blockscale_gemm_sm90=gemm))
    p = Projection(keep)
    p._prepare_input_rms_fp8 = lambda *a: ns["_prepare_input_rms_fp8"](p, *a)
    q, scales = (
        Tensor((1, 5120), "fp8"),
        Tensor((1, 40), "f32", stride=(1, 4), capacity=640),
    )
    production["torch"].cuda.is_current_stream_capturing = lambda: True
    with pytest.raises(RuntimeError, match="Warm"):
        ns["_forward_input_prequantized"](p, q, scales, (1, 1, 5120), keep)
    assert not events and p.native_fp8_calls == 0
    p._input_rms_fp8_ready_device = q.device
    assert ns["_forward_input_prequantized"](p, q, scales, (1, 1, 5120), keep) == (
        "outputs",
    )
    assert events[0] == (
        "gemm",
        (q, p.weight),
        {
            "input_scale": scales,
            "weight_scale": p.weight_scale_inv,
            "out_dtype": "bf16",
        },
    )
    assert events[1:] == [("shape", (1, 1, p.out_features)), ("split", p.out_sizes, -1)]
    assert p.native_fp8_calls == p.kernel_calls["flashinfer"] == 1


@pytest.mark.parametrize("keep", [False, True])
def test_actual_decoder_pair_route(production, keep):
    methods = [
        function("model.py", n, "DecoderLayer")
        for n in ("input_rms_fp8_execution", "_forward_input")
    ]
    ns = execute(
        methods, dict(production, project=lambda *a, **k: ("bf16", ("parts",)))
    )
    p, x, norm = Projection(keep), Tensor(), Norm()
    layer = SimpleNamespace(
        kind="linear_attention" if keep else "full_attention",
        linear_attn=SimpleNamespace(in_proj_qkvz=p),
        self_attn=SimpleNamespace(qkv_proj=p),
        input_layernorm=norm,
        input_rms_fp8_backend="triton",
        input_rms_fp8_calls={"native": 0, "triton": 0},
    )
    layer.input_rms_fp8_execution = lambda x, **kw: ns["input_rms_fp8_execution"](
        layer, x, **kw
    )
    assert ns["_forward_input"](layer, x) == ("bf16" if keep else x, ("parts",))
    assert layer.input_rms_fp8_calls == {"native": 0, "triton": 1}
    assert ns["_forward_input"](layer, x, fresh_prefill=True) == ("native_norm", None)
    assert (
        layer.input_rms_fp8_execution(x, fresh_prefill=True)["fallback_reason"]
        == "prefill_native"
    )
    x.shape = (1, 2, 5120)
    assert ns["_forward_input"](layer, x) == ("native_norm", None)
    assert layer.input_rms_fp8_calls == {"native": 2, "triton": 1}


@pytest.mark.parametrize("keep", [False, True])
def test_actual_producer_allocations(production, keep):
    allocated, calls = [], []

    def empty(shape, *, device=None, dtype="bf16"):
        value = Tensor(
            shape, dtype, pointer=10**6 + len(allocated) * 100000, device=device
        )
        value.as_strided = lambda shape, stride: SimpleNamespace(
            shape=shape, stride=stride, owner=value
        )
        allocated.append(value)
        return value

    torch = production["torch"]
    torch.empty, torch.empty_like = empty, lambda x: empty(x.shape, device=x.device)

    class Kernel:
        def __getitem__(self, grid):
            return lambda *a, **kw: calls.append((grid, a, kw))

    ns = execute(
        [function("_triton_input_rms_fp8.py", "rms_quant")],
        dict(production, _input_rms_quant_kernel=Kernel()),
    )
    x, norm = Tensor(), Norm()
    normalized, _q, scales = ns["rms_quant"](x, norm.gamma, norm.eps, keep_bf16=keep)
    assert (normalized is not None) is keep
    assert [v.shape for v in allocated] == ([(1, 1, 5120)] if keep else []) + [
        (1, 5120),
        (40, 4),
    ]
    assert scales.owner is allocated[-1] and scales.owner.capacity == 640
    assert scales.shape == (1, 40) and scales.stride == (1, 4)
    assert calls[0][1][2] is (normalized if keep else x)
    assert calls[0][2] == {
        "BLOCK": 8192,
        "KEEP_BF16": keep,
        "num_warps": 4,
        "enable_fp_fusion": False,
    }
    assert "owner" not in vars(x)
