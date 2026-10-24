"""Focused host metadata/control-flow checks; no CUDA arithmetic qualification."""

import ast
import copy
import math
from types import SimpleNamespace

import pytest
from test_qwen38_postnorm_fp8 import Device, execute, function, tree
from test_qwen38_postnorm_fp8 import Tensor as BaseTensor


class Tensor(BaseTensor):
    def __init__(self, shape=(1, 1, 48, 128), **kwargs):
        kwargs.setdefault("stride", tuple(math.prod(shape[i + 1:]) for i in range(len(shape))))
        super().__init__(shape=shape, **kwargs)


class Projection:
    def __init__(self):
        self.in_features, self.out_features, self.use_flashinfer = 6144, 5120, True
        self.weight = Tensor((5120, 6144), dtype="fp8", pointer=10**8)
        self.weight_scale_inv = Tensor((40, 48), dtype="f32", pointer=10**9)
        self.kernel_calls, self.native_fp8_calls = {"flashinfer": 0}, 0


def norm_module():
    weight = Tensor((128,), pointer=50000)
    weight.requires_grad = True  # The real nn.Parameter is detached by _fused_weight.
    cached = Tensor((128,), pointer=50000)
    return SimpleNamespace(outnorm_backend="triton", weight=weight, eps=1e-6,
                           _fused_weight=lambda dtype: cached)


@pytest.fixture
def production():
    torch = SimpleNamespace(bfloat16="bf16", float8_e4m3fn="fp8", float32="f32",
                            version=SimpleNamespace(hip=None),
                            cuda=SimpleNamespace(get_device_capability=lambda device: (9, 0),
                                                 device=Device, is_current_stream_capturing=lambda: False))
    nodes = [n for n in tree("_gdn_outnorm_fp8.py").body if isinstance(n, ast.FunctionDef)]
    return execute(nodes, {"torch": torch, "math": math, "FP8Linear": Projection})


def test_metadata_observer_keeps_merged_gate_view_and_does_not_prepare_cache(production):
    x, gate, norm, projection = Tensor(), Tensor(contiguous=False, pointer=80000), norm_module(), Projection()
    gate.strides = (16384, 16384, 128, 1)
    gate.offset = 10240
    norm._fused_weight = lambda dtype: pytest.fail("metadata query must not prepare cache")
    before = copy.deepcopy([vars(t) for t in (x, gate, norm.weight, projection.weight)])
    info = production["execution"](x, gate, norm, projection, "triton")
    assert info["selected_backend"] == "triton" and info["execution_observed"] is False
    assert [vars(t) for t in (x, gate, norm.weight, projection.weight)] == before
    assert production["execution"](x, gate, norm, projection, "native")["fallback_reason"] == "configured_native"
    with pytest.raises(ValueError):
        production["execution"](x, gate, norm, projection, "typo")


@pytest.mark.parametrize("change,reason", [
    ("m2", "requires_m1_48x128"), ("heads", "requires_m1_48x128"),
    ("cpu", "requires_nvidia_cuda"), ("hip", "requires_nvidia_cuda"),
    ("sm80", "requires_sm90"), ("dtype", "requires_bf16_operands"),
    ("device", "device_mismatch"), ("stride", "unsupported_operand_layout"),
    ("noncanonical", "unsupported_operand_layout"), ("unaligned", "unsupported_operand_layout"),
    ("grad", "unsupported_operand_layout"), ("norm_backend", "requires_triton_gdn_outnorm"),
    ("norm_dtype", "requires_bf16_norm_weight"), ("norm_layout", "requires_bf16_norm_weight"),
    ("epsilon", "requires_positive_epsilon"), ("plain", "requires_plain_fp8_output"),
    ("shape", "unsupported_projection_shape"), ("backend", "requires_flashinfer_small_m"),
    ("weight_dtype", "invalid_checkpoint_dtypes"), ("weight_device", "checkpoint_device_mismatch"),
    ("weight_layout", "invalid_checkpoint_layout"),
])
def test_unsupported_metadata_selects_native_before_launch(production, change, reason):
    x, gate, norm, p = Tensor(), Tensor(pointer=80000), norm_module(), Projection()
    if change == "m2":
        x.shape = gate.shape = (1, 2, 48, 128)
    elif change == "heads":
        x.shape = gate.shape = (1, 1, 16, 128)
    elif change == "cpu":
        x.device.type = "cpu"
    elif change == "hip":
        production["torch"].version.hip = "test"
    elif change == "sm80":
        production["torch"].cuda.get_device_capability = lambda device: (8, 0)
    elif change == "dtype":
        gate.dtype = "f32"
    elif change == "device":
        gate.device.index = 1
    elif change == "stride":
        gate.strides = (6144, 6144, 128, -1)
    elif change == "noncanonical":
        x.strides = (6144, 1, 128, 1)
    elif change == "unaligned":
        x.pointer += 2
    elif change == "grad":
        x.requires_grad = True
    elif change == "norm_backend":
        norm.outnorm_backend = "native"
    elif change == "norm_dtype":
        norm.weight.dtype = "f32"
    elif change == "norm_layout":
        norm.weight.contiguous = False
    elif change == "epsilon":
        norm.eps = float("nan")
    elif change == "plain":
        p = None
    elif change == "shape":
        p.in_features = 5120
    elif change == "backend":
        p.use_flashinfer = False
    elif change == "weight_dtype":
        p.weight.dtype = "bf16"
    elif change == "weight_device":
        p.weight.device.index = 1
    elif change == "weight_layout":
        p.weight_scale_inv.contiguous = False
    info = production["execution"](x, gate, norm, p, "triton")
    assert (info["selected_backend"], info["fallback_reason"]) == ("native", reason)


@pytest.mark.parametrize("change", [None, "short", "compact", "offset", "dtype"])
def test_prequantized_consumer_requires_full_scale_backing(production, change):
    q = Tensor((1, 6144), dtype="fp8")
    scales = Tensor((1, 48), dtype="f32", stride=(1, 4), capacity=768)
    if change == "short":
        scales.capacity -= 12
    elif change == "compact":
        scales.strides = (48, 1)
    elif change == "offset":
        scales.offset = 1
    elif change == "dtype":
        scales.dtype = "bf16"
    if change:
        with pytest.raises(ValueError, match="scales"):
            production["validate_prequantized"](Projection(), q, scales, (1, 1, 6144))
    else:
        production["validate_prequantized"](Projection(), q, scales, (1, 1, 6144))


def test_cold_capture_and_complete_bridge_readiness(production):
    method = function("quant.py", "_prepare_gdn_outnorm_fp8", "FP8Linear")
    ns = execute([method], dict(production, fp8_blockscale_gemm_sm90=object()))
    p, x, gate, norm = Projection(), Tensor(), Tensor(), norm_module()
    production["torch"].cuda.is_current_stream_capturing = lambda: True
    with pytest.raises(RuntimeError, match="Warm"):
        ns["_prepare_gdn_outnorm_fp8"](p, x.device)
    production["torch"].cuda.is_current_stream_capturing = lambda: False
    events = []
    p._prepare_gdn_outnorm_fp8 = lambda device: events.append("prepare")
    production["gated_rms_quant"] = lambda *args: events.append("producer") or ("q", "s")
    def fail(*args):
        raise RuntimeError("synthetic FI failure")
    p._forward_gdn_outnorm_prequantized = fail
    with pytest.raises(RuntimeError, match="synthetic FI"):
        production["outnorm_fp8_output"](x, gate, norm, p)
    assert not hasattr(p, "_gdn_outnorm_fp8_ready_device")
    p._forward_gdn_outnorm_prequantized = lambda *args: events.append("consumer") or "output"
    assert production["outnorm_fp8_output"](x, gate, norm, p) == "output"
    assert p._gdn_outnorm_fp8_ready_device == x.device
    assert events == ["prepare", "producer", "prepare", "producer", "consumer"]
    production["torch"].cuda.is_current_stream_capturing = lambda: True
    assert ns["_prepare_gdn_outnorm_fp8"](p, x.device) is ns["fp8_blockscale_gemm_sm90"]


def test_producer_requires_detached_actual_cached_weight(production):
    norm = norm_module()
    with pytest.raises(ValueError, match="norm_weight"):
        production["validate_producer"](Tensor(), Tensor(), norm.weight, norm.eps)
    production["validate_producer"](Tensor(), Tensor(), norm._fused_weight("bf16"), norm.eps)


def test_selected_failure_has_no_native_retry_and_counters_count_successes(production):
    method = function("model.py", "_forward_outnorm", "DeltaNet")
    events = []
    norm = lambda *args: events.append("norm") or SimpleNamespace(reshape=lambda *shape: shape)
    module = SimpleNamespace(outnorm_fp8_backend="triton", norm=norm, value_dim=6144,
                             outnorm_fp8_calls={"native": 0, "triton": 0},
                             outnorm_fp8_execution=lambda *a: {"selected_backend": "triton"},
                             out_proj=lambda *a: events.append("projection") or a)
    def fail(*args):
        raise RuntimeError("synthetic producer failure")
    ns = execute([method], dict(production, outnorm_fp8_output=fail))
    with pytest.raises(RuntimeError, match="synthetic producer"):
        ns["_forward_outnorm"](module, Tensor(), Tensor())
    assert events == [] and module.outnorm_fp8_calls == {"native": 0, "triton": 0}
    ns["outnorm_fp8_output"] = lambda *args: "candidate"
    assert ns["_forward_outnorm"](module, Tensor(), Tensor()) == "candidate"
    assert module.outnorm_fp8_calls == {"native": 0, "triton": 1}
    module.outnorm_fp8_execution = lambda *a: {"selected_backend": "native"}
    assert ns["_forward_outnorm"](module, Tensor(), Tensor()) == ((1, 1, 6144),)
    assert events == ["norm", "projection"]
    assert module.outnorm_fp8_calls == {"native": 1, "triton": 1}


def test_prequantized_consumer_preserves_original_fi_objects_and_counts(production):
    method = function("quant.py", "_forward_gdn_outnorm_prequantized", "FP8Linear")
    q, s, p = Tensor((1, 6144), dtype="fp8"), Tensor((1, 48), dtype="f32", stride=(1, 4), capacity=768), Projection()
    def gemm(values, weight, **kwargs):
        assert values is q and weight is p.weight
        assert kwargs == {"input_scale": s, "weight_scale": p.weight_scale_inv, "out_dtype": "bf16"}
        return SimpleNamespace(reshape=lambda *shape: shape)
    p._prepare_gdn_outnorm_fp8 = lambda device: gemm
    ns = execute([method], dict(production))
    assert ns["_forward_gdn_outnorm_prequantized"](p, q, s, (1, 1, 6144)) == (1, 1, 5120)
    assert p.kernel_calls == {"flashinfer": 1} and p.native_fp8_calls == 1
    assert not hasattr(p, "_gdn_outnorm_fp8_ready_device")


@pytest.mark.parametrize("requested", ["native", "triton", "typo"])
def test_selector_is_default_off_and_construction_bound(requested):
    init = function("model.py", "__init__", "DeltaNet")
    nodes = [n for n in init.body if "outnorm_fp8" in ast.unparse(n)]
    reads, obj = [], SimpleNamespace()
    env = SimpleNamespace(get=lambda name, default: reads.append((name, default)) or requested)
    if requested == "typo":
        with pytest.raises(ValueError):
            execute(nodes, {"self": obj, "os": SimpleNamespace(environ=env)})
    else:
        execute(nodes, {"self": obj, "os": SimpleNamespace(environ=env)})
        assert obj.outnorm_fp8_backend == requested and obj.outnorm_fp8_calls == {"native": 0, "triton": 0}
    assert reads == [("NANOVLLM_GDN_OUTNORM_FP8", "native")]


def test_loader_mixed_selector_guard_precedes_parameter_loading():
    load = function("loader.py", "load_weights")
    nodes = [n for n in load.body if ast.unparse(n).startswith(("outnorm_fp8_backends =", "if len(outnorm_fp8_backends)"))]
    ns = {"gdn_modules": [SimpleNamespace(outnorm_fp8_backend=b) for b in ("native", "triton")]}
    with pytest.raises(ValueError, match="Mixed GDN output norm FP8"):
        execute(nodes, ns)
    assert load.body.index(nodes[-1]) < next(i for i, n in enumerate(load.body) if ast.unparse(n).startswith("shards ="))


def test_kernel_retains_active_reduction_rounding_and_h2_quantization_ast():
    original = function("_triton_gdn_outnorm.py", "_outnorm_kernel")
    fused = function("_triton_gdn_outnorm_fp8.py", "_outnorm_quant_kernel")
    def through_inv(nodes):
        end = next(i for i, n in enumerate(nodes) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "inv")
        return ast.dump(ast.Module(body=nodes[:end + 1], type_ignores=[]), include_attributes=False)
    assert through_inv(original.body) == through_inv(fused.body)
    old_stages = {ast.unparse(n.targets[0]): ast.dump(n.value, include_attributes=False)
                  for n in ast.walk(original) if isinstance(n, ast.Assign)}
    new_stages = {ast.unparse(n.targets[0]): ast.dump(n.value, include_attributes=False)
                  for n in ast.walk(fused) if isinstance(n, ast.Assign)}
    for name in ("norm", "silu", "rounded_gate", "result"):
        assert old_stages[name] == new_stages[name]
    h2 = function("_triton_rms.py", "_add_rms_quant_kernel")
    quant = {ast.unparse(n.targets[0]): ast.dump(n.value, include_attributes=False)
             for n in ast.walk(h2) if isinstance(n, ast.Assign)}
    for name in ("magnitudes", "amax", "scale", "inverse_scale", "quantized"):
        assert quant[name] == new_stages[name]


def test_producer_allocates_complete_independent_outputs_and_fixed_head_grid(production):
    wrapper = function("_triton_gdn_outnorm_fp8.py", "gated_rms_quant")
    allocated, launches = [], []
    def empty(shape, *, dtype, device):
        tensor = Tensor(shape, dtype=dtype, device=device)
        if shape == (48, 4):
            tensor.as_strided = lambda shape, stride: Tensor(shape, dtype=dtype, device=device,
                                                           stride=stride, capacity=tensor.capacity)
        allocated.append(tensor)
        return tensor
    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, args, kwargs))
            return launch
    production["torch"].empty = empty
    ns = execute([wrapper], dict(production, _outnorm_quant_kernel=Kernel()))
    x, gate, weight = Tensor(), Tensor(stride=(16384, 16384, 128, 1)), Tensor((128,))
    q1, s1 = ns["gated_rms_quant"](x, gate, weight, 1e-6)
    q2, s2 = ns["gated_rms_quant"](x, gate, weight, 1e-6)
    assert q1 is not q2 and s1 is not s2 and len(allocated) == 4
    assert s1.shape == (1, 48) and s1.stride() == (1, 4) and s1.capacity == 768
    for grid, args, options in launches:
        assert grid == (12,) and args[:3] == (x, gate, weight)
        assert args[5:] == (48, 48, 16384, 128, 1, 1e-6)
        assert options == {"num_warps": 4, "enable_fp_fusion": False}


def test_loader_report_exposes_constructed_selector_and_scale_layout():
    load = function("loader.py", "load_weights")
    node = next(n for n in ast.walk(load) if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "report['gdn_outnorm_fp8']")
    ns = execute([node], {"report": {}, "outnorm_fp8_backends": {"triton"}, "gdn_modules": [None] * 48})
    assert ns["report"]["gdn_outnorm_fp8"] == {
        "requested_backend": "triton", "layers": 48, "scope": "m1_gdn_outnorm_fp8_output",
        "norm_intermediate_dtype": "bfloat16", "silu_intermediate_dtype": "bfloat16",
        "product_intermediate_dtype": "bfloat16", "quantizer": "fi_block128_fp32_scales",
        "scale_backing": [48, 4], "fallback": "native_for_unsupported_operand_metadata",
        "execution_observed": False,
    }


def test_service_counters_expose_successful_host_calls_without_replay_claim():
    provenance = function("engine.py", "provenance", "DedicatedEngine")
    node = next(n for n in ast.walk(provenance) if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "execution['gdn_outnorm_fp8_execution']")
    gdn = {"layer0": SimpleNamespace(outnorm_fp8_calls={"native": 2, "triton": 3}),
           "layer1": SimpleNamespace(outnorm_fp8_calls={"native": 2, "triton": 3})}
    ns = execute([node], {"gdn": gdn, "execution": {}})
    info = ns["execution"]["gdn_outnorm_fp8_execution"]
    assert info["calls"] == {"native": 4, "triton": 6}
    assert info["per_module_calls"] == {name: m.outnorm_fp8_calls for name, m in gdn.items()}
    assert info["call_counter_scope"] == "python_producer_output_calls_excludes_graph_replay"
    assert info["candidate_kernel"] == "_outnorm_quant_kernel"
