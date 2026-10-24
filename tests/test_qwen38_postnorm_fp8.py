"""Execute the production metadata/bridge AST without importing Torch/Triton."""

import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "nanovllm/qwen38"


def tree(name):
    return ast.parse((QWEN / name).read_text())


def function(name, method, cls=None):
    nodes = tree(name).body
    if cls:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == cls).body
    return next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == method)


def execute(nodes, namespace):
    class Imports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None

    nodes = [Imports().visit(copy.deepcopy(n)) for n in nodes]
    code = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(code, "<actual-production-ast>", "exec"), namespace)  # noqa: S102
    return namespace


class Tensor:
    def __init__(self, shape=(1, 1, 5120), dtype="bf16", pointer=10000, device=None,
                 contiguous=True, stride=None, capacity=None):
        self.shape, self.dtype, self.pointer = shape, dtype, pointer
        self.device = device or SimpleNamespace(type="cuda", index=0)
        self.requires_grad = False
        self.contiguous, self.strides = contiguous, stride
        self.capacity = capacity if capacity is not None else self.numel() * self.element_size()
        self.offset = 0

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return {"bf16": 2, "fp8": 1, "f32": 4}[self.dtype]

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return self.contiguous

    def stride(self):
        return self.strides

    def storage_offset(self):
        return self.offset

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: self.capacity)

    def __getattr__(self, name):
        raise AssertionError("Unexpected tensor operation: " + name)


class Projection:
    def __init__(self):
        self.names, self.out_sizes = ("gate_proj", "up_proj"), (17408, 17408)
        self.in_features, self.out_features, self.use_flashinfer = 5120, 34816, True
        self.weight = Tensor((34816, 5120), "fp8", 10**8)
        self.weight_scale_inv = Tensor((272, 40), "f32", 10**9)


class Device:
    def __init__(self, device):
        self.device = device

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def production():
    torch = SimpleNamespace(bfloat16="bf16", float8_e4m3fn="fp8", float32="f32",
                            version=SimpleNamespace(hip=None),
                            cuda=SimpleNamespace(get_device_capability=lambda device: (9, 0),
                                                 device=Device, is_current_stream_capturing=lambda: False))
    nodes = [n for n in tree("_postnorm_fp8.py").body if isinstance(n, ast.FunctionDef)]
    return execute(nodes, {"torch": torch, "math": math, "FP8MergedLinear": Projection})


def operands():
    return Tensor(), Tensor(pointer=30000), Projection()


def test_readonly_observer_and_native_default(production):
    r, d, p = operands()
    observed = (r, d, p.weight, p.weight_scale_inv)
    before = copy.deepcopy([vars(t) for t in observed])
    assert production["execution"](r, d, p, "triton")["selected_backend"] == "triton"
    assert production["execution"](r, d, p, "native")["fallback_reason"] == "configured_native"
    assert [vars(t) for t in observed] == before
    with pytest.raises(ValueError):
        production["execution"](r, d, p, "typo")
    assert not production["execution"](r, d, p, "triton")["execution_observed"]


@pytest.mark.parametrize("change,reason", [
    ("m2", "requires_m1_k5120"), ("cpu", "requires_nvidia_cuda"),
    ("sm80", "requires_sm90"), ("dtype", "requires_bf16_operands"),
    ("device", "device_mismatch"), ("stride", "unsupported_operand_layout"),
    ("grad", "unsupported_operand_layout"), ("alias", "residual_alias"),
    ("weights_alias", "residual_alias"), ("separate", "requires_merged_fp8_gate_up"),
    ("not_gate_up", "unsupported_projection_shape"), ("not_fi", "requires_flashinfer_small_m"),
    ("weight_dtype", "invalid_checkpoint_dtypes"), ("scale_device", "checkpoint_device_mismatch"),
    ("scale_layout", "invalid_checkpoint_layout"),
])
def test_production_fallback_guards(production, change, reason):
    r, d, p = operands()
    if change == "m2":
        r.shape = d.shape = (1, 2, 5120)
    elif change == "cpu":
        r.device.type = "cpu"
    elif change == "sm80":
        production["torch"].cuda.get_device_capability = lambda device: (8, 0)
    elif change == "dtype":
        d.dtype = "f32"
    elif change == "device":
        d.device.index = 1
    elif change == "stride":
        d.contiguous = False
    elif change == "grad":
        d.requires_grad = True
    elif change == "alias":
        d.pointer = r.pointer + 2
    elif change == "weights_alias":
        p.weight.pointer = r.pointer
    elif change == "separate":
        p = None
    elif change == "not_gate_up":
        p.names = ("q_proj", "k_proj")
    elif change == "not_fi":
        p.use_flashinfer = False
    elif change == "weight_dtype":
        p.weight.dtype = "bf16"
    elif change == "scale_device":
        p.weight_scale_inv.device.index = 1
    elif change == "scale_layout":
        p.weight_scale_inv.contiguous = False
    info = production["execution"](r, d, p, "triton")
    assert info["selected_backend"] == "native" and info["fallback_reason"] == reason


def test_producer_rejects_bad_gamma_before_mutation(production):
    r, d, p = operands()
    events = []
    p._prepare_postnorm_fp8 = lambda device: events.append("prepare")
    gamma = Tensor((5120,), pointer=50000, dtype="f32")
    norm = SimpleNamespace(_fused_weight=lambda dtype: gamma, eps=1e-6)
    production["fused_add_rms_quant"] = lambda *args: events.append("mutate")
    with pytest.raises(ValueError, match="BF16"):
        production["postnorm_gate_up"](r, d, norm, p, "triton")
    assert events == ["prepare"]


def test_actual_wrapper_selected_and_m2_native_routes(production):
    r, d, p = operands()
    gamma = Tensor((5120,), pointer=50000)
    events = []
    norm = SimpleNamespace(weight=gamma, _fused_weight=lambda dtype: gamma, eps=1e-6)
    p._prepare_postnorm_fp8 = lambda device: events.append("prepare")
    p.forward_prequantized = lambda q, s, shape: events.append(("fp8", q, s, shape)) or ("gate", "up")
    production["fused_add_rms_quant"] = lambda *a: events.append("producer") or ("q", "scales")
    result = production["postnorm_gate_up"](r, d, norm, p, "triton")
    assert result == (r, "gate", "up", "triton")
    assert p._postnorm_fp8_ready_device == r.device
    assert events == ["prepare", "producer", ("fp8", "q", "scales", r.shape)]
    r.shape = d.shape = (1, 2, 5120)
    events.clear()
    production["fused_add_rms"] = lambda *a: events.append("native_producer") or (r, "normed")
    Projection.__call__ = lambda self, x: events.append(("bf16", x)) or ("gate", "up")
    assert production["postnorm_gate_up"](r, d, norm, p, "triton") == (r, "gate", "up", "native")
    assert events == ["native_producer", ("bf16", "normed")]


@pytest.mark.parametrize("change", [None, "short_storage", "compact", "offset", "dtype"])
def test_live_scale_view_full_capacity(production, change):
    q = Tensor((1, 5120), "fp8")
    s = Tensor((1, 40), "f32", stride=(1, 4), capacity=640)
    if change == "short_storage":
        s.capacity = 628  # empty_strided's minimum is three FP32 values short.
    elif change == "compact":
        s.strides = (40, 1)
    elif change == "offset":
        s.offset = 1
    elif change == "dtype":
        s.dtype = "bf16"
    if change:
        with pytest.raises(ValueError, match="scales"):
            production["validate_prequantized"](Projection(), q, s, (1, 1, 5120))
    else:
        production["validate_prequantized"](Projection(), q, s, (1, 1, 5120))


def test_cold_capture_rejected_before_import_or_mutation(production):
    prepare = function("quant.py", "_prepare_postnorm_fp8", "FP8MergedLinear")
    events = []
    production["torch"].cuda.is_current_stream_capturing = lambda: True
    ns = execute([prepare], dict(production, fp8_blockscale_gemm_sm90=lambda *a: events.append("gemm")))
    p = Projection()
    with pytest.raises(RuntimeError, match="Warm"):
        ns["_prepare_postnorm_fp8"](p, p.weight.device)
    assert events == []
    p._postnorm_fp8_ready_device = p.weight.device
    assert ns["_prepare_postnorm_fp8"](p, p.weight.device) is ns["fp8_blockscale_gemm_sm90"]


def test_public_consumer_keeps_live_scale_and_split_contract(production):
    forward = function("quant.py", "forward_prequantized", "FP8MergedLinear")
    q, s, p = Tensor((1, 5120), "fp8"), Tensor((1, 40), "f32", stride=(1, 4), capacity=640), Projection()
    events = []
    class Result:
        def reshape(self, *shape):
            events.append(("shape", shape))
            return self
        def split(self, sizes, dim):
            events.append(("split", sizes, dim))
            return ("gate_view", "up_view")
    def gemm(*args, **kwargs):
        assert args == (q, p.weight)
        assert kwargs == {"input_scale": s, "weight_scale": p.weight_scale_inv, "out_dtype": "bf16"}
        return Result()
    p._prepare_postnorm_fp8 = lambda device: gemm
    p.kernel_calls, p.native_fp8_calls = {"flashinfer": 0}, 0
    ns = execute([forward], dict(production))
    assert ns["forward_prequantized"](p, q, s, (1, 1, 5120)) == ("gate_view", "up_view")
    assert events == [("shape", (1, 1, 34816)), ("split", (17408, 17408), -1)]
    assert p.kernel_calls == {"flashinfer": 1} and p.native_fp8_calls == 1
    assert not hasattr(p, "_postnorm_fp8_ready_device")


def test_producer_allocates_full_backing_and_returns_view(production):
    wrapper = function("_triton_rms.py", "fused_add_rms_quant")
    r, d, _ = operands()
    gamma = Tensor((5120,), pointer=50000)
    allocations, launches = [], []
    class Allocation:
        def as_strided(self, shape, stride):
            assert (shape, stride) == ((1, 40), (1, 4))
            return SimpleNamespace(base=self)
    def empty(shape, **kwargs):
        value = Allocation()
        allocations.append((shape, kwargs, value))
        return value
    class Kernel:
        def __getitem__(self, grid):
            return lambda *a, **kw: launches.append((grid, a, kw))
    production["torch"].empty = empty
    ns = execute([wrapper], dict(production, _add_rms_quant_kernel=Kernel()))
    q, s = ns["fused_add_rms_quant"](r, d, gamma, 1e-6)
    assert [x[0] for x in allocations] == [(1, 5120), (40, 4)]
    assert q is allocations[0][2] and s.base is allocations[1][2]
    assert launches == [((1,), (r, d, gamma, q, s.base, 5120, 1e-6), {"BLOCK": 8192, "num_warps": 4})]


def test_preserved_rms_math_and_bf16_quant_boundary():
    original = function("_triton_rms.py", "_add_rms_kernel")
    candidate = function("_triton_rms.py", "_add_rms_quant_kernel")
    def assignments(node):
        return {ast.unparse(n.targets[0]): ast.unparse(n.value) for n in node.body if isinstance(n, ast.Assign)}
    a, b = assignments(original), assignments(candidate)
    for name in ("h", "h_rounded", "var", "n"):
        assert a[name] == b[name]
    assert b["normalized"] == '(n * w).to(tl.bfloat16).to(tl.float32)'
    assert b["blocks"] == 'tl.reshape(normalized, (BLOCK // 128, 128))'
    assert b["scale"] == 'tl.where(amax != 0.0, tl.div_rn(448.0, amax), 1.0)'
    assert b["inverse_scale"] == 'tl.div_rn(1.0, scale)'
    assert b["magnitudes"] == 'tl.where(blocks == blocks, tl.abs(blocks), 0.0)'
    stores = [n for n in ast.walk(candidate) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'tl.store']
    assert [ast.unparse(n.args[0]) for n in stores] == ['res_ptr + cols', 'q_ptr + cols', 'scale_ptr + groups * 4']
    assert not any(isinstance(n, ast.Try) for n in ast.walk(tree('_postnorm_fp8.py')))


@pytest.mark.parametrize("requested", ["native", "triton", "typo"])
def test_actual_construction_selector(requested):
    init = function("model.py", "__init__", "MLP")
    nodes = [n for n in init.body if 'postnorm_fp8' in ast.unparse(n)]
    reads = []
    env = SimpleNamespace(get=lambda name, default: reads.append((name, default)) or requested)
    obj = SimpleNamespace()
    if requested == 'typo':
        with pytest.raises(ValueError):
            execute(nodes, {'self': obj, 'os': SimpleNamespace(environ=env)})
    else:
        execute(nodes, {'self': obj, 'os': SimpleNamespace(environ=env)})
        assert obj.postnorm_fp8_backend == requested and obj.postnorm_fp8_calls == {'native': 0, 'triton': 0}
    assert reads == [('NANOVLLM_POSTNORM_FP8', 'native')]


def test_loader_scope_and_prewrite_mixed_backend_rejection():
    load = function('loader.py', 'load_weights')
    nodes = [n for n in load.body if ast.unparse(n).startswith(('postnorm_backends =', 'if len(postnorm_backends)'))]
    report = next(n for n in ast.walk(load) if isinstance(n, ast.Assign)
                  and ast.unparse(n.targets[0]) == "report['postnorm_fp8']")
    for backends in [('native',), ('triton',), ('native', 'triton')]:
        ns = {'mlps': [SimpleNamespace(postnorm_fp8_backend=b) for b in backends], 'report': {}}
        if len(set(backends)) > 1:
            with pytest.raises(ValueError):
                execute(nodes, ns)
        else:
            execute([*nodes, report], ns)
            assert ns['report']['postnorm_fp8']['requested_backend'] == backends[0]
            assert ns['report']['postnorm_fp8']['execution_observed'] is False
    assert load.body.index(nodes[-1]) < next(i for i,n in enumerate(load.body) if ast.unparse(n).startswith('shards ='))


def test_actual_mlp_wrapper_counts_only_completed_gate_up(production):
    wrapper = copy.deepcopy(function('model.py', 'postnorm_gate_up', 'MLP'))
    wrapper.name = '_mlp_wrapper'  # Avoid shadowing the imported bridge in this AST namespace.
    r, d, p = operands()
    mlp = SimpleNamespace(gate_up_proj=p, postnorm_fp8_backend='triton',
                          postnorm_fp8_calls={'native': 0, 'triton': 0})
    seen = []
    def bridge(*args):
        seen.append(args)
        return r, 'gate', 'up', 'triton'
    ns = execute([wrapper], {'postnorm_gate_up': bridge})
    norm = object()
    assert ns['_mlp_wrapper'](mlp, r, d, norm) == (r, 'gate', 'up')
    assert seen == [(r, d, norm, p, 'triton')]
    assert mlp.postnorm_fp8_calls == {'native': 0, 'triton': 1}
    def failed(*args):
        raise RuntimeError('synthetic launch failure')
    ns = execute([wrapper], {'postnorm_gate_up': failed})
    with pytest.raises(RuntimeError, match='synthetic launch'):
        ns['_mlp_wrapper'](mlp, r, d, norm)
    assert mlp.postnorm_fp8_calls == {'native': 0, 'triton': 1}
