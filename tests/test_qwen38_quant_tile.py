"""CPU source/dispatch contracts only; FP8 conversion and CUDA are not executed."""

import ast
import hashlib
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "nanovllm/qwen38/_deepgemm_fp8.py"
TILED = ROOT / "nanovllm/qwen38/_triton_quant4.py"


def definition(path, name):
    source = path.read_text()
    node = ast.parse(source)
    for part in name.split("."):
        node = next(n for n in node.body if getattr(n, "name", None) == part)
    return source, node


def load_definition(path, name, namespace):
    _, node = definition(path, name)
    node.decorator_list = []
    # Function annotations are not part of this source-level execution.
    for item in ast.walk(node):
        if isinstance(item, ast.arg):
            item.annotation = None
        elif isinstance(item, ast.FunctionDef):
            item.returns = None
    # Execute only the selected local source definition with the CPU test stubs.
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace[node.name]


class Values(np.ndarray):
    def to(self, dtype):
        # The CPU model ends before FP8 conversion; it checks the division input.
        return self.astype(np.float32 if dtype == "unmodeled_fp8" else dtype)


def values(data):
    return np.asarray(data).view(Values)


class Pointer:
    def __init__(self, buffer, offsets=0):
        self.buffer, self.offsets = buffer, offsets

    def __add__(self, offsets):
        return Pointer(self.buffer, self.offsets + offsets)


class Language:
    float32, int64, float8e4nv = np.float32, np.int64, "unmodeled_fp8"

    def __init__(self, row, group):
        self.ids = row, group

    def program_id(self, axis):
        return values(self.ids[axis])

    @staticmethod
    def arange(start, end):
        return values(np.arange(start, end))

    @staticmethod
    def abs(data):
        return np.abs(data)

    @staticmethod
    def max(data, axis):
        return values(np.max(data, axis=axis))

    @staticmethod
    def maximum(data, floor):
        return np.maximum(data, np.float32(floor))

    @staticmethod
    def live(pointer, mask):
        offsets = np.asarray(pointer.offsets)
        live = np.broadcast_to(mask, offsets.shape)
        selected = offsets[live]
        assert np.all((selected >= 0) & (selected < pointer.buffer.size)), "out-of-bounds access"
        return offsets, live

    def load(self, pointer, mask=True, other=0):
        offsets, live = self.live(pointer, mask)
        result = np.full(offsets.shape, other, dtype=pointer.buffer.dtype)
        result[live] = pointer.buffer[offsets[live]]
        return values(result)

    def store(self, pointer, data, mask=True):
        offsets, live = self.live(pointer, mask)
        pointer.buffer[offsets[live]] = np.broadcast_to(data, offsets.shape)[live]


@pytest.mark.parametrize("m", [1, 3, 4, 5, 31, 32, 33])
@pytest.mark.parametrize("k", [128, 5120, 6144, 17408])
def test_actual_source_masks_and_independent_row_group_maxima(m, k):
    padded = (m + 3) // 4 * 4
    # Distinct row/group maxima at distinct lanes; zero and tiny groups included.
    x = np.arange(m * k, dtype=np.float32).reshape(m, k) % 113 - 56
    for row in range(m):
        for group in range(k // 128):
            x[row, group * 128 + (row + group) % 128] = 200 + row * 17 + group
    x[0, :128] = 0
    if k > 128:
        x[0, 128:256] = np.float32(1e-14)
    results = []
    for path, name, tile in ((NATIVE, "_quantize_groups", 1), (TILED, "_quantize_groups4", 4)):
        q = np.full(m * k, np.nan, np.float32)
        # Exact minimal allocation: the last column does not own padded tail rows.
        scale = np.full((k // 128 - 1) * padded + m, np.nan, np.float32)
        namespace = {}
        kernel = load_definition(path, name, namespace)
        for row in range((m + tile - 1) // tile):
            for group in range(k // 128):
                namespace["tl"] = Language(row, group)
                args = (Pointer(x.reshape(-1)), Pointer(q), Pointer(scale))
                kernel(*args, *([m] if tile == 4 else []), k, padded)
        results.append((q, scale))
    for left, right in zip(*results):
        assert np.array_equal(left.view(np.uint32), right.view(np.uint32))
    live = {group * padded + row for group in range(k // 128) for row in range(m)}
    assert results[1][1][0] == 0  # Stored scale is not the denominator floor.
    assert all(np.isnan(v) for i, v in enumerate(results[1][1]) if i not in live)


def test_native_and_tiled_gemm_consumer_are_identical():
    left, ln = definition(NATIVE, "matmul")
    right, rn = definition(TILED, "matmul")
    assert ast.get_source_segment(left, ln) == ast.get_source_segment(right, rn)


class Tensor:
    def __init__(self, shape, dtype="bf16", *, cuda=True, contiguous=True):
        self.shape, self.dtype = tuple(shape), dtype
        self.is_cuda, self._contiguous, self.device = cuda, contiguous, "synthetic"
        self.is_meta = True
        self.ndim = len(shape)

    def numel(self):
        return int(np.prod(self.shape))

    def reshape(self, *shape):
        return Tensor(shape, self.dtype)

    def contiguous(self):
        return self

    def is_contiguous(self):
        return self._contiguous


@pytest.fixture
def module_type(monkeypatch):
    torch = SimpleNamespace(bfloat16="bf16", float8_e4m3fn="fp8", float32="fp32",
                            empty=lambda *shape, **kwargs: Tensor(shape, kwargs["dtype"]))
    nn = SimpleNamespace(Module=object, Parameter=lambda data, **kwargs: data)
    path = ROOT / "nanovllm/qwen38/quant.py"
    namespace = {"os": os, "torch": torch, "nn": nn, "FP8_QUANTIZATION": {}}
    cls = load_definition(path, "FP8Linear", namespace)
    report = load_definition(path, "fp8_report", namespace)
    monkeypatch.setenv("NANOVLLM_FP8_GEMM", "flashinfer_deepgemm")
    monkeypatch.delenv("NANOVLLM_DEEPGEMM_QUANTIZATION", raising=False)
    return cls, report, namespace


@pytest.mark.parametrize("quantizer", ["native", "row4"])
@pytest.mark.parametrize("rows", [1, 31, 32, 33, 8192])
def test_constructed_selection_and_small_m_dispatch(module_type, monkeypatch, quantizer, rows):
    cls, report, namespace = module_type
    namespace["__package__"] = "nanovllm.qwen38"
    monkeypatch.setenv("NANOVLLM_DEEPGEMM_QUANTIZATION", quantizer)
    module = cls(128, 64)
    monkeypatch.setenv("NANOVLLM_DEEPGEMM_QUANTIZATION", "row4" if quantizer == "native" else "native")
    observed = []
    for name, mode in (("flashinfer.gemm", "flashinfer"),
                       ("nanovllm.qwen38._deepgemm_fp8", "native"),
                       ("nanovllm.qwen38._triton_quant4", "row4")):
        stub = ModuleType(name)
        def call(inputs, weight, scales=None, *, weight_scale=None, out_dtype=None, mode=mode):
            assert weight is module.weight
            assert (weight_scale if mode == "flashinfer" else scales) is module.weight_scale_inv
            observed.append(mode)
            return Tensor((rows, 64))
        setattr(stub, "fp8_blockscale_gemm_sm90" if mode == "flashinfer" else "matmul", call)
        monkeypatch.setitem(sys.modules, name, stub)
    result = module._matmul(Tensor((rows, 128)))
    assert result.shape == (rows, 64)
    assert observed == ["flashinfer" if rows < 32 else quantizer]
    assert module.deepgemm_quantization_calls == {
        mode: int(rows >= 32 and mode == quantizer) for mode in ("native", "row4")}
    assert report([module])["deepgemm_quantization"] == {
        "backend": quantizer, "active": True, "group_size": 128,
        "rows_per_program": 4 if quantizer == "row4" else 1,
        "scale_dtype": "float32", "scale_layout": "column_major_padded4"}


def test_default_and_invalid_or_mixed_selection(module_type, monkeypatch):
    cls, report, _ = module_type
    native = cls(128, 64)
    assert native.deepgemm_quantization == "native"
    monkeypatch.setenv("NANOVLLM_DEEPGEMM_QUANTIZATION", "typo")
    with pytest.raises(ValueError, match="DEEPGEMM_QUANTIZATION"):
        cls(128, 64)
    monkeypatch.setenv("NANOVLLM_DEEPGEMM_QUANTIZATION", "row4")
    tiled = cls(128, 64)
    with pytest.raises(ValueError, match="consistent backend"):
        report([native, tiled])
    monkeypatch.setenv("NANOVLLM_FP8_GEMM", "flashinfer")
    assert report([cls(128, 64)])["deepgemm_quantization"]["active"] is False


def test_quantize_wrapper_preserves_layout_and_fixed_tile():
    launches, allocations = [], []
    class Kernel:
        def __getitem__(self, grid):
            def call(*args, **kwargs):
                launches.append((grid, args, kwargs))
            return call
    def empty(shape, **kwargs):
        tensor = Tensor(shape, kwargs["dtype"])
        allocations.append(tensor)
        return tensor
    def empty_strided(shape, stride, **kwargs):
        tensor = empty(shape, **kwargs)
        tensor.stride = stride
        return tensor
    torch = SimpleNamespace(bfloat16="bf16", float8_e4m3fn="fp8", float32="fp32",
                            empty=empty, empty_strided=empty_strided)
    quantize = load_definition(TILED, "quantize", {"torch": torch,
        "triton": SimpleNamespace(cdiv=lambda n, d: (n + d - 1) // d), "_quantize_groups4": Kernel()})
    x = Tensor((33, 5120))
    q, s = quantize(x)
    assert q.shape == (33, 5120) and s.shape == (33, 40) and s.stride == (1, 36)
    assert launches == [((9, 40), (x, q, s, 33, 5120, 36), {"num_warps": 4})]
    for invalid in (Tensor((33, 129)), Tensor((33, 128), cuda=False),
                    Tensor((33, 128), contiguous=False), Tensor((33, 128), dtype="fp32")):
        with pytest.raises(ValueError):
            quantize(invalid)
    assert len(allocations) == 2


@pytest.mark.parametrize("mixed", [False, True])
def test_merge_preserves_bound_quantizer_and_rejects_mixed_before_mutation(module_type, monkeypatch, mixed):
    cls, report, namespace = module_type
    namespace["__package__"] = "nanovllm.qwen38"
    model_types = ModuleType("nanovllm.qwen38.model")
    for name in ("MLP", "DeltaNet", "FullAttention"):
        setattr(model_types, name, type(name, (), {}))
    monkeypatch.setitem(sys.modules, model_types.__name__, model_types)
    monkeypatch.setenv("NANOVLLM_DEEPGEMM_QUANTIZATION", "row4")
    parent = model_types.MLP()
    parent.gate_proj, parent.up_proj = cls(128, 128, device="meta"), cls(128, 128, device="meta")
    if mixed:
        parent.up_proj.deepgemm_quantization = "native"
    originals = parent.gate_proj, parent.up_proj
    monkeypatch.setenv("NANOVLLM_FP8_GEMM", "triton")
    monkeypatch.setenv("NANOVLLM_DEEPGEMM_QUANTIZATION", "native")
    path = ROOT / "nanovllm/qwen38/quant.py"
    load_definition(path, "FP8MergedLinear", namespace)
    fuse = load_definition(path, "fuse_fp8_projections", namespace)
    model = SimpleNamespace(modules=lambda: [parent])
    if mixed:
        with pytest.raises(ValueError, match="compatible"):
            fuse(model)
        assert (parent.gate_proj, parent.up_proj) == originals
        assert not hasattr(parent, "gate_up_proj")
    else:
        assert fuse(model) == 1
        merged = parent.gate_up_proj
        assert merged.deepgemm_quantization == "row4" and merged.use_deepgemm
        assert report([merged])["deepgemm_quantization"]["active"] is True


def test_mixed_loading_fails_before_copy_and_health_exposes_counter_scope():
    loader = (ROOT / "nanovllm/qwen38/loader.py").read_text()
    assert loader.index("fp8_loading_report = fp8_report") < loader.index("parameters[name].copy_")
    engine = (ROOT / "nanovllm/qwen38/engine.py").read_text()
    assert 'sum(module.deepgemm_quantization_calls[backend]' in engine
    assert "excludes_graph_replay" in engine


def test_actual_health_code_aggregates_quantizer_calls_and_adapter_origin(module_type, monkeypatch):
    cls, report, _ = module_type
    monkeypatch.setenv("NANOVLLM_DEEPGEMM_QUANTIZATION", "row4")
    module = cls(128, 64)
    module.native_fp8_calls = 7
    module.kernel_calls.update(flashinfer=3, deepgemm=4)
    module.deepgemm_quantization_calls["row4"] = 4
    torch = ModuleType("torch")
    torch.float8_e4m3fn, torch.float32 = "fp8", "fp32"
    torch.cuda = SimpleNamespace(device_count=lambda: 1, get_device_name=lambda _: "synthetic")
    quant = ModuleType("nanovllm.qwen38.quant")
    quant.FP8Linear = cls
    adapter = ModuleType("nanovllm.qwen38._triton_quant4")
    adapter.__file__ = str(TILED)
    for imported in (torch, quant, adapter):
        monkeypatch.setitem(sys.modules, imported.__name__, imported)
    namespace = {"sys": sys, "Path": Path, "hashlib": hashlib, "__package__": "nanovllm.qwen38"}
    provenance = load_definition(ROOT / "nanovllm/qwen38/engine.py", "DedicatedEngine.provenance", namespace)
    model = SimpleNamespace(named_modules=lambda: [("projection", module)])
    engine = SimpleNamespace(loader_report=report([module]), model_id="synthetic",
                             runner=SimpleNamespace(model=model))
    health = provenance(engine)
    assert health["fp8_execution"]["deepgemm_quantization_calls"] == {"native": 0, "row4": 4}
    assert health["fp8_execution"]["kernel_calls"]["flashinfer"] == 3
    assert health["module_origins"][adapter.__name__]["sha256"] == hashlib.sha256(TILED.read_bytes()).hexdigest()


@pytest.mark.parametrize("backend", ["native", "row4"])
def test_torch_loader_reports_constructed_quantizer(tmp_path, monkeypatch, backend):
    pytest.importorskip("torch")
    from test_qwen38_fp8_projections import loaded_models

    from nanovllm.qwen38.loader import load_weights

    monkeypatch.setenv("NANOVLLM_FP8_GEMM", "flashinfer_deepgemm")
    monkeypatch.setenv("NANOVLLM_DEEPGEMM_QUANTIZATION", backend)
    _, model, _, report = loaded_models(tmp_path)
    monkeypatch.setenv("NANOVLLM_DEEPGEMM_QUANTIZATION", "row4" if backend == "native" else "native")
    actual = load_weights(model, tmp_path)
    assert actual["deepgemm_quantization"] == report["deepgemm_quantization"]
    assert actual["deepgemm_quantization"]["backend"] == backend


def test_torch_mixed_quantizers_fail_before_loading_mutates_parameters(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from test_qwen38_fp8_projections import loaded_models

    from nanovllm.qwen38.loader import load_weights
    from nanovllm.qwen38.quant import FP8Linear

    monkeypatch.setenv("NANOVLLM_FP8_GEMM", "flashinfer_deepgemm")
    monkeypatch.setenv("NANOVLLM_DEEPGEMM_QUANTIZATION", "native")
    _, model, _, _ = loaded_models(tmp_path)
    next(m for m in model.modules() if isinstance(m, FP8Linear)).deepgemm_quantization = "row4"
    versions = {name: p._version for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match="consistent backend"):
        load_weights(model, tmp_path)
    assert versions == {name: p._version for name, p in model.named_parameters()}
