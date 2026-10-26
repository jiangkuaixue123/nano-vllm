"""Production metadata and dispatch controls; never import Torch or compile CUDA."""
import ast
import copy
import hashlib
import json
import os
import runpy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from test_qwen38_postnorm_fp8 import Device, Tensor, execute, function, tree

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "nanovllm/qwen38"
BUILD = runpy.run_path(str(QWEN / "_fp8_token8_build.py"))
TOKEN = execute([n for n in tree("_fp8_token8.py").body if not isinstance(n, ast.ImportFrom)], dict(BUILD))


def tensor(shape, dtype, device, *, strides=None, capacity=None):
    value = Tensor(shape, dtype, 4096, device, stride=strides, capacity=capacity)
    value._version = 0
    if strides is None:
        value.strides = (shape[1], 1)
    return value


def setup(case="gate_up"):
    device = NS(type="cuda", index=0)
    cuda = NS(current_device=lambda: 0, get_device_capability=lambda _: (9, 0),
              is_current_stream_capturing=lambda: False, device=Device,
              current_stream=lambda _: NS(cuda_stream=123))
    torch = NS(cuda=cuda, version=NS(hip=None), float8_e4m3fn="fp8", float32="f32", bfloat16="bf16")
    torch.empty = lambda shape, **kw: tensor(shape, kw["dtype"], kw["device"])
    spec = BUILD["CASES"][case]
    n, k = spec["N"], spec["K"]
    p = NS(in_features=k, out_features=n, use_flashinfer=True, fp8_decode_consumer="token8",
           _fp8_decode_consumer="token8", _token8_active=True,
           weight=tensor((n, k), "fp8", device),
           weight_scale_inv=tensor((n // 128, k // 128), "f32", device), kernel_calls={"token8": 0})
    calls = []
    binary = NS(device=0, identity=dict(source_sha256="source", binary_sha256="binary", resource_admitted=True),
                call=lambda *args: calls.append(args), resource_rows={}, compiled_resources={}, resource_accounting={})
    p._token8_binding = TOKEN["Consumer"](torch, p, case, binary)
    q = tensor((1, k), "fp8", device)
    scales = tensor((1, k // 128), "f32", device, strides=(1, 4), capacity=k // 128 * 16)
    return torch, p, binary, calls, q, scales


def model():
    layers = [NS(input_rms_fp8_backend="triton", mlp=NS(postnorm_fp8_backend="triton", silu_fp8_backend="triton"),
                 linear_attn=NS(outnorm_fp8_backend="triton", norm=NS(outnorm_backend="triton"))) for _ in range(64)]
    return NS(model=NS(layers=layers))


@pytest.mark.parametrize("case", list(BUILD["CASES"]))
def test_actual_consumer_strict_abi_and_selected_readiness(case):
    torch, p, binary, calls, q, scales = setup(case)
    binding = p._token8_binding
    p._input_rms_fp8_ready_device = p.weight.device
    with pytest.raises(RuntimeError, match="Warm"):
        binding.check(q.device, capturing=True)
    output = binding(q, p.weight, input_scale=scales, weight_scale=p.weight_scale_inv, out_dtype="bf16")
    assert output.shape == (1, p.out_features) and len(calls) == 1
    assert calls[0][:3] == ("token_n8_launch", BUILD["CASES"][case]["id"], 1)
    assert binding.ready == binding.identity and binding.binary is binary
    binding.check(q.device, capturing=True)
    for field, bad in (("strides", (40, 1)), ("capacity", 4), ("offset", 1), ("dtype", "bf16")):
        old = getattr(scales, field)
        setattr(scales, field, bad)
        with pytest.raises(RuntimeError, match="scale ABI"):
            binding(q, p.weight, input_scale=scales, weight_scale=p.weight_scale_inv, out_dtype="bf16")
        setattr(scales, field, old)
    p.weight._version += 1
    with pytest.raises(RuntimeError, match="identity"):
        binding.check(q.device, capturing=False)
    assert len(calls) == 1


def test_model_preflight_all_owners_before_activation_and_graph():
    torch, p, binary, _, q, _ = setup()
    _, last, _, _, _, _ = setup("down")
    last.weight.device = p.weight.device
    last.weight_scale_inv.device = p.weight.device
    last._token8_binding = TOKEN["Consumer"](torch, last, "down", binary)
    m = model()
    runtime = TOKEN["ModelConsumers"](torch, m, [("first", p), ("last", last)], binary)
    ids = NS(shape=(1, 1), device=q.device)
    for state, count, expected in ((NS(length=0, positions=None), 1, False),
                                   (NS(length=1, positions=None), 2, False),
                                   (NS(length=0, positions=object()), 1, True),
                                   (NS(length=8192, positions=None), 1, True)):
        ids.shape = (1, count)
        runtime.begin_forward(m, state, ids)
        assert p._token8_active is last._token8_active is expected
    p._token8_active = last._token8_active = False
    last.weight._version += 1
    with pytest.raises(RuntimeError, match="identity"):
        runtime.begin_forward(m, NS(length=1, positions=None), ids)
    assert not p._token8_active and not last._token8_active
    last.weight._version -= 1
    torch.cuda.is_current_stream_capturing = lambda: True
    with pytest.raises(RuntimeError, match="Warm"):
        runtime.begin_forward(m, NS(length=1, positions=None), ids)
    m.model.layers[-1].mlp.silu_fp8_backend = "native"
    with pytest.raises(RuntimeError, match="producers"):
        runtime.begin_forward(m, NS(length=1, positions=None), ids)


def test_owner_route_excludes_same_shape_fa_output_and_has_no_fallback():
    torch, p, _, calls, _, _ = setup("gdn_output")
    assert TOKEN["consumer_for"](p, p.weight.device, "gdn_outnorm") is p._token8_binding
    with pytest.raises(RuntimeError, match="owner/entry"):
        TOKEN["consumer_for"](p, p.weight.device, "input_rms")
    del p._token8_binding
    with pytest.raises(RuntimeError, match="owner/entry"):
        TOKEN["consumer_for"](p, p.weight.device, "gdn_outnorm")
    assert calls == []


@pytest.mark.parametrize("cls,method,route", [("FP8Linear", "_prepare_silu_fp8", "silu"),
    ("FP8Linear", "_prepare_gdn_outnorm_fp8", "gdn_outnorm"),
    ("FP8MergedLinear", "_prepare_postnorm_fp8", "postnorm"),
    ("FP8MergedLinear", "_prepare_input_rms_fp8", "input_rms")])
def test_actual_prepared_dispatch_keeps_inactive_native(cls, method, route):
    torch, p, _, _, _, _ = setup()
    native, selected = object(), object()
    checks = []
    ns = execute([function("quant.py", method, cls)], dict(torch=torch,
        projection_reason=lambda *a, **k: None, consumer_for=lambda *a: checks.append(a[-1]) or selected,
        fp8_blockscale_gemm_sm90=native))
    extra = [True] if route == "input_rms" else []
    call = lambda: ns[method](p, p.weight.device, *extra)
    assert call() is selected and checks == [route]
    p._token8_active = False
    assert call() is native
    p._fp8_decode_consumer = "native"
    checks.clear()
    assert call() is native and checks == []


def test_immutable_selection_and_fusion_preserves_originals(monkeypatch):
    class Module:
        pass
    torch = NS(float8_e4m3fn="fp8", float32="f32", Tensor=object,
               empty=lambda *a, **k: NS(is_meta=True))
    nn = NS(Module=Module, Parameter=lambda value, **kw: value)
    nodes = [n for n in tree("quant.py").body if isinstance(n, ast.ClassDef) and n.name in ("FP8Linear", "FP8MergedLinear")]
    for node in nodes:
        node.body = [n for n in node.body if isinstance(n, ast.Assign) or
                     isinstance(n, ast.FunctionDef) and n.name in ("__init__", "fp8_decode_consumer")]
    ns = execute(nodes, dict(torch=torch, nn=nn, os=os))
    first = ns["FP8Linear"](128, 128, decode_consumer="token8")
    with pytest.raises(AttributeError):
        first.fp8_decode_consumer = "native"
    class MLP:
        pass
    parent = MLP()
    parent.gate_proj = first
    parent.up_proj = ns["FP8Linear"](128, 128, decode_consumer="token8")
    m = NS(modules=lambda: [parent])
    fusion = execute([function("quant.py", "fuse_fp8_projections")], dict(ns, MLP=MLP, FullAttention=type("FA", (), {}), DeltaNet=type("GDN", (), {})))["fuse_fp8_projections"]
    monkeypatch.setenv("NANOVLLM_FP8_DECODE_CONSUMER", "native")
    assert fusion(m) == 1 and parent.gate_up_proj.fp8_decode_consumer == "token8"
    parent.gate_proj, parent.up_proj = first, ns["FP8Linear"](128, 128, decode_consumer="native")
    with pytest.raises(ValueError, match="compatible"):
        fusion(m)


def test_source_packaging_identity_and_admission_order():
    identity = BUILD["source_identity"]()
    assert len(identity) == 64
    source = BUILD["SOURCE"]
    provenance = json.loads((source / "provenance.json").read_text())
    for name, record in provenance["vendor_files"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == record["sha256"]
    line = next(line for line in (ROOT / "pyproject.toml").read_text().splitlines()
                if line.startswith('"nanovllm.qwen38" ='))
    patterns = ast.literal_eval(line.split("=", 1)[1].strip())
    covered = {p for pattern in patterns for p in QWEN.glob(pattern)}
    assert all(p in covered for p in source.rglob("*") if p.is_file())
    forward = function("model.py", "forward", "Qwen38ForCausalLM")
    text = ast.unparse(forward)
    assert text.index("token8.begin_forward") < text.index("pending =")
    load = ast.unparse(function("loader.py", "load_checkpoint"))
    assert load.index("load_weights(") < load.index("prepare_model(") < load.index("return (model, report)")
    assert "token8" not in ast.unparse(function("quant.py", "_matmul", "FP8Linear"))
    assert 'self.model = model' in (QWEN / "graph.py").read_text()
    assert "fp8_decode_tile64" not in (source / "NOTICE.txt").read_text()


def test_actual_reporting_includes_successful_token8_calls():
    # Execute the changed report block with empty producer inventories; avoid optional imports.
    fn = function("engine.py", "provenance", "DedicatedEngine")
    block = next(n for n in fn.body if isinstance(n, ast.If) and "self.loader_report['dtype'] == 'fp8'" == ast.unparse(n.test))
    class Projection:
        pass
    class Other:
        pass
    for selected in (False, True):
        p = Projection()
        p.native_fp8_calls = 256
        p.kernel_calls = dict(flashinfer=16 if selected else 256, triton=0, deepgemm=0)
        if selected:
            p.kernel_calls["token8"] = 240
        p.deepgemm_quantization_calls = dict(native=0, row4=0)
        p.weight, p.weight_scale_inv = NS(dtype="fp8"), NS(dtype="f32")
        m = NS(named_modules=lambda: [("projection", p)])
        if selected:
            m._fp8_token8_runtime = NS(snapshot=lambda: dict(owners=240))
        self = NS(runner=NS(model=m), loader_report=dict(dtype="fp8", backend="test", kernel_name="test"))
        ns = execute([block], dict(self=self, torch=NS(float8_e4m3fn="fp8", float32="f32"), FP8Linear=Projection,
             MLP=Other, DecoderLayer=Other, DeltaNet=Other, FullAttention=Other))
        result = ns["execution"]
        assert sum(result["fp8_execution"]["kernel_calls"].values()) == result["fp8_execution"]["total_calls"] == 256
        assert ("token8" in result["fp8_execution"]["kernel_calls"]) is selected
        for name in ("input_rms", "postnorm", "silu", "gdn_outnorm"):
            assert result[name + "_fp8_execution"]["consumer"] == ("nanovllm.fp8_token8_prequantized" if selected else "flashinfer.fp8_blockscale_gemm_sm90_prequantized")


def test_eager_build_owns_sources_and_rejects_dependency_drift(tmp_path, monkeypatch):
    """Run the actual builder with filesystem fixtures and a compiler double."""
    globals_ = BUILD["build"].__globals__
    cuda = tmp_path / "cuda"
    for name in ("bin/nvcc", "bin/ptxas", "bin/fatbinary", "bin/nvlink", "bin/cudafe++", "bin/cuobjdump",
                 "nvvm/bin/cicc", "nvvm/libdevice/libdevice.bc"):
        path = cuda / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    host = tmp_path / "cxx"
    host.write_text("host")
    source = tmp_path / "packaged"
    source.mkdir()
    (source / "launch.cu").write_text("source")
    header = tmp_path / "header.h"
    header.write_text("dependency")
    monkeypatch.setenv("CUDA_HOME", str(cuda))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    for key in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CPATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(globals_, "SOURCE", source)
    monkeypatch.setitem(globals_, "source_identity", lambda: "a" * 64)
    monkeypatch.setattr(globals_["shutil"], "which", lambda _: str(host))
    commands = []
    drift = False

    def compiler(command, **kwargs):
        commands.append(command)
        output = Path(command[command.index("-o") + 1])
        assert Path(kwargs["env"]["TMPDIR"]) == kwargs["cwd"] == output.parent
        assert "--gpu-architecture=compute_90a" in command and "--gpu-code=sm_90a,compute_90a" in command
        if "-M" in command:
            output.write_text("object: " + str(header))
        else:
            assert "-shared" in command and "--cudart=shared" in command
            output.write_bytes(b"mock binary")
            if drift:
                header.write_text("changed")
        return NS(stdout=b"", stderr=b"")

    monkeypatch.setattr(globals_["subprocess"], "run", compiler)
    work, library, _, identity = BUILD["build"]()
    try:
        assert library.is_file() and len(commands) == 2
        assert identity["disk_cache_reused"] is False and identity["dependency_count"] == 1
        assert identity["binary_sha256"] == hashlib.sha256(b"mock binary").hexdigest()
    finally:
        work.cleanup()
    drift = True
    with pytest.raises(RuntimeError, match="dependency drift"):
        BUILD["build"]()
    assert list((tmp_path / "cache/nanovllm/fp8_token8").iterdir()) == []
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "--gpu-architecture=compute_90")
    with pytest.raises(RuntimeError, match="ambient compiler"):
        BUILD["build"]()
    assert len(commands) == 4
