"""Actual dispatch and consumer control flow using metadata doubles, without Torch."""
# ruff: noqa: S102 -- compile only this checkout's selected consumer AST.
import ast
import importlib.util
import os
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODEL = ast.parse((ROOT / "nanovllm/qwen38/model.py").read_text())
ATTENTION = next(n for n in MODEL.body if isinstance(n, ast.ClassDef) and n.name == "FullAttention")


class Tensor:
    def __init__(self, shape, device, dtype="bf16", contiguous=True, grad=False):
        self.shape, self.device, self.dtype = shape, device, dtype
        self.contiguous, self.requires_grad = contiguous, grad
        self.writes = 0

    def is_contiguous(self):
        return self.contiguous

    def reshape(self, *shape):
        return self

    def index_copy_(self, *args):
        self.writes += 1

    def item(self):
        raise AssertionError("Metadata dispatch must not read device data")


class GatedAttentionContracts(unittest.TestCase):
    def setUp(self):
        self.device = SimpleNamespace(type="cuda", index=0)
        self.capture = False
        self.capability = (9, 0)
        self.torch = SimpleNamespace(
            bfloat16="bf16", int64="int64", version=SimpleNamespace(hip=None),
            cuda=SimpleNamespace(
                get_device_capability=lambda device: self.capability,
                is_current_stream_capturing=lambda: self.capture,
                device=lambda device: nullcontext(),
            ),
        )
        spec = importlib.util.spec_from_file_location(
            "gated_test.bridge", ROOT / "nanovllm/qwen38/_gated_attention_decode.py",
        )
        self.bridge = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": self.torch}):
            spec.loader.exec_module(self.bridge)
        self.q = Tensor((1, 24, 1, 256), self.device)
        self.k = Tensor((1, 4, 10240, 256), self.device)
        self.v = Tensor(self.k.shape, self.device)
        self.gate = Tensor((1, 1, 6144), self.device)
        self.pos = Tensor((1,), self.device, "int64")

    def route(self, backend="triton_split", **kwargs):
        return self.bridge.execution(
            self.q, self.k, self.v, self.gate, self.pos, backend,
            **({"capacity": 10240, "fresh_prefill": False} | kwargs),
        )

    def test_supported_selection_is_metadata_only(self):
        route = self.route()
        self.assertEqual(route["selected_backend"], "triton_split")
        self.assertFalse(route["execution_observed"])
        self.assertEqual(self.k.writes + self.v.writes, 0)

    def test_default_fresh_and_capacity_fallback(self):
        for kwargs, reason in [({"backend": "native"}, "configured_native"),
                               ({"fresh_prefill": True}, "fresh_prefill"),
                               ({"capacity": 8448}, "requires_fixed_capacity_10240")]:
            with self.subTest(kwargs=kwargs):
                self.assertEqual(self.route(**kwargs)["fallback_reason"], reason)
        self.k = None
        self.assertEqual(self.route()["fallback_reason"], "requires_fixed_capacity_10240")

    def test_metadata_fallback_matrix(self):
        cases = [
            ("q", "shape", (1, 24, 2, 256), "unsupported_shape"),
            ("k", "shape", (1, 4, 10239, 256), "unsupported_shape"),
            ("v", "shape", (1, 4, 10240, 128), "unsupported_shape"),
            ("gate", "shape", (1, 6144), "unsupported_shape"),
            ("pos", "shape", (2,), "unsupported_shape"),
            ("q", "dtype", "fp32", "unsupported_dtype"),
            ("k", "dtype", "fp16", "unsupported_dtype"),
            ("pos", "dtype", "int32", "unsupported_dtype"),
            ("gate", "contiguous", False, "unsupported_layout_or_grad"),
            ("k", "requires_grad", True, "unsupported_layout_or_grad"),
            ("v", "device", SimpleNamespace(type="cuda", index=1), "device_mismatch"),
        ]
        for name, attr, value, reason in cases:
            tensor = getattr(self, name)
            original = getattr(tensor, attr)
            with self.subTest(name=name, attr=attr):
                setattr(tensor, attr, value)
                self.assertEqual(self.route()["fallback_reason"], reason)
                setattr(tensor, attr, original)

    def test_architecture_fallback(self):
        self.capability = (10, 0)
        self.assertEqual(self.route()["fallback_reason"], "requires_sm90")
        self.torch.version.hip = "synthetic"
        self.assertEqual(self.route()["fallback_reason"], "requires_nvidia_cuda")
        self.torch.version.hip = None
        self.q.device = SimpleNamespace(type="cpu")
        self.assertEqual(self.route()["fallback_reason"], "requires_nvidia_cuda")

    def test_unknown_backend_rejected(self):
        for backend in ("", "triton", "fa2", "TRITON_SPLIT"):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                self.route(backend)

    def test_constructed_option_default_and_validation(self):
        init = next(n for n in ATTENTION.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        body = []
        for node in init.body[1:]:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                break
            body.append(node)
        code = compile(ast.Module(body=body, type_ignores=[]), "actual_constructor", "exec")
        for option in (None, "native", "triton_split", "unknown"):
            env = {} if option is None else {"NANOVLLM_GATED_DECODE_ATTENTION": option}
            with patch.dict(os.environ, env, clear=True):
                module = SimpleNamespace()
                if option == "unknown":
                    with self.assertRaises(ValueError):
                        exec(code, {"os": os, "self": module, "config": None})
                else:
                    exec(code, {"os": os, "self": module, "config": None})
                    self.assertEqual(module.decode_attention, option or "native")

    def consumer(self, kernel, *, backend="triton_split"):
        # Execute the unchanged actual selection -> KV write -> selected return
        # section. This excludes projection math, not its ordering around writes.
        forward = next(n for n in ATTENTION.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        body, copying = [], False
        for node in forward.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "decode_kernel":
                copying = True
            if copying:
                body.append(node)
            if isinstance(node, ast.AugAssign) and "gated_decode_calls" in ast.unparse(node):
                break
        template = ast.parse("def dispatch(self, q, k, v, gate, state, positions, fresh_prefill):\n    pass\n")
        template.body[0].body = body
        namespace = {"__package__": "gated_test"}
        exec(compile(ast.fix_missing_locations(template), "actual_consumer", "exec"), namespace)
        module = SimpleNamespace(
            decode_attention=backend, gated_decode_calls={"native": 0, "triton_split": 0},
            gated_decode_last_route=None, o_proj=lambda value: value,
        )
        state = SimpleNamespace(capacity=10240, key=self.k, value=self.v)
        package = ModuleType("gated_test")
        package.__path__ = []
        modules = {
            "gated_test": package, "gated_test._gated_attention_decode": self.bridge,
            "gated_test._triton_gated_attention_decode": SimpleNamespace(gated_attention_decode=kernel),
        }

        def call():
            with patch.dict(sys.modules, modules):
                return namespace["dispatch"](
                    module, self.q, self.k, self.v, self.gate, state, self.pos, False,
                )
        return module, call

    def test_cold_capture_rejects_before_kv_write(self):
        self.capture = True
        _, call = self.consumer(lambda *args: None)
        with self.assertRaisesRegex(RuntimeError, "Warm gated"):
            call()
        self.assertEqual((self.k.writes, self.v.writes), (0, 0))

    def test_success_marks_readiness_and_observed_route(self):
        sentinel = object()
        module, call = self.consumer(lambda *args: sentinel)
        self.assertIs(call(), sentinel)
        self.assertEqual((self.k.writes, self.v.writes), (1, 1))
        self.assertTrue(module.gated_decode_last_route["execution_observed"])
        self.assertEqual(module.gated_decode_calls, {"native": 0, "triton_split": 1})
        self.assertEqual(module._gated_decode_ready_device, self.device)
        self.capture = True
        self.assertIs(call(), sentinel)

    def test_selected_failure_propagates_without_native_retry(self):
        def fail(*args):
            raise RuntimeError("selected launch failure")
        module, call = self.consumer(fail)
        with self.assertRaisesRegex(RuntimeError, "selected launch failure"):
            call()
        self.assertEqual(module.gated_decode_calls, {"native": 0, "triton_split": 0})
        self.assertFalse(module.gated_decode_last_route["execution_observed"])
        self.assertFalse(hasattr(module, "_gated_decode_ready_device"))

    def test_unsupported_shape_retains_native_after_original_writes(self):
        self.q.shape = (1, 24, 2, 256)
        module, call = self.consumer(lambda *args: self.fail("kernel must not run"))
        self.assertIsNone(call())
        self.assertEqual((self.k.writes, self.v.writes), (1, 1))
        self.assertEqual(module.gated_decode_calls, {"native": 1, "triton_split": 0})
        self.assertEqual(module.gated_decode_last_route["fallback_reason"], "unsupported_shape")

    def test_migration_invalidates_capture_readiness(self):
        module, call = self.consumer(lambda *args: None)
        call()
        module._gated_decode_ready_device = SimpleNamespace(type="cuda", index=1)
        self.capture = True
        before = (self.k.writes, self.v.writes)
        with self.assertRaisesRegex(RuntimeError, "Warm gated"):
            call()
        self.assertEqual((self.k.writes, self.v.writes), before)


if __name__ == "__main__":
    unittest.main()
