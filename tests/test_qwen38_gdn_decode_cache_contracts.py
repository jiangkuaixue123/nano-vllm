"""Execute the actual cache method with metadata doubles; no Torch dependency."""
# ruff: noqa: S102 -- executes only this checkout's selected AST, without Torch.
import ast
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

TREE = ast.parse((Path(__file__).parents[1] / "nanovllm/qwen38/model.py").read_text())
DELTA = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "DeltaNet")
METHOD = next(n for n in DELTA.body if isinstance(n, ast.FunctionDef) and n.name == "_decode_a_log")


class Tensor:
    def __init__(self, dtype="bf16", device="cuda", version=0):
        self.dtype, self.device, self.version = dtype, device, version
        self.shape, self.casts = (48,), 0
        self.is_cuda = device == "cuda"

    @property
    def _version(self):
        if self.version is None:
            raise RuntimeError("Inference tensor has no version")
        return self.version

    def stride(self):
        return (1,)

    def data_ptr(self):
        return id(self)

    def detach(self):
        return self

    def float(self):
        self.casts += 1
        return Tensor("fp32", self.device)


class CacheControls(unittest.TestCase):
    def setUp(self):
        self.capture = False
        torch = SimpleNamespace(float32="fp32", cuda=SimpleNamespace(
            is_current_stream_capturing=lambda: self.capture))
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[METHOD], type_ignores=[]), "actual_cache", "exec"), namespace)
        self.module = SimpleNamespace(A_log=Tensor())
        self.registrations = []

        def register(name, tensor, *, persistent):
            self.registrations.append((name, persistent))
            setattr(self.module, name, tensor)
        self.module.register_buffer = register
        self.call = lambda: namespace["_decode_a_log"](self.module)

    def test_warm_reuse_and_nonpersistent(self):
        first = self.call()
        self.capture = True
        self.assertIs(self.call(), first)
        self.assertEqual(self.module.A_log.casts, 1)
        self.assertEqual(self.registrations, [("_decode_a_log_fp32", False)])

    def test_parameter_mutation_and_replacement(self):
        first = self.call()
        self.module.A_log.version += 1
        second = self.call()
        self.assertIsNot(first, second)
        self.module.A_log = Tensor()
        self.assertIsNot(self.call(), second)

    def test_migration_and_cache_dtype_invalidates(self):
        first = self.call()
        self.module.A_log.device = "cpu"
        self.module.A_log.is_cuda = False
        second = self.call()
        self.assertEqual(second.device, "cpu")
        self.assertIsNot(first, second)
        second.dtype = "bf16"
        self.assertIsNot(self.call(), second)

    def test_cold_or_stale_capture_rejects_before_cast(self):
        self.capture = True
        with self.assertRaises(RuntimeError):
            self.call()
        self.assertEqual(self.module.A_log.casts, 0)
        self.capture = False
        self.call()
        self.module.A_log.version += 1
        self.capture = True
        with self.assertRaises(RuntimeError):
            self.call()
        self.assertEqual(self.module.A_log.casts, 1)

    def test_versionless_falls_back_without_preparation(self):
        self.module.A_log.version = None
        self.assertIs(self.call(), self.module.A_log)
        self.assertEqual(self.registrations, [])

    def test_actual_constructor_selection(self):
        init = next(n for n in DELTA.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        # Execute the constructor's actual backend prefix, before tensor allocation.
        body = []
        for n in init.body[1:]:
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Attribute) and n.targets[0].attr == "gdn_conv_backend":
                break
            body.append(n)
        code = compile(ast.Module(body=body, type_ignores=[]), "actual_constructor_prefix", "exec")
        for env, expected in [({}, "native"), ({"NANOVLLM_GDN": "flashinfer", "NANOVLLM_GDN_DECODE_A_LOG": "cached_fp32"}, "cached_fp32")]:
            with patch.dict(os.environ, env, clear=True):
                obj = SimpleNamespace()
                exec(code, {"os": os, "self": obj, "config": None})
                self.assertEqual(obj.gdn_decode_a_log, expected)
        for value in ["unknown", "cached_fp32"]:
            with (patch.dict(os.environ, {"NANOVLLM_GDN_DECODE_A_LOG": value}, clear=True),
                  self.assertRaises(ValueError)):
                exec(code, {"os": os, "self": SimpleNamespace(), "config": None})

    def test_preparation_precedes_state_or_projection_work(self):
        forward = next(n for n in DELTA.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
        self.assertEqual(forward.body[2].body[0].value.func.attr, "_decode_a_log")
        self.assertIn("state.conv is not None", ast.unparse(forward.body[2].test))
        self.assertIn("not state.fresh", ast.unparse(forward.body[2].test))


if __name__ == "__main__":
    unittest.main()
