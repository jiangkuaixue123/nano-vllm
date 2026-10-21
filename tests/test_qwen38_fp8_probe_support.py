"""Source provenance rejection tests require no CUDA or third-party kernels."""

import hashlib
import json
from types import FunctionType, ModuleType

import pytest
from qwen38_fp8_probe_support import (
    bind_reference_kernel,
    case_specs,
    kernel_fingerprint,
)


def snapshot(tmp_path):
    revision = "a" * 40
    cache = tmp_path / "kernel-cache"
    root = cache / "snapshots" / revision / "build" / "torch-cuda"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("# synthetic test kernel\n")
    (root / "matmul.py").write_text("def matmul_2d(): return None\n")
    (root / "utils.py").write_text("SCALE = 448\n")
    module = ModuleType("test_kernel")
    module.__file__ = str(root / "__init__.py")
    def synthetic_matmul():
        return None

    module.matmul_2d = FunctionType(synthetic_matmul.__code__.replace(
        co_filename=str(root / "matmul.py")), module.__dict__)
    return module, root, revision


def test_complete_build_fingerprint_matches_canonical_control_algorithm(tmp_path):
    module, root, revision = snapshot(tmp_path)
    files = [{"path": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
             for name in ("__init__.py", "matmul.py", "utils.py")]
    expected = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    actual = bind_reference_kernel(module, revision, expected)
    assert actual == {"files": files, "snapshot_sha256": expected}


def test_hf_blob_symlinks_work_without_omitting_utils(tmp_path):
    module, root, revision = snapshot(tmp_path)
    expected = kernel_fingerprint(module)["snapshot_sha256"]
    blobs = root.parents[3] / "blobs"
    blobs.mkdir()
    for name in ("__init__.py", "matmul.py", "utils.py"):
        destination = blobs / name
        (root / name).rename(destination)
        (root / name).symlink_to(destination)
    assert bind_reference_kernel(module, revision, expected)["snapshot_sha256"] == expected
    (blobs / "utils.py").write_text("SCALE = 1\n")
    with pytest.raises(ValueError, match="differs"):
        bind_reference_kernel(module, revision, expected)


@pytest.mark.parametrize("change", ["revision", "local_override", "escape", "entrypoint", "utils"])
def test_wrong_reference_rejected(tmp_path, change):
    module, root, revision = snapshot(tmp_path)
    expected = kernel_fingerprint(module)["snapshot_sha256"]
    if change == "revision":
        revision = "b" * 40
    elif change == "local_override":
        module.__file__ = str(tmp_path / "__init__.py")
    elif change == "escape":
        (tmp_path / "outside.py").write_text("SCALE = 448\n")
        (root / "utils.py").unlink()
        (root / "utils.py").symlink_to(tmp_path / "outside.py")
    elif change == "entrypoint":
        module.matmul_2d = lambda: None
    elif change == "utils":
        (root / "utils.py").write_text("SCALE = 1\n")
    with pytest.raises(ValueError):
        bind_reference_kernel(module, revision, expected)


def test_bytecode_cache_is_excluded(tmp_path):
    module, root, revision = snapshot(tmp_path)
    expected = kernel_fingerprint(module)["snapshot_sha256"]
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "utils.pyc").write_bytes(b"bytecode")
    assert bind_reference_kernel(module, revision, expected)["snapshot_sha256"] == expected


def test_all_production_geometries_decode_and_prefill_have_nonpower2_scales():
    production = {(10240, 5120), (6144, 5120), (12288, 5120), (1024, 5120),
                  (17408, 5120), (5120, 17408), (5120, 6144)}
    for m in (1, 17):
        assert {(n, k) for rows, n, k, _, scale in case_specs()
                if rows == m and scale == "nonpower2_bf16"} == production
    assert len(case_specs()) == 18
