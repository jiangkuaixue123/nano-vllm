"""Standalone source binding for the opt-in FP8 probe; no engine/runtime imports."""

import hashlib
import inspect
import json
import re
from pathlib import Path


def kernel_fingerprint(kernel):
    """Match the control adapter's canonical complete-build fingerprint."""
    root = Path(kernel.__file__).absolute().parent.resolve(strict=True)
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        resolved = path.resolve(strict=True)
        cached_blob = (len(root.parents) >= 4 and root.parent.name == "build"
                       and root.parents[2].name == "snapshots"
                       and resolved.is_relative_to(root.parents[3] / "blobs"))
        if not resolved.is_relative_to(root) and not cached_blob:
            raise ValueError("Reference kernel source escapes package/cache")
        files.append({"path": path.relative_to(root).as_posix(),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if not files:
        raise ValueError("Reference kernel source missing")
    snapshot = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"files": files, "snapshot_sha256": snapshot}


def bind_reference_kernel(kernel, revision, expected_sha256):
    """Reject local overrides, wrong revisions, changed builds or entrypoints."""
    if (not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)):
        raise ValueError("Reference requires immutable revision and source SHA-256")
    root = Path(kernel.__file__).absolute().parent
    for candidate in (root, root.resolve(strict=True)):
        if (len(candidate.parents) < 4 or candidate.name != "torch-cuda"
                or candidate.parent.name != "build" or candidate.parents[1].name != revision
                or candidate.parents[2].name != "snapshots"):
            raise ValueError("Reference must originate in the requested torch-cuda snapshot")
    function = getattr(kernel, "matmul_2d", None)
    if not callable(function):
        raise TypeError("Reference lacks the FP8 matmul entrypoint")
    source = Path(inspect.getfile(function)).absolute()
    if not source.is_relative_to(root):
        raise ValueError("Reference entrypoint originates outside the pinned build")
    fingerprint = kernel_fingerprint(kernel)
    if source.relative_to(root).as_posix() not in {item["path"] for item in fingerprint["files"]}:
        raise ValueError("Reference entrypoint source is not fingerprinted")
    if fingerprint["snapshot_sha256"] != expected_sha256:
        raise ValueError("Reference kernel differs from frozen source")
    return fingerprint


def case_specs():
    """All distinct native FP8 production N/K geometries at decode and prefill."""
    return [(m, n, k, "random", "nonpower2_bf16") for m in (1, 17) for n, k in (
        (10240, 5120), (6144, 5120), (12288, 5120), (1024, 5120),
        (17408, 5120), (5120, 17408), (5120, 6144),
    )] + [(3, 129, 257, "random", "power2"),
         (3, 129, 257, "random", "nonpower2_bf16"),
         (3, 256, 256, "zero", "nonpower2_bf16"),
         (3, 256, 256, "amplitudes", "nonpower2_bf16")]
