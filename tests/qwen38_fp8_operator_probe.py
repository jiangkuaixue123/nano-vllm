"""Opt-in CUDA L0 probe; acceptance thresholds are supplied by the campaign.

This is deliberately not an automatically collected CPU smoke test. It runs
native FP8, pinned reference FP8, and an independent explicitly quantized FP32
oracle. The oracle's dequantization is diagnostic only, never an engine path.
"""

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch

from nanovllm.qwen38.quant import FP8Linear


def digest(tensor):
    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def golden(inputs, weight, scales):
    """Eager, explicitly rounded W8A8 operands and FP32 accumulation."""
    rows, k = inputs.shape
    n = weight.shape[0]
    output = torch.zeros(rows, n, device=inputs.device, dtype=torch.float32)
    for start in range(0, k, 128):
        values = inputs[:, start:start + 128].float()
        a_scale = values.abs().amax(dim=-1) / 448.0
        quant = (values / a_scale[:, None].clamp_min(1e-12)).to(torch.float8_e4m3fn)
        partial = quant.float() @ weight[:, start:start + 128].float().T
        b_scale = scales[:, start // 128].repeat_interleave(128)[:n]
        output += partial * a_scale[:, None] * b_scale[None, :]
    return output.bfloat16()


def compare(actual, expected, args):
    a, b = actual.float(), expected.float()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"finite": False, "passed": False}
    error = (a - b).abs()
    rmse = error.square().mean().sqrt().item()
    rms = b.square().mean().sqrt().item()
    nrmse = rmse / max(rms, 1e-8)
    # BF16 has seven fraction bits; retain ULP discrepancies as diagnostics,
    # including cancellation-sensitive values. Acceptance is external.
    _, exponent = torch.frexp(b.abs())
    spacing = torch.exp2((exponent - 8).float()).clamp_min(2.0 ** -133)
    spacing = torch.where(b == 0, 2.0 ** -133, spacing)
    max_ulp = (error.double() / spacing.double()).max().item()
    close = bool(torch.all(error <= args.atol + args.rtol * b.abs()).item())
    return {"finite": True, "max_abs": error.max().item(), "rmse": rmse,
            "reference_rms": rms, "nrmse": nrmse, "max_bf16_ulp": max_ulp,
            "allclose": close, "passed": close and nrmse <= args.max_nrmse}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-revision", required=True)
    parser.add_argument("--rtol", type=float, required=True)
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--max-nrmse", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.kernel_revision) != 40 or any(c not in "0123456789abcdef" for c in args.kernel_revision):
        parser.error("kernel revision must be an immutable lowercase SHA")
    if any(not math.isfinite(x) or x < 0 for x in (args.rtol, args.atol, args.max_nrmse)):
        parser.error("acceptance bars must be finite and nonnegative")
    if args.output.exists():
        parser.error("refusing to overwrite an existing result")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (9, 0):
        raise RuntimeError("L0 probe requires one visible SM90 CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from kernels import get_kernel

    from nanovllm.qwen38 import _triton_fp8

    reference = get_kernel("kernels-community/finegrained-fp8", revision=args.kernel_revision)
    reference_matmul = reference.matmul_2d
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    records = []
    # Exact production K/N geometries and bounded tail/zero/amplitude probes.
    specs = [(m, n, k, "random") for m in (1, 17) for n, k in (
        (10240, 5120), (17408, 5120), (5120, 17408), (5120, 6144),
    )] + [(3, 129, 257, "random"), (3, 256, 256, "zero"),
         (3, 256, 256, "amplitudes")]
    with torch.inference_mode():
        for m, n, k, kind in specs:
            inputs = torch.randn(m, k, device="cuda", generator=generator).bfloat16()
            weight = (torch.randn(n, k, device="cuda", generator=generator) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
            scales = torch.exp2(torch.randint(-8, 1, ((n + 127) // 128, (k + 127) // 128),
                                             device="cuda", generator=generator).float()).bfloat16().float()
            if kind == "zero":
                inputs.zero_()
            elif kind == "amplitudes":
                inputs *= torch.tensor([1e-8, 1, 1e8], device="cuda").bfloat16()[:, None]
            module = FP8Linear(k, n, device="cuda")
            module.weight.copy_(weight)
            module.weight_scale_inv.copy_(scales)
            actual = module(inputs)
            compiled = _triton_fp8.last_compiled_kernel
            ptx = compiled.asm["ptx"]
            mma_signatures = sorted({line.strip() for line in ptx.splitlines()
                                     if "mma" in line and "e4m3" in line})
            expected = reference_matmul(inputs, weight, scales, [128, 128], torch.bfloat16)
            oracle = golden(inputs, weight, scales)
            torch.cuda.synchronize()
            against_reference = compare(actual, expected, args)
            against_golden = compare(actual, oracle, args)
            # The independent oracle has an external NRMSE bar; allclose is
            # recorded but not a gate because eager division can round inputs
            # differently at E4M3 half-ULP boundaries.
            golden_passed = against_golden["finite"] and against_golden["nrmse"] <= args.max_nrmse
            records.append({"shape": [m, n, k], "kind": kind,
                            "inputs_sha256": digest(inputs), "weight_sha256": digest(weight),
                            "scales_sha256": digest(scales), "native_sha256": digest(actual),
                            "reference_sha256": digest(expected), "golden_sha256": digest(oracle),
                            "native_fp8_calls": module.native_fp8_calls,
                            "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
                            "fp8_mma_signatures": mma_signatures,
                            "against_reference": against_reference, "against_golden": against_golden,
                            "passed": against_reference["passed"] and golden_passed and bool(mma_signatures)})

    report = {"schema_version": 1, "scope": "operator_l0", "seed": args.seed,
              "kernel_revision": args.kernel_revision,
              "thresholds": {"rtol": args.rtol, "atol": args.atol, "max_nrmse": args.max_nrmse},
              "tf32": False, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "source_sha256": {"probe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                "native_kernel": hashlib.sha256(Path(_triton_fp8.__file__).read_bytes()).hexdigest(),
                                "reference_matmul": hashlib.sha256(Path(inspect.getfile(reference_matmul)).read_bytes()).hexdigest()},
              "cases": records, "passed": all(case["passed"] for case in records)}
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
