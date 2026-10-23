"""Explicit CUDA operator probe, excluded from ordinary pytest discovery.

Run only in an authorized GPU allocation. No tolerance or service/quality claim.
"""

import argparse
import json
from pathlib import Path

import torch

from nanovllm.qwen38._triton_silu import silu_and_mul as kernel_fn


def reference_fn(gate, up):
    return torch.nn.functional.silu(gate) * up


def get_inputs():
    # Every BF16 bit pattern, including signed zeros, subnormals, infinities/NaNs.
    bits = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16)
    values = bits.view(torch.bfloat16).reshape(256, 256)
    yield "all-bf16-gate-up-one", values, torch.ones_like(values)
    yield "all-bf16-paired-values", values, values.flip((0, 1)).contiguous()
    for multiplier in (-3.0, 0.0, 0.5, 1.5, 3.0):
        yield f"all-bf16-gate-up-{multiplier}", values, torch.full_like(values, multiplier)
    torch.manual_seed(8128)
    for rows, columns in ((1, 17408), (7, 17408), (8192, 17408), (3, 17), (5, 1025)):
        packed = torch.randn(1, rows, 2 * columns, dtype=torch.bfloat16, device="cuda")
        gate, up = packed.split(columns, dim=-1)
        yield f"merged-view-{rows}x{columns}", gate, up
    # Different positive column and batch strides, plus expanded read-only inputs.
    base = torch.randn(2, 5, 1030, dtype=torch.bfloat16, device="cuda")
    yield "column-strides", base[:, :, 1::2], base[:, :, ::2]
    yield "batch-transpose", base.transpose(0, 1), base.flip((2,)).transpose(0, 1)
    yield "broadcast-rows", base[:, :1, :].expand(2, 7, 1030), base[:, :1, :].expand(2, 7, 1030)


def check(gate, up):
    expected = reference_fn(gate, up)
    snapshots = gate.clone(), up.clone()
    actual = kernel_fn(gate, up)
    torch.cuda.synchronize()
    finite = torch.isfinite(expected)
    # Compare finite results bitwise (including signed zero); NaN payload bits
    # are not an inference contract, but NaN/Inf classifications and signs are.
    finite_exact = torch.equal(actual[finite].view(torch.int16), expected[finite].view(torch.int16))
    classifications = (torch.equal(torch.isnan(actual), torch.isnan(expected))
                       and torch.equal(torch.isposinf(actual), torch.isposinf(expected))
                       and torch.equal(torch.isneginf(actual), torch.isneginf(expected)))
    immutable = all(torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16))
                    for x, y in zip((gate, up), snapshots))
    changed = (actual.view(torch.int16) != expected.view(torch.int16)) & finite
    return {"shape": list(gate.shape), "gate_stride": list(gate.stride()),
            "up_stride": list(up.stride()), "finite_bitwise_equal": finite_exact,
            "finite_mismatches": int(changed.sum().item()),
            "nonfinite_classification_equal": classifications, "inputs_unchanged": immutable,
            "fresh_contiguous_output": actual.is_contiguous() and actual.data_ptr() not in (
                gate.data_ptr(), up.data_ptr()),
            "complete": finite_exact and classifications and immutable}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    with torch.inference_mode():
        for name, gate, up in get_inputs():
            row = {"case": name, **check(gate, up)}
            rows.append(row)
            Path(args.output).write_text(json.dumps({"complete": False, "rows": rows,
                                                    "verdict": "INCONCLUSIVE"}, indent=2) + "\n")
            if not row["complete"] or not row["fresh_contiguous_output"]:
                raise RuntimeError("exact BF16 operator qualification failed; no tolerance substituted")
    Path(args.output).write_text(json.dumps({"complete": True, "rows": rows,
                                            "verdict": "INCONCLUSIVE"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
