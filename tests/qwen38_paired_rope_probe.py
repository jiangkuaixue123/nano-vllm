"""Explicit CUDA exactness probe; never run by default pytest discovery.

Use only inside an authorized allocation. No operator timing or service claim.
Optional captured operands must be trusted task-owned tensor dictionaries.
"""

import argparse
import json
from itertools import product
from pathlib import Path

import torch

from nanovllm.qwen38._triton_rope import paired_partial_rope
from nanovllm.qwen38.config import Qwen38TextConfig
from nanovllm.qwen38.model import apply_rope, rotary_tables


def directed_zero_cases():
    """Joint operands omitted by individual-word coverage; not exhaustive RoPE."""
    zeros = (0x0000, 0x8000)
    units = (0x3F80, 0xBF80)
    table_words = zeros + units
    for label, first_words in (("zero-halves", zeros), ("unit-first-zero-partner", units)):
        rows = list(product(first_words, zeros, table_words, table_words))
        values = torch.tensor(rows, dtype=torch.int32, device="cuda").to(torch.int16).view(torch.bfloat16)
        tokens = len(rows)
        q = torch.zeros((1, 24, tokens, 256), dtype=torch.bfloat16, device="cuda")
        k = torch.zeros((1, 4, tokens, 256), dtype=torch.bfloat16, device="cuda")
        for x in (q, k):
            x[..., :32] = values[:, 0, None]
            x[..., 32:64] = values[:, 1, None]
            x[..., 64::2] = -0.0
        c = values[:, 2, None].expand(tokens, 64)[None, None]
        s = values[:, 3, None].expand(tokens, 64)[None, None]
        yield f"directed-joint-{label}", (q, k, c, s)


def cases():
    torch.manual_seed(8128)
    config = Qwen38TextConfig()
    for tokens in (1, 2, 7, 8192):
        # Native norm allocates token-major; the attention path transposes it.
        q = torch.randn(1, tokens, 24, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        k = torch.randn(1, tokens, 4, 256, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        for start in (0, 8192):
            c, s = rotary_tables(config, torch.arange(start, start + tokens, device="cuda"), q.dtype)
            yield f"postnorm-view-{tokens}-position-{start}", (q, k, c[None, None], s[None, None])
    for tokens in (1, 3):
        q = torch.randn(1, 24, tokens, 513, device="cuda", dtype=torch.bfloat16)[..., 1::2]
        k = torch.randn(1, 4, tokens, 769, device="cuda", dtype=torch.bfloat16)[..., 1::3]
        # Deliberately independent halves and table strides, including offsets.
        c = torch.randn(1, 1, tokens, 129, device="cuda", dtype=torch.bfloat16)[..., 1::2]
        s = torch.randn(1, 1, tokens, 193, device="cuda", dtype=torch.bfloat16)[..., 1::3]
        yield f"offset-column-strides-{tokens}", (q, k, c, s)
        yield f"broadcast-{tokens}", (q[:, :1, :1].expand(1, 24, tokens, 256),
                                      k[:, :1, :1].expand(1, 4, tokens, 256),
                                      c[:, :, :1].expand(1, 1, tokens, 64), s)
    # All BF16 operand encodings, not a claim of exhaustive joint combinations.
    def bit_values(shape):
        count = 1
        for size in shape:
            count *= size
        return (torch.arange(count, device="cuda", dtype=torch.int32) % 65536).to(
            torch.int16,
        ).view(torch.bfloat16).reshape(shape)

    def rope_bit_values(heads):
        # Each rotary half independently contains all 65,536 words, even for
        # four K heads: 4 * 512 * 32 == 65,536. Sequential full-width rows would
        # instead hide three quarters of the words (including Inf/NaN) in tails.
        first = bit_values((1, heads, 512, 32))
        second = first.flip((-1,)).contiguous()
        tail = bit_values((1, heads, 512, 192))
        return torch.cat((first, second, tail), dim=-1)

    q = rope_bit_values(24)
    k = rope_bit_values(4)
    for cosine, sine in ((1.0, 0.0), (0.0, -0.0), (0.70703125, 0.70703125), (-1.0, 1.0)):
        c = torch.full((1, 1, 512, 64), cosine, device="cuda", dtype=torch.bfloat16)
        s = torch.full_like(c, sine)
        yield f"all-bf16-inputs-{cosine}-{sine}", (q, k, c, s)
    c = bit_values((1, 1, 1024, 64))
    s = c.flip((-1,)).contiguous()
    yield "all-bf16-tables", (torch.ones((1, 24, 1024, 256), device="cuda", dtype=torch.bfloat16),
                               torch.ones((1, 4, 1024, 256), device="cuda", dtype=torch.bfloat16), c, s)
    q = torch.ones((1, 24, 1, 256), device="cuda", dtype=torch.bfloat16)
    k = torch.ones((1, 4, 1, 256), device="cuda", dtype=torch.bfloat16)
    q[..., :32] = k[..., :32] = 1.0078125
    c = torch.full((1, 1, 1, 64), 0.70703125, device="cuda", dtype=torch.bfloat16)
    yield "two-product-rounding-cancellation", (q, k, c, c.clone())
    yield from directed_zero_cases()


def check(operands):
    q, k, c, s = operands
    snapshots = [x.clone() for x in operands]
    expected = apply_rope(q, c, s), apply_rope(k, c, s)
    actual = paired_partial_rope(q, k, c, s)
    torch.cuda.synchronize()
    rows = []
    for reference, result in zip(expected, actual):
        finite = torch.isfinite(reference)
        mismatches = (reference.contiguous().view(torch.int16) != result.view(torch.int16)) & finite
        classes = (torch.equal(torch.isnan(reference), torch.isnan(result))
                   and torch.equal(torch.isposinf(reference), torch.isposinf(result))
                   and torch.equal(torch.isneginf(reference), torch.isneginf(result)))
        rows.append({"shape": list(result.shape), "stride": list(result.stride()),
                     "dtype": str(result.dtype), "finite_bit_mismatches": int(mismatches.sum().item()),
                     "shape_dtype_equal": result.shape == reference.shape and result.dtype == reference.dtype,
                     "nonfinite_classes_equal": classes,
                     "tail_bits_equal": torch.equal(reference[..., 64:].contiguous().view(torch.int16),
                                                     result[..., 64:].contiguous().view(torch.int16)),
                     "fresh_contiguous": result.is_contiguous() and all(
                         result.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
                         for x in operands)})
    immutable = all(torch.equal(x.contiguous().view(torch.int16), y.contiguous().view(torch.int16))
                    for x, y in zip(operands, snapshots))
    separate = actual[0].untyped_storage().data_ptr() != actual[1].untyped_storage().data_ptr()
    complete = immutable and separate and all(
        row["finite_bit_mismatches"] == 0 and row["nonfinite_classes_equal"]
        and row["shape_dtype_equal"] and row["tail_bits_equal"] and row["fresh_contiguous"] for row in rows)
    return {"operands": [{"shape": list(x.shape), "stride": list(x.stride()),
                          "dtype": str(x.dtype), "storage_offset": x.storage_offset()} for x in operands],
            "outputs": rows, "inputs_unchanged": immutable,
            "outputs_separate": separate, "complete": complete}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--capture", action="append", default=[])
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise ValueError("Probe output must be a fresh path")
    rows = []

    def record(name, operands):
        row = {"case": name, **check(operands)}
        rows.append(row)
        destination.write_text(json.dumps({"complete": False, "verdict": "INCONCLUSIVE",
                                           "rows": rows}, indent=2) + "\n")
        if not row["complete"]:
            raise RuntimeError("Exact partial-RoPE comparison failed; no tolerance substituted")

    with torch.inference_mode():
        for name, operands in cases():
            record(name, operands)
        for index, path in enumerate(args.capture):
            captured = torch.load(path, map_location="cuda", weights_only=True)
            if not isinstance(captured, dict) or set(captured) != {"query", "key", "cos", "sin"}:
                raise ValueError("Captured operands must contain exactly query/key/cos/sin")
            record(f"captured-operands-{index}", tuple(captured[n] for n in ("query", "key", "cos", "sin")))
    destination.write_text(json.dumps({"complete": True, "verdict": "INCONCLUSIVE",
                                       "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
