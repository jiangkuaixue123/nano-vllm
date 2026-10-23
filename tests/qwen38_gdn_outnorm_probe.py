"""Explicit authorized CUDA diagnostic; no default CI, tolerances or PASS claim.

Checkpoint bytes must be authenticated by the caller's frozen model manifest.
Optional captured operand dictionaries contain x, gate, weight, epsilon and
layer_index from an actual GatedRMSNorm call, retaining the gate storage/strides.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from nanovllm.qwen38._triton_gdn_outnorm import gated_rms_norm
from nanovllm.qwen38.config import Qwen38TextConfig
from nanovllm.qwen38.loader import _file, _object


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def checkpoint_weights(root):
    config = Qwen38TextConfig.from_dict(json.loads(_file(root, "config.json").read_bytes(),
                                                 object_pairs_hook=_object))
    index = json.loads(_file(root, "model.safetensors.index.json").read_bytes(),
                       object_pairs_hook=_object)["weight_map"]
    result = {}
    for layer, kind in enumerate(config.layer_types):
        if kind != "linear_attention":
            continue
        key = f"model.language_model.layers.{layer}.linear_attn.norm.weight"
        with safe_open(_file(root, index[key]), framework="pt", device="cpu") as shard:
            weight = shard.get_tensor(key)
        if weight.shape != (128,) or weight.dtype != torch.bfloat16:
            raise ValueError("Expected real BF16 GDN norm weight with width 128")
        result[layer] = weight.cuda()
    return config, result


def check(x, gate, weight, eps):
    snapshots = [tensor_hash(t) for t in (x, gate, weight)]
    expected = torch.nn.functional.rms_norm(x, (128,), weight, eps) * torch.nn.functional.silu(gate)
    actual = gated_rms_norm(x, gate, weight, eps)
    torch.cuda.synchronize()
    finite = torch.isfinite(expected)
    changed = (actual.view(torch.int16) != expected.view(torch.int16)) & finite
    classification = all(torch.equal(fn(actual), fn(expected))
                         for fn in (torch.isnan, torch.isposinf, torch.isneginf))
    unchanged = snapshots == [tensor_hash(t) for t in (x, gate, weight)]
    fresh = actual.is_contiguous() and all(actual.data_ptr() != t.data_ptr()
                                         for t in (x, gate, weight))
    mismatches = int(changed.sum().item())
    return {"shape": list(x.shape), "gate_stride": list(gate.stride()),
            "gate_storage_offset": gate.storage_offset(), "input_sha256": snapshots,
            "finite_mismatches": mismatches, "nonfinite_classification_equal": classification,
            "inputs_unchanged": unchanged, "fresh_contiguous_output": fresh,
            "output_dtype": str(actual.dtype), "output_sha256": tensor_hash(actual),
            "reference_sha256": tensor_hash(expected),
            "complete": mismatches == 0 and classification and unchanged and fresh}


def cases(weights):
    torch.manual_seed(8128)
    # All actual GDN norm weights, including nonunit/negative channels.
    for layer, weight in weights.items():
        x = torch.randn(1, 7, 48, 128, dtype=torch.bfloat16, device="cuda")
        packed = torch.randn(1, 7, 16384, dtype=torch.bfloat16, device="cuda")
        gate = packed[..., 10240:].view_as(x)
        yield f"checkpoint-{layer}-merged-7", x, gate, weight
    first, last = min(weights), max(weights)
    for layer in dict.fromkeys((first, last)):
        for tokens in (1, 8192):
            x = torch.randn(1, tokens, 48, 128, dtype=torch.bfloat16, device="cuda")
            packed = torch.randn(1, tokens, 16384, dtype=torch.bfloat16, device="cuda")
            yield f"checkpoint-{layer}-merged-{tokens}", x, packed[..., 10240:].view_as(x), weights[layer]
    weight = weights[first]
    x = torch.randn(1, 7, 48, 128, dtype=torch.bfloat16, device="cuda")
    packed = torch.randn(1, 7, 16480, dtype=torch.bfloat16, device="cuda")
    yield "merged-qkvzba-view", x, packed[..., 10240:16384].view_as(x), weight
    bits = torch.arange(65536, dtype=torch.int32, device="cuda").to(torch.int16)
    values = bits.view(torch.bfloat16).reshape(1, 512, 1, 128)
    yield "all-bf16-gate", torch.ones_like(values), values, weight
    yield "all-bf16-input", values, torch.ones_like(values), weight
    for scale in (0.0, 2.0**-60, 2.0**60):
        x = torch.randn(1, 3, 5, 128, dtype=torch.bfloat16, device="cuda") * scale
        storage = torch.randn(1, 3, 5, 256, dtype=torch.bfloat16, device="cuda")
        yield f"strided-tail-scale-{scale}", x, storage[..., 1::2], weight
    x = torch.randn(1, 7, 3, 128, dtype=torch.bfloat16, device="cuda")
    yield "broadcast-gate", x, x[:, :1, :1, :].expand_as(x), weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--operands", action="append", default=[])
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a new output receipt; failed checks must be retained")
    report = {"complete": False, "verdict": "INCONCLUSIVE", "rows": [],
              "actual_operand_files": len(args.operands), "model_or_graph_qualified": False}

    def save():
        output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        with torch.inference_mode():
            config, weights = checkpoint_weights(Path(args.model).resolve(strict=True))
            report["weights_sha256"] = {str(layer): tensor_hash(w) for layer, w in weights.items()}
            report["epsilon"] = config.rms_norm_eps

            def record(name, x, gate, weight):
                row = {"case": name, **check(x, gate, weight, config.rms_norm_eps)}
                report["rows"].append(row)
                save()
                if not row["complete"]:
                    raise RuntimeError("Exact output norm diagnostic failed; no tolerance substituted")

            for name, x, gate, weight in cases(weights):
                record(name, x, gate, weight)
            for index, path in enumerate(args.operands):
                captured = torch.load(path, map_location="cuda", weights_only=True)
                weight = weights[captured["layer_index"]]
                if (tensor_hash(captured["weight"]) != tensor_hash(weight)
                        or captured["epsilon"] != config.rms_norm_eps):
                    raise ValueError("Captured operand norm weight/epsilon differs from checkpoint")
                record(f"actual-operands-{index}", captured["x"], captured["gate"], captured["weight"])
        report["complete"] = True
    except Exception as error:
        report["error_type"] = type(error).__name__
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
