"""Explicit CUDA diagnostic, outside pytest discovery; no acceptance tolerance.

Run only inside a separately authorized GPU allocation after source, model and
runtime guards. Uses real checkpoint convolution weights and synthetic inputs;
whole-model activation/state and HTTP checks remain a separate requirement.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors import safe_open

from nanovllm.qwen38._triton_conv import causal_conv_silu as kernel_fn


@triton.jit
def _accumulation_probe(X, W, FMA, SEPARATE, C: tl.constexpr, T: tl.constexpr,
                        XC: tl.constexpr, XT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c, t = index // T, index % T
    fused = tl.full((BLOCK,), 0.0, tl.float32)
    separate = tl.full((BLOCK,), 0.0, tl.float32)
    for k in tl.static_range(4):
        valid = (c < C) & (t - 3 + k >= 0)
        x = tl.load(X + c * XC + (t - 3 + k) * XT, valid, other=0).to(tl.float32)
        w = tl.load(W + c * 4 + k, c < C, other=0).to(tl.float32)
        fused = tl.where(valid, tl.fma(w, x, fused), fused)
        # Launch disables implicit contraction, giving a distinct diagnostic.
        separate = tl.where(valid, separate + w * x, separate)
    tl.store(FMA + index, fused.to(tl.bfloat16, fp_downcast_rounding="rtne"), c < C)
    tl.store(SEPARATE + index, separate.to(tl.bfloat16, fp_downcast_rounding="rtne"), c < C)


def accumulation_case(raw, weight):
    channels, count = raw.shape[1:]
    fused, separate = torch.empty(raw.shape, device="cuda", dtype=raw.dtype), torch.empty(raw.shape, device="cuda", dtype=raw.dtype)
    _accumulation_probe[(triton.cdiv(channels * count, 256),)](
        raw, weight, fused, separate, channels, count, *raw.stride()[1:], 256,
        enable_fp_fusion=False)
    expected = F.conv1d(raw, weight, groups=channels, padding=3)[:, :, :count]
    return {"explicit_fma_vs_torch": compare(fused, expected),
            "separate_mul_add_vs_torch": compare(separate, expected),
            "fma_vs_separate": compare(fused, separate)}


def reference_fn(raw, weight, history=None, *, state_out=None):
    count = raw.shape[-1]
    mixed = torch.cat((history, raw), dim=-1) if history is not None else raw
    window = F.pad(mixed, (max(0, 4 - mixed.shape[-1]), 0))[:, :, -4:]
    if state_out is None:
        state_out = window.clone()
    else:
        state_out.copy_(window)
    padding = 0 if history is not None and count == 1 else 3
    convolved = F.conv1d(mixed, weight, groups=mixed.shape[1], padding=padding)
    if padding:
        convolved = convolved[:, :, :mixed.shape[-1]]
    return F.silu(convolved[:, :, -count:]), state_out


def get_inputs(weight, count, *, merged=True):
    channels = weight.shape[0]
    width = channels + channels * 3 // 5 if merged else channels
    packed = torch.randn(1, count, width, device=weight.device, dtype=torch.bfloat16)
    return packed[:, :, :channels].transpose(1, 2)


def digest(tensor):
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def compare(actual, reference):
    a, b = actual.float(), reference.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = (a[finite].double() - b[finite].double()).abs()
    return {
        "shape": list(actual.shape), "actual_stride": list(actual.stride()),
        "reference_stride": list(reference.stride()),
        "bit_mismatches": int((actual.view(torch.int16) != reference.view(torch.int16)).sum().item()),
        "finite_pairs": int(finite.sum().item()),
        "finite_max_abs": float(delta.max().item()) if delta.numel() else None,
        "finite_rmse": float(delta.square().mean().sqrt().item()) if delta.numel() else None,
        "nan_class_mismatches": int((a.isnan() != b.isnan()).sum().item()),
        "posinf_class_mismatches": int((a.isposinf() != b.isposinf()).sum().item()),
        "neginf_class_mismatches": int((a.isneginf() != b.isneginf()).sum().item()),
        "actual_sha256": digest(actual), "reference_sha256": digest(reference),
    }


def case(raw, weight, history, *, buffered):
    raw_before, weight_before = digest(raw), digest(weight)
    old_before = digest(history) if history is not None else None
    target = history.clone() if history is not None else torch.full(
        (1, raw.shape[1], 4), 19, device=raw.device, dtype=raw.dtype)
    expected_target = target.clone()
    # In-place decode exercises identical history/output pointers.
    actual_history = target if buffered and history is not None else history
    expected_history = expected_target if buffered and history is not None else history
    expected, expected_state = reference_fn(
        raw, weight, expected_history, state_out=expected_target if buffered else None)
    actual, state = kernel_fn(raw, weight, actual_history, state_out=target if buffered else None)
    torch.cuda.synchronize()
    state_stats = compare(state, expected_state)
    immutable = digest(raw) == raw_before and digest(weight) == weight_before
    immutable = immutable and (history is None or digest(history) == old_before)
    correct_storage = (state.data_ptr() == target.data_ptr()) if buffered else (
        history is None or state.data_ptr() != history.data_ptr())
    return {
        "tokens": raw.shape[-1], "input_stride": list(raw.stride()),
        "decode": history is not None, "buffered": buffered,
        "output": compare(actual, expected), "state": state_stats,
        "input_weight_old_state_unchanged": immutable,
        "state_storage_contract": correct_storage,
        "state_exact": state_stats["bit_mismatches"] == 0,
        "state_layout_exact": state.stride() == expected_state.stride(),
        "output_layout_exact": actual.stride() == expected.stride(),
    }


def graph_case(weight):
    # Mutating a separate warmup state leaves the real comparison seed intact.
    channels = weight.shape[0]
    raw = get_inputs(weight, 1)
    seed = torch.randn(1, channels, 4, device="cuda", dtype=torch.bfloat16)
    eager, captured = seed.clone(), seed.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warm = seed.clone()
        for _ in range(2):
            kernel_fn(raw, weight, warm, state_out=warm)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, final_state = kernel_fn(raw, weight, captured, state_out=captured)
    captured.copy_(seed)
    rows = []
    for step in range(16):
        raw.copy_(get_inputs(weight, 1))
        expected, _ = kernel_fn(raw, weight, eager, state_out=eager)
        graph.replay()
        torch.cuda.synchronize()
        rows.append({"step": step, "output": compare(output, expected),
                     "state": compare(final_state, eager)})
    return {"rows": rows, "fixed_state_pointer": final_state.data_ptr() == captured.data_ptr(),
            "exact": all(r[k]["bit_mismatches"] == 0 for r in rows for k in ("output", "state"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing diagnostic")
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.cuda.get_device_capability(0) != (9, 0)):
        raise RuntimeError("This bounded diagnostic requires one SM90 CUDA GPU")
    torch.manual_seed(67)
    root = args.model.resolve(strict=True)
    mapping = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted((n for n in mapping if n.endswith(".linear_attn.conv1d.weight")),
                   key=lambda n: int(n.split(".layers.")[1].split(".")[0]))
    if len(names) != 48:
        raise ValueError("Expected the dedicated model's 48 convolution weights")
    selected = [names[0], names[len(names) // 2], names[-1]]
    report = {
        "complete": False, "scope": "operator_observation_only", "quality_verdict": "INCONCLUSIVE",
        "torch": torch.__version__, "torch_git": torch.version.git_version,
        "torch_build_config": torch.__config__.show(), "triton": triton.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
        "weights": [], "cases": [], "graphs": [], "trace_kernel_names": [],
        "accumulation_observations": [],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    with torch.inference_mode():
        for index, name in enumerate(selected):
            path = (root / mapping[name]).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("Checkpoint shard escapes model root")
            with safe_open(path, framework="pt", device="cpu") as shard:
                weight = shard.get_tensor(name).cuda()
            if weight.dtype != torch.bfloat16 or weight.shape != (10240, 1, 4):
                raise ValueError("Unexpected dedicated convolution weight contract")
            report["weights"].append({"key": name, "sha256": digest(weight)})
            for count in ([1, 3, 4, 33, 8192] if index == 0 else [1, 33]):
                raw = get_inputs(weight, count)
                for buffered in (False, True):
                    record = case(raw, weight, None, buffered=buffered)
                    report["cases"].append({"weight": name, "kind": "fresh", **record})
                    save()
            raw = get_inputs(weight, 1)
            history = torch.randn(1, 10240, 4, device="cuda", dtype=torch.bfloat16)
            for buffered in (False, True):
                report["cases"].append({"weight": name, "kind": "decode",
                                        **case(raw, weight, history, buffered=buffered)})
                save()
            report["graphs"].append({"weight": name, **graph_case(weight)})
            save()
            if index == 0:
                report["accumulation_observations"].append(
                    {"kind": "real_weight", **accumulation_case(get_inputs(weight, 33), weight)})
                # Record the actual Torch convolution/SiLU kernel names. Source
                # and NVCC defaults alone do not prove the installed binary's FMA.
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    reference_fn(raw, weight, history)
                    torch.cuda.synchronize()
                report["trace_kernel_names"] = sorted({event.name for event in prof.events()})
        # Masked tails, separate projection layout, signed zeros, subnormals,
        # infinities and NaNs. No output tolerance or finite-only waiver.
        edge = torch.tensor([0, 0x8000, 1, 0x8001, 0x0080, 0x3f80, 0xbf80,
                             0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1],
                            dtype=torch.uint16, device="cuda").view(torch.bfloat16)
        raw = edge.repeat(33 * 5 // edge.numel() + 1)[:33 * 5].reshape(1, 5, 33).transpose(1, 2)
        weight = torch.tensor([1, -1, 0.5, 0.25], device="cuda", dtype=torch.bfloat16).repeat(33, 1).unsqueeze(1)
        report["accumulation_observations"].append(
            {"kind": "edge", **accumulation_case(raw, weight)})
        for count in (1, 3, 4, 5):
            report["cases"].append({"weight": "synthetic", "kind": "edge",
                                    **case(raw[:, :, :count], weight, None, buffered=False)})
        history = raw[:, :, :4].clone()
        report["cases"].append({"weight": "synthetic", "kind": "edge_decode",
                                **case(raw[:, :, 4:], weight, history, buffered=True)})
        # Noncontiguous positive weight/state strides and a masked channel tail.
        weight_view = torch.randn(33, 1, 8, device="cuda", dtype=torch.bfloat16)[:, :, ::2]
        history_view = torch.randn(1, 4, 33, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
        report["cases"].append({"weight": "synthetic_strided", "kind": "decode",
                                **case(get_inputs(weight_view, 1), weight_view, history_view, buffered=True)})
    keys = ("state_exact", "state_layout_exact", "output_layout_exact",
            "input_weight_old_state_unchanged", "state_storage_contract")
    report["structural_checks_ok"] = all(r[k] for r in report["cases"] for k in keys)
    report["same_candidate_graph_exact"] = all(r["exact"] and r["fixed_state_pointer"] for r in report["graphs"])
    report["complete"] = True
    save()
    if not report["structural_checks_ok"] or not report["same_candidate_graph_exact"]:
        raise RuntimeError("Structural/Graph diagnostic failed; observations retained")


if __name__ == "__main__":
    main()
