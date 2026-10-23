"""Checkpoint byte/layout and dispatch checks; no GPU accuracy or speed claim."""

import json

import pytest
import torch
from safetensors.torch import save_file

from nanovllm.qwen38.config import Qwen38TextConfig
from nanovllm.qwen38.loader import load_weights
from nanovllm.qwen38.model import Qwen38ForCausalLM
from nanovllm.qwen38.quant import (
    FP8Linear,
    FP8MergedLinear,
    convert_fp8_linears,
    fuse_fp8_projections,
)


def construct(*, production=False):
    config = Qwen38TextConfig() if production else Qwen38TextConfig.for_testing(
        hidden_size=128, intermediate_size=256, head_dim=128,
        linear_key_head_dim=32, linear_value_head_dim=32,
    )
    exclusions = ["lm_head"]
    for index, kind in enumerate(config.layer_types):
        if kind == "linear_attention":
            exclusions.extend(f"model.layers.{index}.linear_attn.in_proj_{part}" for part in ("a", "b"))
    with torch.device("meta"):
        model = Qwen38ForCausalLM(config).bfloat16()
        convert_fp8_linears(model, tuple(exclusions))
    return model


def test_production_layout_reduces_400_to_256_fp8_linears_and_keeps_exclusions():
    model = construct(production=True)
    assert sum(isinstance(m, FP8Linear) for m in model.modules()) == 400
    assert fuse_fp8_projections(model) == 128
    assert sum(isinstance(m, FP8Linear) for m in model.modules()) == 256
    assert all(p.is_meta for p in model.parameters())
    assert isinstance(model.lm_head, torch.nn.Linear)
    for layer in model.model.layers:
        if layer.kind == "linear_attention":
            assert isinstance(layer.linear_attn.in_proj_a, torch.nn.Linear)
            assert isinstance(layer.linear_attn.in_proj_b, torch.nn.Linear)


def loaded_models(tmp_path):
    original = construct().to_empty(device="cpu")
    tensors = {}
    for index, (name, parameter) in enumerate(original.named_parameters()):
        source = name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name
        dtype = torch.bfloat16 if name.endswith(".weight_scale_inv") else parameter.dtype
        values = ((torch.arange(parameter.numel()).reshape(parameter.shape) + index) % 31 + 1) / 32
        tensors[source] = values.to(dtype)
    save_file(tensors, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "weights.safetensors" for name in tensors},
    }))
    load_weights(original, tmp_path)
    merged = construct()
    fuse_fp8_projections(merged)
    merged.to_empty(device="cpu")
    report = load_weights(merged, tmp_path)
    return original, merged, tensors, report


def test_loader_copies_every_original_weight_byte_and_scale_block(tmp_path):
    original, merged, tensors, report = loaded_models(tmp_path)
    assert report["projection_layout"] == "merged" and report["fp8_linear_modules"] == 8
    assert report["loaded_parameters"] == len(tensors) and report["dequantized"] is False
    ordinary = dict(original.named_parameters())
    merged_names = set()
    for name, module in merged.named_modules():
        if not isinstance(module, FP8MergedLinear):
            continue
        parent = name.rpartition(".")[0]
        merged_names.update((name + ".weight", name + ".weight_scale_inv"))
        for part, (start, end) in module.part_ranges().items():
            weight = ordinary[f"{parent}.{part}.weight"]
            scale = ordinary[f"{parent}.{part}.weight_scale_inv"]
            assert torch.equal(module.weight[start:end].view(torch.uint8), weight.view(torch.uint8))
            assert torch.equal(module.weight_scale_inv[start // 128:end // 128], scale)
    for name, parameter in merged.named_parameters():
        if name not in merged_names:
            assert torch.equal(parameter.view(torch.uint8), ordinary[name].view(torch.uint8))


@pytest.mark.parametrize("sizes", [(64, 128), (128, 0), (128, 129)])
def test_scale_block_boundaries_cannot_be_repacked(sizes):
    with pytest.raises(ValueError, match="128-aligned"):
        FP8MergedLinear(128, sizes, ("a", "b"), device="meta")


def test_allocated_weights_are_never_silently_discarded():
    model = construct().to_empty(device="cpu")
    before = {name: id(p) for name, p in model.named_parameters()}
    with pytest.raises(ValueError, match="unallocated"):
        fuse_fp8_projections(model)
    assert before == {name: id(p) for name, p in model.named_parameters()}


def test_one_projection_call_is_split_in_original_output_order(monkeypatch):
    module = FP8MergedLinear(128, (128, 256), ("first", "second"), device="meta")
    output = torch.arange(3 * 384).reshape(1, 3, 384).bfloat16()
    calls = []

    def simulated(self, inputs):
        calls.append((self, inputs))
        return output

    monkeypatch.setattr(FP8Linear, "forward", simulated)
    inputs = torch.zeros(1, 3, 128)
    first, second = module(inputs)
    assert calls == [(module, inputs)]
    assert torch.equal(first, output[..., :128])
    assert torch.equal(second, output[..., 128:])


def test_real_consumers_keep_prefill_continuation_and_states(monkeypatch, tmp_path):
    original, merged, _, _ = loaded_models(tmp_path)

    def cpu_projection(self, inputs):
        # A fixed CPU block oracle isolates consumer layouts from GPU algorithms.
        # Production FP8Linear continues to reject CPU execution.
        weight = self.weight.float() * self.weight_scale_inv.repeat_interleave(
            128, 0
        ).repeat_interleave(128, 1)
        return torch.cat([
            torch.nn.functional.linear(inputs.float(), block).bfloat16()
            for block in weight.split(128, dim=0)
        ], dim=-1)

    monkeypatch.setattr(FP8Linear, "forward", cpu_projection)
    left, right = original.new_state(), merged.new_state()
    for tokens in ([[1, 3, 7]], [[9]], [[11]]):
        ids = torch.tensor(tokens, dtype=torch.long)
        expected, actual = original(ids, left), merged(ids, right)
        assert torch.isfinite(actual).all() and torch.equal(actual, expected)
        assert left.length == right.length
        for a, b in zip(left.layers, right.layers):
            for name in ("key", "value", "conv", "recurrent"):
                av, bv = getattr(a, name), getattr(b, name)
                if av is None:
                    assert bv is None
                else:
                    assert torch.equal(av, bv)


def test_late_merged_scale_corruption_rejects_before_any_copy(tmp_path):
    _, merged, tensors, _ = loaded_models(tmp_path)
    name = "model.language_model.layers.1.mlp.up_proj.weight_scale_inv"
    tensors[name][-1, -1] = float("nan")
    save_file(tensors, tmp_path / "weights.safetensors")
    before = {name: p.view(torch.uint8).clone() for name, p in merged.named_parameters()}
    with pytest.raises(ValueError, match="finite and positive"):
        load_weights(merged, tmp_path)
    assert all(torch.equal(p.view(torch.uint8), before[name]) for name, p in merged.named_parameters())
