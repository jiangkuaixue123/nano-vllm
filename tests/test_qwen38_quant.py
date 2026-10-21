"""CPU structural and checkpoint tests, not evidence of CUDA FP8 execution."""

import json

import pytest
import torch
from safetensors.torch import save_file

from nanovllm.qwen38.config import Qwen38TextConfig
from nanovllm.qwen38.loader import load_weights
from nanovllm.qwen38.model import Qwen38ForCausalLM
from nanovllm.qwen38.quant import FP8Linear, convert_fp8_linears


def tiny_fp8_model():
    model = Qwen38ForCausalLM(Qwen38TextConfig.for_testing()).to(torch.bfloat16)
    exclusions = ("lm_head", "model.language_model.layers.0.linear_attn.in_proj_a",
                  "model.language_model.layers.0.linear_attn.in_proj_b")
    count = convert_fp8_linears(model, exclusions)
    return model, count


def write_fp8_checkpoint(root, model, corruption=None):
    tensors = {}
    for name, parameter in model.named_parameters():
        source_name = name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name
        dtype = torch.bfloat16 if name.endswith(".weight_scale_inv") else parameter.dtype
        tensors[source_name] = torch.full(parameter.shape, 0.25, dtype=torch.float32).to(dtype)
    scales = next(key for key in tensors if key.endswith(".weight_scale_inv"))
    weight = scales.removesuffix("_scale_inv")
    if corruption == "scale_dtype":
        tensors[scales] = tensors[scales].float()
    elif corruption == "dequantized_weight":
        tensors[weight] = tensors[weight].bfloat16()
    elif corruption == "missing_scale":
        del tensors[scales]
    elif corruption == "scale_shape":
        tensors[scales] = torch.ones(2, 2, dtype=torch.bfloat16)
    elif corruption in {"nan", "negative", "zero"}:
        tensors[scales].fill_({"nan": float("nan"), "negative": -1, "zero": 0}[corruption])
    # outside.safetensors is a valid indexed shard, regardless of its name.
    save_file(tensors, root / "outside.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: "outside.safetensors" for key in tensors},
    }))


def test_conversion_preserves_excluded_parameters_and_shape_contract():
    model, count = tiny_fp8_model()
    assert count == 13
    assert isinstance(model.lm_head, torch.nn.Linear)
    assert isinstance(model.model.layers[0].linear_attn.in_proj_a, torch.nn.Linear)
    assert isinstance(model.model.layers[0].linear_attn.in_proj_b, torch.nn.Linear)
    assert sum(isinstance(m, FP8Linear) for m in model.modules()) == count
    for module in model.modules():
        if isinstance(module, FP8Linear):
            assert module.weight.dtype == torch.float8_e4m3fn
            assert module.weight_scale_inv.dtype == torch.float32
            assert not module.weight.requires_grad


@pytest.mark.parametrize("exclusions", [(), ("model",), ("lm_head", "model.layers.0.mlp.gate_proj")])
def test_wrong_exclusions_fail_before_conversion(exclusions):
    model = Qwen38ForCausalLM(Qwen38TextConfig.for_testing()).bfloat16()
    with pytest.raises(ValueError, match="exclusions"):
        convert_fp8_linears(model, exclusions)
    assert not any(isinstance(m, FP8Linear) for m in model.modules())


def test_meta_allocation_preserves_mixed_parameter_dtypes():
    with torch.device("meta"):
        model, _ = tiny_fp8_model()
    model.to_empty(device="cpu")
    assert model.model.layers[0].mlp.gate_proj.weight.dtype == torch.float8_e4m3fn
    assert model.model.layers[0].mlp.gate_proj.weight_scale_inv.dtype == torch.float32
    assert model.lm_head.weight.dtype == torch.bfloat16


def test_fp8_loading_preserves_bytes_and_upcasts_scales_exactly(tmp_path):
    model, count = tiny_fp8_model()
    write_fp8_checkpoint(tmp_path, model)
    report = load_weights(model, tmp_path)
    assert report["dtype"] == "fp8" and report["dequantized"] is False
    assert report["backend"] == "triton_block_fp8" and report["fp8_linear_modules"] == count
    assert report["checkpoint_scale_dtype"] == "bfloat16"
    for parameter in model.parameters():
        assert torch.equal(parameter.float(), torch.full(parameter.shape, 0.25))


@pytest.mark.parametrize("corruption", ["scale_dtype", "dequantized_weight", "missing_scale",
                                        "scale_shape", "nan", "negative", "zero"])
def test_fp8_corruption_rejected_before_model_mutation(tmp_path, corruption):
    model, _ = tiny_fp8_model()
    write_fp8_checkpoint(tmp_path, model, corruption)
    before = {name: parameter.detach().view(torch.uint8).clone() for name, parameter in model.named_parameters()}
    with pytest.raises(ValueError):
        load_weights(model, tmp_path)
    assert all(torch.equal(parameter.detach().view(torch.uint8), before[name]) for name, parameter in model.named_parameters())


def test_cpu_execution_cannot_silently_dequantize():
    module = FP8Linear(128, 128)
    with pytest.raises(ValueError, match="CUDA"):
        module(torch.zeros(1, 128, dtype=torch.bfloat16))
    assert module.native_fp8_calls == 0


def test_dtype_conversion_cannot_silently_enable_bf16_linear():
    module = FP8Linear(128, 128).bfloat16()
    with pytest.raises(ValueError, match="dtype contract"):
        module(torch.zeros(1, 128, dtype=torch.bfloat16))
