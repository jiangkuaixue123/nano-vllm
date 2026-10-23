"""BF16 b/a layout, loading and mocked routing; no GPU numerical qualification."""

import json
from dataclasses import replace

import pytest
import torch
from safetensors.torch import save_file

from nanovllm.qwen38 import _flashinfer_gdn, loader
from nanovllm.qwen38.config import Qwen38TextConfig
from nanovllm.qwen38.model import FusedProjection, LayerState, Qwen38ForCausalLM
from nanovllm.qwen38.quant import (
    FP8Linear,
    convert_fp8_linears,
    fuse_bf16_gdn_ba_projections,
    fuse_fp8_projections,
)


def config(*, production=False, flashinfer=False):
    cfg = Qwen38TextConfig() if production else Qwen38TextConfig.for_testing(
        hidden_size=128, intermediate_size=256, head_dim=128,
        linear_key_head_dim=128 if flashinfer else 32,
        linear_value_head_dim=128 if flashinfer else 32,
        num_hidden_layers=3,
        layer_types=("linear_attention", "linear_attention", "full_attention"),
    )
    exclusions = ["lm_head"]
    for index, kind in enumerate(cfg.layer_types):
        if kind == "linear_attention":
            exclusions.extend(f"model.layers.{index}.linear_attn.in_proj_{part}" for part in ("a", "b"))
    return replace(cfg, weight_format="fp8", modules_to_not_convert=tuple(exclusions))


def construct(*, production=False, fp8_merged=False, ba_merged=False, flashinfer=False):
    cfg = config(production=production, flashinfer=flashinfer)
    with torch.device("meta"):
        model = Qwen38ForCausalLM(cfg).bfloat16()
        convert_fp8_linears(model, cfg.modules_to_not_convert)
        if fp8_merged:
            fuse_fp8_projections(model)
        if ba_merged:
            fuse_bf16_gdn_ba_projections(model)
    return model


@pytest.mark.parametrize("fp8_merged", [False, True])
def test_production_fusion_preserves_fp8_parameters_and_bf16_exclusions(fp8_merged):
    model = construct(production=True, fp8_merged=fp8_merged)
    fp8 = [(name, id(module.weight), id(module.weight_scale_inv))
           for name, module in model.named_modules() if isinstance(module, FP8Linear)]
    assert len(fp8) == (256 if fp8_merged else 400)
    assert fuse_bf16_gdn_ba_projections(model) == 48
    assert fp8 == [(name, id(module.weight), id(module.weight_scale_inv))
                   for name, module in model.named_modules() if isinstance(module, FP8Linear)]
    assert all(p.is_meta for p in model.parameters())
    assert model.lm_head.weight.dtype == torch.bfloat16
    for layer in model.model.layers:
        if layer.kind == "linear_attention":
            module = layer.linear_attn.in_proj_ba
            assert type(module) is FusedProjection
            assert module.names == ("in_proj_b", "in_proj_a")
            assert module.weight.shape == (96, 5120) and module.weight.dtype == torch.bfloat16
            assert not hasattr(layer.linear_attn, "in_proj_a")
            assert not hasattr(layer.linear_attn, "in_proj_b")


@pytest.mark.parametrize("bad", ["allocated", "float32", "bias", "shape", "not_fp8"])
def test_late_incompatible_layer_rejects_before_any_fusion(bad):
    model = construct()
    late = model.model.layers[1].linear_attn
    if bad == "allocated":
        late.in_proj_a.to_empty(device="cpu")
    elif bad == "float32":
        late.in_proj_a.float()
    elif bad == "bias":
        late.in_proj_a.bias = torch.nn.Parameter(torch.empty(late.in_proj_a.out_features, device="meta"))
    elif bad == "shape":
        late.in_proj_a = torch.nn.Linear(128, 7, bias=False, device="meta", dtype=torch.bfloat16)
    else:
        late.in_proj_qkv = torch.nn.Linear(128, late.key_dim * 2 + late.value_dim, device="meta")
    before = {name: id(parameter) for name, parameter in model.named_parameters()}
    with pytest.raises(ValueError, match="unallocated excluded linears after FP8 conversion"):
        fuse_bf16_gdn_ba_projections(model)
    assert before == {name: id(parameter) for name, parameter in model.named_parameters()}
    assert not hasattr(model.model.layers[0].linear_attn, "in_proj_ba")


def write_checkpoint(root, tensors=None):
    if tensors is None:
        model = construct().to_empty(device="cpu")
        tensors = {}
        for index, (name, parameter) in enumerate(model.named_parameters()):
            key = name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name
            dtype = torch.bfloat16 if name.endswith(".weight_scale_inv") else parameter.dtype
            values = ((torch.arange(parameter.numel()).reshape(parameter.shape) + index) % 31 + 1) / 32
            tensors[key] = values.to(dtype)
    save_file(tensors, root / "weights.safetensors")
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {name: "weights.safetensors" for name in tensors},
    }))
    return tensors


@pytest.mark.parametrize("fp8_merged", [False, True])
def test_loader_keeps_every_checkpoint_byte_and_scale(fp8_merged, tmp_path, monkeypatch):
    tensors = write_checkpoint(tmp_path)
    original = construct(fp8_merged=fp8_merged).to_empty(device="cpu")
    merged = construct(fp8_merged=fp8_merged, ba_merged=True).to_empty(device="cpu")
    original_report = loader.load_weights(original, tmp_path)
    # Report the constructed layout rather than a later environment change.
    monkeypatch.setenv("NANOVLLM_GDN_BA_PROJECTIONS", "separate")
    report = loader.load_weights(merged, tmp_path)
    assert original_report["gdn_ba_projection_layout"] == "separate"
    assert report["gdn_ba_projection_layout"] == "merged"
    assert report["loaded_parameters"] == len(tensors) == original_report["loaded_parameters"]
    assert report["fp8_linear_modules"] == original_report["fp8_linear_modules"]
    ordinary = dict(original.named_parameters())
    merged_names = set()
    for name, module in merged.named_modules():
        if not isinstance(module, FusedProjection):
            continue
        merged_names.add(name + ".weight")
        parent = name.rpartition(".")[0]
        for part, (start, end) in module.part_ranges().items():
            assert torch.equal(module.weight[start:end].view(torch.uint8),
                               ordinary[f"{parent}.{part}.weight"].view(torch.uint8))
    for name, parameter in merged.named_parameters():
        if name not in merged_names:
            assert torch.equal(parameter.view(torch.uint8), ordinary[name].view(torch.uint8))


@pytest.mark.parametrize("bad", ["missing", "shape", "dtype"])
def test_bad_ba_checkpoint_is_rejected_before_any_parameter_copy(bad, tmp_path):
    tensors = write_checkpoint(tmp_path)
    key = "model.language_model.layers.1.linear_attn.in_proj_a.weight"
    if bad == "missing":
        del tensors[key]
    elif bad == "shape":
        tensors[key] = tensors[key][:-1].clone()
    else:
        tensors[key] = tensors[key].float()
    write_checkpoint(tmp_path, tensors)
    model = construct(fp8_merged=True, ba_merged=True).to_empty(device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    before = {name: p.view(torch.uint8).clone() for name, p in model.named_parameters()}
    with pytest.raises(ValueError):
        loader.load_weights(model, tmp_path)
    assert all(torch.equal(p.view(torch.uint8), before[name]) for name, p in model.named_parameters())


def test_single_bf16_linear_call_returns_b_then_a_views(monkeypatch):
    module = FusedProjection(128, (48, 48), ("in_proj_b", "in_proj_a"),
                             device="meta", dtype=torch.bfloat16)
    output = torch.arange(3 * 96).reshape(1, 3, 96).bfloat16()
    calls = []

    def linear(x, weight):
        calls.append((x, weight))
        return output

    monkeypatch.setattr(torch.nn.functional, "linear", linear)
    inputs = torch.zeros(1, 3, 128, dtype=torch.bfloat16)
    b, a = module(inputs)
    assert calls == [(inputs, module.weight)]
    assert torch.equal(b, output[..., :48]) and torch.equal(a, output[..., 48:])
    assert b.stride(-2) == a.stride(-2) == 96


@pytest.mark.parametrize("fp8_merged", [False, True])
def test_actual_deltanet_routes_ba_once_for_prefill_and_buffered_decode(monkeypatch, fp8_merged):
    monkeypatch.setenv("NANOVLLM_GDN", "flashinfer")
    model = construct(fp8_merged=fp8_merged, ba_merged=True, flashinfer=True).to_empty(device="cpu")
    layer = model.model.layers[0].linear_attn
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.zero_()
        layer.in_proj_ba.weight[:layer.config.linear_num_value_heads].fill_(1 / 128)
        layer.in_proj_ba.weight[layer.config.linear_num_value_heads:].fill_(-2 / 128)
    calls = []
    hook = layer.in_proj_ba.register_forward_hook(lambda _module, _args, result: calls.append(result))
    monkeypatch.setattr(FP8Linear, "forward", lambda module, x: x.new_zeros(*x.shape[:-1], module.out_features))
    state = LayerState(capacity=8,
                       conv=torch.zeros(1, 2 * layer.key_dim + layer.value_dim, 4, dtype=torch.bfloat16),
                       recurrent=torch.zeros(1, layer.config.linear_num_value_heads, 128, 128))
    pointers = (state.conv.data_ptr(), state.recurrent.data_ptr())

    def prefill(q, k, v, decay, beta, initial):
        assert initial is None
        assert torch.equal(beta, calls[-1][0].sigmoid())
        return v, torch.zeros_like(state.recurrent)

    def decode(q, k, v, initial, a_log, a, dt_bias, b):
        assert initial is state.recurrent
        assert torch.all(b == 1) and torch.all(a == -2)
        initial.add_(1)
        return v, initial

    monkeypatch.setattr(_flashinfer_gdn, "prefill", prefill)
    monkeypatch.setattr(_flashinfer_gdn, "decode", decode)
    for count in (3, 1, 1):
        result = layer(torch.ones(1, count, 128, dtype=torch.bfloat16), state)
        assert result.shape == (1, count, 128)
        assert pointers == (state.conv.data_ptr(), state.recurrent.data_ptr())
    hook.remove()
    assert len(calls) == 3 and torch.all(state.recurrent == 2)
    assert not state.fresh


@pytest.mark.parametrize("fp8_layout", ["separate", "merged"])
@pytest.mark.parametrize("ba_layout", [None, "separate", "merged", "typo"])
def test_loader_option_is_opt_in_and_independent(monkeypatch, tmp_path, fp8_layout, ba_layout):
    cfg = config()
    monkeypatch.setattr(Qwen38TextConfig, "from_dict", lambda *args, **kwargs: cfg)
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (9, 0))
    monkeypatch.setenv("NANOVLLM_FP8_PROJECTIONS", fp8_layout)
    if ba_layout is None:
        monkeypatch.delenv("NANOVLLM_GDN_BA_PROJECTIONS", raising=False)
    else:
        monkeypatch.setenv("NANOVLLM_GDN_BA_PROJECTIONS", ba_layout)
    allocate = torch.nn.Module.to_empty
    monkeypatch.setattr(torch.nn.Module, "to_empty", lambda self, **kwargs: allocate(self, device="cpu"))
    monkeypatch.setattr(loader, "load_weights", lambda *args: {})
    if ba_layout == "typo":
        with pytest.raises(ValueError, match="Unsupported NANOVLLM_GDN_BA_PROJECTIONS"):
            loader.load_checkpoint(tmp_path)
    else:
        model, _ = loader.load_checkpoint(tmp_path)
        layer = model.model.layers[0].linear_attn
        assert hasattr(layer, "in_proj_ba") == (ba_layout == "merged")
        assert hasattr(layer, "in_proj_qkvz") == (fp8_layout == "merged")
        # The same flag cannot silently change ordinary BF16 checkpoint loading.
        monkeypatch.setenv("NANOVLLM_GDN_BA_PROJECTIONS", "merged")
        monkeypatch.setenv("NANOVLLM_FP8_PROJECTIONS", "separate")
        monkeypatch.setattr(Qwen38TextConfig, "from_dict", lambda *args, **kwargs: replace(cfg, weight_format="bfloat16"))
        with pytest.raises(ValueError, match="require an FP8 checkpoint"):
            loader.load_checkpoint(tmp_path)
