"""Synthetic config boundary tests, without downloading model artifacts."""

from dataclasses import asdict

import pytest

from nanovllm.qwen38.config import FP8_QUANTIZATION, TEXT_SEMANTICS, Qwen38TextConfig


def production_config():
    text = asdict(Qwen38TextConfig())
    text.pop("_test_only")
    theta = text.pop("rope_theta")
    text["layer_types"] = list(text["layer_types"])
    text.update(TEXT_SEMANTICS)
    text["rope_parameters"] = {
        "rope_type": "default",
        "rope_theta": theta,
        "partial_rotary_factor": text["partial_rotary_factor"],
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
    }
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": text,
        "tie_word_embeddings": False,
    }


def test_exact_production_signature():
    config = Qwen38TextConfig.from_dict(production_config())
    assert config == Qwen38TextConfig()
    assert config.layer_types.count("linear_attention") == 48
    assert config.layer_types.count("full_attention") == 16
    assert not config._test_only


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 4096),
        ("vocab_size", 100),
        ("attention_bias", True),
        ("hidden_act", "gelu"),
        ("dtype", "float16"),
        ("mamba_ssm_dtype", "bfloat16"),
        ("layer_types", ["full_attention"] * 64),
        ("output_gate_type", "sigmoid"),
    ],
)
def test_reject_other_model_shapes_and_semantics(field, value):
    config = production_config()
    config["text_config"][field] = value
    with pytest.raises(ValueError):
        Qwen38TextConfig.from_dict(config)


@pytest.mark.parametrize("owner", ["outer", "text"])
def test_no_quantized_checkpoint(owner):
    config = production_config()
    target = config if owner == "outer" else config["text_config"]
    target["quantization_config"] = {"quant_method": "fp8"}
    with pytest.raises(ValueError):
        Qwen38TextConfig.from_dict(config)


def test_tiny_factory_is_explicit_not_a_checkpoint_fallback():
    config = Qwen38TextConfig.for_testing()
    assert config._test_only and config.hidden_size == 16
    with pytest.raises(ValueError):
        Qwen38TextConfig(hidden_size=16)
    with pytest.raises(ValueError, match="for_testing"):
        Qwen38TextConfig.from_dict(production_config(), strict=False)


def test_dynamic_block_fp8_config():
    raw = production_config()
    raw["quantization_config"] = {**FP8_QUANTIZATION, "modules_to_not_convert": ["lm_head"]}
    config = Qwen38TextConfig.from_dict(raw)
    assert config.weight_format == "fp8"
    assert config.modules_to_not_convert == ("lm_head",)


@pytest.mark.parametrize("key,value", [
    ("activation_scheme", "static"), ("weight_block_size", [64, 128]),
    ("fmt", "e5m2"), ("dequantize", True), ("modules_to_not_convert", []),
])
def test_other_fp8_semantics_rejected(key, value):
    raw = production_config()
    raw["quantization_config"] = {**FP8_QUANTIZATION, "modules_to_not_convert": ["lm_head"], key: value}
    with pytest.raises(ValueError):
        Qwen38TextConfig.from_dict(raw)


@pytest.mark.parametrize(
    "changes",
    [
        {"num_attention_heads": 3},
        {"linear_num_value_heads": 3},
        {"head_dim": 7},
        {"partial_rotary_factor": 2},
        {"num_hidden_layers": 4},
        {"linear_conv_kernel_dim": 0},
        {"vocab_size": True},
    ],
)
def test_invalid_tiny_shapes_fail_before_model_allocation(changes):
    with pytest.raises(ValueError):
        Qwen38TextConfig.for_testing(**changes)
