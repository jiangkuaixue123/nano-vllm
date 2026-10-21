"""Configuration boundary for the dedicated BF16 and block-FP8 text paths."""

from dataclasses import dataclass, fields

MODEL_ID = "Qwen/Qwen3.8-27B"
FP8_QUANTIZATION = {
    "activation_scheme": "dynamic", "fmt": "e4m3", "quant_method": "fp8",
    "weight_block_size": [128, 128],
}


@dataclass(frozen=True)
class Qwen38TextConfig:
    hidden_size: int = 5120
    intermediate_size: int = 17408
    vocab_size: int = 248320
    num_hidden_layers: int = 64
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    layer_types: tuple[str, ...] = (
        ("linear_attention",) * 3 + ("full_attention",)
    ) * 16
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25
    max_position_embeddings: int = 262144
    eos_token_id: int = 248044
    bos_token_id: int = 248044
    weight_format: str = "bfloat16"
    modules_to_not_convert: tuple[str, ...] = ()
    _test_only: bool = False

    def __post_init__(self):
        if self.weight_format not in {"bfloat16", "fp8"}:
            raise ValueError("Unsupported weight format")
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in INTEGER_FIELDS and (type(value) is not int or value <= 0):
                raise ValueError(
                    "Model dimensions and context limit must be positive integers"
                )
        if len(self.layer_types) != self.num_hidden_layers or set(self.layer_types) - {
            "linear_attention",
            "full_attention",
        }:
            raise ValueError("Unsupported layer layout")
        if (
            self.num_attention_heads % self.num_key_value_heads
            or self.linear_num_value_heads % self.linear_num_key_heads
        ):
            raise ValueError(
                "Query/value head counts must be divisible by key head counts"
            )
        rotary = self.head_dim * self.partial_rotary_factor
        if (
            rotary != int(rotary)
            or rotary <= 0
            or rotary > self.head_dim
            or int(rotary) % 2
        ):
            raise ValueError(
                "Partial rotary dimension must be positive, even and fit the head"
            )
        if self.rms_norm_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("Invalid normalization or rotary parameters")
        if not self._test_only:
            for name, expected in PRODUCTION_SIGNATURE.items():
                if getattr(self, name) != expected:
                    raise ValueError("Only the Qwen3.8-27B text signature is supported")

    @classmethod
    def from_dict(cls, outer: dict, strict: bool = True):
        if not strict:
            raise ValueError(
                "Use for_testing() explicitly for synthetic tiny configurations"
            )
        text = outer.get("text_config", {})
        if (
            outer.get("model_type") != "qwen3_5"
            or outer.get("architectures") != ["Qwen3_5ForConditionalGeneration"]
            or text.get("quantization_config")
            or outer.get("tie_word_embeddings") is not False
        ):
            raise ValueError("Expected the supported Qwen3.8-27B checkpoint")
        quant = outer.get("quantization_config")
        quant_args = {}
        if quant is not None:
            if not isinstance(quant, dict) or set(quant) != set(FP8_QUANTIZATION) | {"modules_to_not_convert"}:
                raise ValueError("Expected the dynamic block-FP8 quantization configuration")
            if any(quant.get(key) != value for key, value in FP8_QUANTIZATION.items()):
                raise ValueError("Unsupported FP8 quantization semantics")
            excluded = quant["modules_to_not_convert"]
            if not isinstance(excluded, list) or not excluded or any(not isinstance(x, str) or not x for x in excluded):
                raise ValueError("FP8 requires explicit module exclusions")
            quant_args = {"weight_format": "fp8", "modules_to_not_convert": tuple(excluded)}
        for name, expected in TEXT_SEMANTICS.items():
            if text.get(name) != expected:
                raise ValueError("Unsupported text execution semantics")
        rope = text.get("rope_parameters", {})
        if (
            rope.get("rope_type") != "default"
            or rope.get("mrope_interleaved") is not True
            or rope.get("mrope_section") != [11, 11, 10]
            or rope.get("partial_rotary_factor") != text.get("partial_rotary_factor")
        ):
            raise ValueError("Unsupported rotary configuration")
        args = {
            name: text[name] for name in PRODUCTION_SIGNATURE if name != "rope_theta"
        }
        args["layer_types"] = tuple(args["layer_types"])
        args["rope_theta"] = rope["rope_theta"]
        return cls(**args, **quant_args)

    @classmethod
    def for_testing(cls, **overrides):
        """Explicit synthetic shapes only; never use to admit a production checkpoint."""
        args = {
            "hidden_size": 16,
            "intermediate_size": 32,
            "vocab_size": 97,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "layer_types": ("linear_attention", "full_attention"),
            "partial_rotary_factor": 0.5,
            "rope_theta": 10000.0,
            "max_position_embeddings": 256,
            "eos_token_id": 0,
            "bos_token_id": 0,
        }
        args.update(overrides)
        args["_test_only"] = True
        return cls(**args)


PRODUCTION_PATTERN = (("linear_attention",) * 3 + ("full_attention",)) * 16
INTEGER_FIELDS = {
    "hidden_size",
    "intermediate_size",
    "vocab_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "max_position_embeddings",
}
PRODUCTION_SIGNATURE = {
    field.name: (PRODUCTION_PATTERN if field.name == "layer_types" else field.default)
    for field in fields(Qwen38TextConfig)
    if field.name not in {"_test_only", "weight_format", "modules_to_not_convert"}
}
TEXT_SEMANTICS = {
    "model_type": "qwen3_5_text",
    "dtype": "bfloat16",
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_output_gate": True,
    "output_gate_type": "swish",
    "mamba_ssm_dtype": "float32",
    "tie_word_embeddings": False,
    "pad_token_id": None,
}
