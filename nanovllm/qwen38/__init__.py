"""Dedicated Qwen3.8 text implementation; optional Torch imports stay lazy."""

from .config import Qwen38TextConfig

__all__ = ["HybridState", "Qwen38ForCausalLM", "Qwen38TextConfig"]


def __getattr__(name):
    if name in {"Qwen38ForCausalLM", "HybridState"}:
        from .model import HybridState, Qwen38ForCausalLM

        return {"Qwen38ForCausalLM": Qwen38ForCausalLM, "HybridState": HybridState}[
            name
        ]
    raise AttributeError(name)
