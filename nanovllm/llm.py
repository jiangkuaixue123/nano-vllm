"""Public dispatch without importing the optional legacy CUDA stack eagerly."""

import json
from pathlib import Path


class LLM:
    def __new__(cls, model, **kwargs):
        config_file = Path(model) / "config.json"
        config = json.loads(config_file.read_bytes()) if config_file.is_file() else {}
        if config.get("model_type") == "qwen3_5":
            from nanovllm.qwen38.engine import DedicatedEngine

            return DedicatedEngine(model, **kwargs)
        from nanovllm.engine.llm_engine import LLMEngine

        return LLMEngine(model, **kwargs)
