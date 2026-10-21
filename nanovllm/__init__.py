from nanovllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]


def __getattr__(name):
    # The dedicated eager text path does not require the legacy Triton/Flash
    # Attention runner. Preserve the existing public import lazily.
    if name == "LLM":
        from nanovllm.llm import LLM

        return LLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
