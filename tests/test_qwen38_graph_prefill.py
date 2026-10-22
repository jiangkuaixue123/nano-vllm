"""Reserving graph KV storage must not change initial prefill computation."""
import pytest
import torch

from nanovllm.qwen38.config import Qwen38TextConfig
from nanovllm.qwen38.model import Qwen38ForCausalLM


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("length", [1, 7, 65])
def test_initial_prefill_is_independent_of_reserved_decode_capacity(dtype, length):
    torch.manual_seed(17)
    model = Qwen38ForCausalLM(Qwen38TextConfig.for_testing()).to(dtype).eval()
    tokens = torch.randint(0, model.config.vocab_size, (1, length))
    compact, buffered = model.new_state(), model.new_state(capacity=256)
    a, b = model(tokens, compact), model(tokens, buffered)
    assert torch.equal(a, b)
    assert compact.length == buffered.length == length
    for left, right in zip(compact.layers, buffered.layers):
        for field in ("key", "value", "conv", "recurrent"):
            x, y = getattr(left, field), getattr(right, field)
            if x is None:
                assert y is None
            else:
                if field in ("key", "value"):
                    assert y.shape[2] == 256
                    y = y[:, :, :length]
                assert torch.equal(x, y)
