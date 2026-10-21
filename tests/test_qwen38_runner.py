"""CPU contracts for native request ownership, without importing GPU packages."""

import importlib.util
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "nanovllm/qwen38/runner.py"
SPEC = importlib.util.spec_from_file_location("native_request_runner", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
RequestRunner = MODULE.RequestRunner


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class Row:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def clone(self):
        return Row(self.values.copy())

    def argmax(self):
        return Scalar(max(range(len(self.values)), key=self.values.__getitem__))


class Logits:
    def __init__(self, rows, vocabulary):
        self.rows = rows
        self.shape = (1, len(rows), vocabulary)

    def __getitem__(self, key):
        batch, position = key
        assert batch == 0
        return self.rows[position]


class NativeModel:
    def __init__(self):
        self.calls = []
        self.fail = False
        self.bad_shape = False

    def new_state(self):
        return {"tokens": []}

    def __call__(self, inputs, *, state):
        assert len(inputs) == 1
        rows = []
        self.calls.append((state, inputs[0].copy()))
        for token in inputs[0]:
            state["tokens"].append(token)
            if self.fail:
                raise RuntimeError("model failed after mutating state")
            prediction = sum(state["tokens"]) % 8
            values = [0.0] * 8
            values[prediction] = 1.0
            rows.append(Row(values))
        result = Logits(rows, 8)
        if self.bad_shape:
            result.shape = (1, 0, 8)
        return result


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [1, 2]

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is True
        return ",".join(map(str, tokens))

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        assert messages == [{"role": "user", "content": "hello"}]
        assert tokenize and add_generation_prompt
        assert kwargs == {"enable_thinking": False}
        return [1, 2]


def runner(**kwargs):
    return RequestRunner(
        NativeModel(), tensor_factory=lambda value: value, max_context_tokens=12,
        vocab_size=8, **kwargs,
    )


def test_interleaved_request_states_are_isolated_and_reset_is_fresh():
    engine = runner(max_active_requests=2)
    assert engine.prefill("first", [1, 2])[0, -1].argmax().item() == 3
    assert engine.prefill("second", [4])[0, -1].argmax().item() == 4
    assert engine.append("first", [3])[0, -1].argmax().item() == 6
    assert engine.append("second", [2])[0, -1].argmax().item() == 6
    assert engine.model.calls[0][0] is not engine.model.calls[1][0]
    engine.reset("first")
    assert engine.append("first", [1])[0, -1].argmax().item() == 1
    assert engine.append("second", [1])[0, -1].argmax().item() == 7
    assert engine.release("first") is True
    assert engine.release("first") is False
    assert engine.active_request_ids == ("second",)


def test_capacity_duplicate_and_unknown_ids_do_not_clobber_existing_state():
    engine = runner()
    engine.prefill("active", [1])
    with pytest.raises(ValueError, match="already active"):
        engine.prefill("active", [4])
    with pytest.raises(ValueError, match="capacity"):
        engine.prefill("other", [4])
    with pytest.raises(KeyError):
        engine.append("unknown", [1])
    with pytest.raises(KeyError):
        engine.reset("unknown")
    assert engine.append("active", [2])[0, -1].argmax().item() == 3


@pytest.mark.parametrize("tokens", [[], [True], [-1], [8], [1.0], "text"])
def test_invalid_tokens_leave_no_partial_request(tokens):
    engine = runner()
    with pytest.raises(ValueError):
        engine.prefill("invalid", tokens)
    assert not engine.active_request_ids


def test_bounds_reject_before_model_execution_and_release_invalidated_request():
    engine = runner()
    with pytest.raises(ValueError, match="prompt exceeds"):
        engine.prefill("a", [1] * 13)
    engine.prefill("a", [1] * 12)
    with pytest.raises(ValueError, match="request exceeds"):
        engine.append("a", [1])
    assert len(engine.model.calls) == 1
    assert not engine.active_request_ids
    with pytest.raises(ValueError, match="teacher-forced"):
        engine.trace_teacher_forced([1] * 12, [2])
    with pytest.raises(ValueError, match="prompt plus"):
        engine.generate([1] * 12, max_tokens=1)


@pytest.mark.parametrize("failure", ["fail", "bad_shape"])
def test_model_failure_releases_only_its_request(failure):
    engine = runner(max_active_requests=2)
    engine.prefill("survivor", [2])
    engine.prefill("failing", [1])
    setattr(engine.model, failure, True)
    with pytest.raises((RuntimeError, ValueError)):
        engine.append("failing", [3])
    setattr(engine.model, failure, False)
    assert engine.active_request_ids == ("survivor",)
    assert engine.append("survivor", [1])[0, -1].argmax().item() == 3
    engine.prefill("failing", [1])
    assert engine.model.calls[-1][0]["tokens"] == [1]


def test_teacher_forced_trace_uses_fixed_tokens_and_always_releases():
    engine = runner()
    rows = engine.trace_teacher_forced([1, 2], [4, 5, 6])
    assert [row.argmax().item() for row in rows] == [3, 7, 4]
    assert [tokens for _, tokens in engine.model.calls] == [[1, 2], [4], [5]]
    assert not engine.active_request_ids
    engine.model.fail = True
    with pytest.raises(RuntimeError):
        engine.trace_teacher_forced([1], [2])
    assert not engine.active_request_ids


def test_generation_is_greedy_with_eos_and_length_finish_reasons():
    engine = runner(tokenizer=Tokenizer())
    result = engine.generate("hello", max_tokens=4, eos_token_ids={6})
    assert result == {"token_ids": [3, 6], "text": "3,6", "finish_reason": "eos"}
    assert [tokens for _, tokens in engine.model.calls] == [[1, 2], [3]]
    assert not engine.active_request_ids
    result = engine.generate([1, 2], max_tokens=1)
    assert result["token_ids"] == [3]
    assert result["finish_reason"] == "length"
    assert not engine.active_request_ids


def test_chat_applies_tokenizer_template_and_preserves_generation_contract():
    engine = runner(tokenizer=Tokenizer())
    result = engine.chat(
        [{"role": "user", "content": "hello"}], max_tokens=2,
        eos_token_ids={3}, template_kwargs={"enable_thinking": False},
    )
    assert result["token_ids"] == [3]
    assert not engine.active_request_ids
    with pytest.raises(ValueError, match="controls tokenization"):
        engine.chat([], max_tokens=1, template_kwargs={"return_tensors": "pt"})


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_generation_limit_does_not_allocate_state(limit):
    engine = runner()
    with pytest.raises(ValueError, match="max_tokens"):
        engine.generate([1], max_tokens=limit)
    assert not engine.active_request_ids


def test_duplicate_high_level_request_does_not_release_callers_state():
    engine = runner()
    engine.prefill("same", [1])
    with pytest.raises(ValueError, match="already active"):
        engine.generate([2], max_tokens=1, request_id="same")
    with pytest.raises(ValueError, match="already active"):
        engine.trace_teacher_forced([2], [3], request_id="same")
    assert engine.append("same", [2])[0, -1].argmax().item() == 3


def test_decode_exception_and_interrupt_release_temporary_requests():
    class BrokenTokenizer(Tokenizer):
        def decode(self, tokens, **kwargs):
            raise RuntimeError("decode failed")

    engine = runner(tokenizer=BrokenTokenizer())
    with pytest.raises(RuntimeError, match="decode failed"):
        engine.generate([1], max_tokens=1)
    assert not engine.active_request_ids

    class InterruptedModel(NativeModel):
        def __call__(self, inputs, *, state):
            state["tokens"].extend(inputs[0])
            raise KeyboardInterrupt

    engine.model = InterruptedModel()
    with pytest.raises(KeyboardInterrupt):
        engine.prefill("interrupted", [1])
    assert not engine.active_request_ids
