"""Eager, single-device request ownership for an already loaded native model.

The model owns numerical execution. This runner owns request identity, bounds,
state lifetime and text generation; it never calls Transformers model APIs.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral


@dataclass
class _Request:
    state: object
    length: int = 0


class RequestRunner:
    def __init__(
        self, model, tokenizer=None, *, max_context_tokens: int,
        max_active_requests: int = 1, device="cuda:0", tensor_factory=None,
        vocab_size: int | None = None,
    ):
        for name, value in (
            ("max_context_tokens", max_context_tokens),
            ("max_active_requests", max_active_requests),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if vocab_size is not None and (type(vocab_size) is not int or vocab_size <= 0):
            raise ValueError("vocab_size must be a positive integer")
        self.model = model
        self.tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens
        self.max_active_requests = max_active_requests
        self.device = device
        self.vocab_size = vocab_size
        self._tensor_factory = tensor_factory
        self._requests: dict[str, _Request] = {}

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(self._requests)

    def _tokens(self, token_ids, *, allow_empty=False) -> list[int]:
        if not isinstance(token_ids, (list, tuple)) or (not token_ids and not allow_empty):
            raise ValueError("token IDs must be a nonempty list or tuple")
        if any(
            isinstance(token, bool) or not isinstance(token, Integral) or token < 0
            or (self.vocab_size is not None and token >= self.vocab_size)
            for token in token_ids
        ):
            raise ValueError("token IDs must be nonnegative integers within the vocabulary")
        return [int(token) for token in token_ids]

    @staticmethod
    def _request_id(request_id):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")

    def _input(self, tokens):
        if self._tensor_factory is not None:
            return self._tensor_factory([tokens]), nullcontext()
        import torch

        return torch.tensor([tokens], dtype=torch.long, device=self.device), torch.inference_mode()

    def prefill(self, request_id: str, token_ids):
        """Create one private state and consume its first tokens, without sampling."""
        self._request_id(request_id)
        if request_id in self._requests:
            raise ValueError("request_id is already active")
        tokens = self._tokens(token_ids)
        if len(tokens) > self.max_context_tokens:
            raise ValueError("prompt exceeds max_context_tokens")
        if len(self._requests) >= self.max_active_requests:
            raise ValueError("active request capacity exhausted")
        self._requests[request_id] = _Request(self.model.new_state())
        return self.append(request_id, tokens)

    def append(self, request_id: str, token_ids):
        """Consume more tokens; any failure invalidates and releases this request."""
        self._request_id(request_id)
        if request_id not in self._requests:
            raise KeyError("unknown request_id")
        try:
            request = self._requests[request_id]
            tokens = self._tokens(token_ids)
            if request.length + len(tokens) > self.max_context_tokens:
                raise ValueError("request exceeds max_context_tokens")
            inputs, context = self._input(tokens)
            with context:
                logits = self.model(inputs, state=request.state)
            if (
                len(logits.shape) != 3 or tuple(logits.shape[:2]) != (1, len(tokens))
                or logits.shape[2] <= 0
                or (self.vocab_size is not None and logits.shape[2] != self.vocab_size)
            ):
                raise ValueError("native model must return logits shaped [1, tokens, vocabulary]")
            if self._tensor_factory is None:
                import torch

                if not torch.isfinite(logits).all().item():
                    raise ValueError("native model returned nonfinite logits")
            request.length += len(tokens)
            return logits
        except BaseException:
            self.release(request_id)
            raise

    def release(self, request_id: str) -> bool:
        """Drop only the named request's state; repeated release is harmless."""
        return self._requests.pop(request_id, None) is not None

    def reset(self, request_id: str):
        """Replace an active request's state with an empty state for later append."""
        if request_id not in self._requests:
            raise KeyError("unknown request_id")
        self.release(request_id)
        self._requests[request_id] = _Request(self.model.new_state())

    def trace_teacher_forced(self, prompt_ids, continuation_ids, *, request_id="trace"):
        """Return one detached [vocabulary] row per supplied continuation token.

        Each row predicts the next fixed token. The final continuation token is
        scored, not consumed. Ownership is temporary and always released.
        """
        prompt = self._tokens(prompt_ids)
        continuation = self._tokens(continuation_ids)
        if len(prompt) + len(continuation) > self.max_context_tokens:
            raise ValueError("teacher-forced sequence exceeds max_context_tokens")
        logits = self.prefill(request_id, prompt)
        try:
            rows = [logits[0, -1].detach().clone()]
            for token in continuation[:-1]:
                logits = self.append(request_id, [token])
                rows.append(logits[0, -1].detach().clone())
            return rows
        finally:
            self.release(request_id)

    def _prompt(self, prompt):
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("text prompts require a tokenizer")
            return self._tokens(self.tokenizer.encode(prompt, add_special_tokens=False))
        return self._tokens(prompt)

    def generate(self, prompt, *, max_tokens: int, eos_token_ids=(), request_id="generate"):
        """Deterministic greedy generation; max context includes generated tokens."""
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        tokens = self._prompt(prompt)
        eos = set(self._tokens(list(eos_token_ids), allow_empty=True))
        if len(tokens) + max_tokens > self.max_context_tokens:
            raise ValueError("prompt plus max_tokens exceeds max_context_tokens")
        logits = self.prefill(request_id, tokens)
        generated = []
        finish_reason = "length"
        try:
            for index in range(max_tokens):
                token = int(logits[0, -1].argmax().item())
                generated.append(token)
                if token in eos:
                    finish_reason = "eos"
                    break
                if index + 1 < max_tokens:
                    logits = self.append(request_id, [token])
            text = (
                self.tokenizer.decode(generated, skip_special_tokens=True)
                if self.tokenizer is not None else None
            )
            return {"token_ids": generated, "text": text, "finish_reason": finish_reason}
        finally:
            self.release(request_id)

    def chat(self, messages, *, max_tokens: int, eos_token_ids=(), request_id="chat",
             template_kwargs=None):
        """Apply the installed tokenizer's chat template, then run native generation."""
        if self.tokenizer is None:
            raise ValueError("chat requires a tokenizer")
        options = dict(template_kwargs or {})
        if "tokenize" in options or "return_tensors" in options:
            raise ValueError("chat controls tokenization and tensor construction")
        options.setdefault("add_generation_prompt", True)
        tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, **options)
        return self.generate(
            tokens, max_tokens=max_tokens, eos_token_ids=eos_token_ids, request_id=request_id,
        )
