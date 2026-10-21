"""Bounded request ownership and serialized native execution for public serving."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import uuid
from collections import Counter, deque
from pathlib import Path


class CapacityError(ValueError):
    pass


class EngineClosedError(RuntimeError):
    pass


class DedicatedEngine:
    """Greedy-only text engine; concurrent callers share no request state.

    Lock order is model lock then registry lock. Admission and cancellation use
    only the registry lock. Cancellation becomes effective at the next model
    call boundary; an in-flight CUDA operation cannot be preempted.
    """

    def __init__(self, model, *, max_context_tokens=4096, max_active_requests=4):
        from transformers import AutoTokenizer

        from .loader import load_checkpoint
        from .runner import RequestRunner

        native, report = load_checkpoint(model)
        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        generation = json.loads((Path(model) / "generation_config.json").read_bytes())
        eos = generation["eos_token_id"]
        if type(eos) is int:
            eos = [eos]
        if max_context_tokens > native.config.max_position_embeddings:
            raise ValueError("max_context_tokens exceeds model context limit")
        runner = RequestRunner(native, tokenizer, max_context_tokens=max_context_tokens,
                               max_active_requests=max_active_requests, vocab_size=native.config.vocab_size)
        model_id = "Qwen/Qwen3.8-27B-FP8" if report["dtype"] == "fp8" else "Qwen/Qwen3.8-27B"
        self._initialize(runner, eos, model_id, report)

    @classmethod
    def from_runner_for_testing(cls, runner, *, eos_token_ids=(), model_id="test-model"):
        engine = cls.__new__(cls)
        engine._initialize(runner, eos_token_ids, model_id, {"dtype": "test_only"})
        return engine

    def _initialize(self, runner, eos_token_ids, model_id, report):
        self.runner = runner
        self.model_id = model_id
        self.loader_report = report
        self.eos_token_ids = frozenset(runner._tokens(list(eos_token_ids), allow_empty=True))
        self._model_lock = threading.Lock()
        self._registry_lock = threading.Lock()
        self._active = {}
        self._closed = False
        self._terminal = deque(maxlen=128)
        self._counts = Counter()
        self._state_count = 0
        self._state_bytes = 0

    def submit(self, prompt, *, max_tokens, request_id=None, ignore_eos=False):
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if type(ignore_eos) is not bool:
            raise ValueError("ignore_eos must be boolean")
        tokens = self.runner._prompt(prompt)
        if len(tokens) + max_tokens > self.runner.max_context_tokens:
            raise ValueError("prompt plus max_tokens exceeds max_context_tokens")
        request_id = "cmpl-" + uuid.uuid4().hex if request_id is None else request_id
        if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", request_id):
            raise ValueError("request_id must contain 1..128 safe identifier characters")
        with self._registry_lock:
            if self._closed:
                raise EngineClosedError("engine is closed")
            if request_id in self._active:
                raise ValueError("request_id is already active")
            if len(self._active) >= self.runner.max_active_requests:
                raise CapacityError("active request capacity exhausted")
            request = Generation(self, request_id, tokens, max_tokens, ignore_eos)
            self._active[request_id] = request
        return request

    def cancel(self, request_id, reason="cancelled"):
        if reason not in {"cancelled", "disconnected", "shutdown"}:
            raise ValueError("unsupported cancellation reason")
        with self._registry_lock:
            request = self._active.get(request_id)
            if request is None:
                return False
            if not request.cancelled.is_set():
                request.cancel_reason = reason
                request.cancelled.set()
            return True

    def _finish(self, request, reason):
        # Caller holds model lock; no socket I/O occurs while holding it.
        if request.finish_reason is not None:
            return
        self.runner.release(request.request_id)
        self._refresh_ownership()
        with self._registry_lock:
            # Terminal publication and cancellation acknowledgement share this
            # lock: a successful cancel cannot race into a normal completion.
            if reason in {"stop", "length", "cancelled"} and request.cancelled.is_set():
                reason = request.cancel_reason
            request.finish_reason = reason
            self._active.pop(request.request_id, None)
            self._counts[reason] += 1
            self._terminal.append({"request_id": request.request_id, "finish_reason": reason,
                                   "token_count": len(request.token_ids), "prefilled": request.started})

    def _refresh_ownership(self):
        # Model lock is held. Tensor metadata requires no device synchronization.
        state_bytes = 0
        for request in self.runner._requests.values():
            for layer in getattr(request.state, "layers", ()):
                for name in ("key", "value", "conv", "recurrent"):
                    tensor = getattr(layer, name, None)
                    if tensor is not None:
                        state_bytes += tensor.numel() * tensor.element_size()
        with self._registry_lock:
            self._state_count = len(self.runner.active_request_ids)
            self._state_bytes = state_bytes

    def stats(self):
        with self._registry_lock:
            return {"active_request_ids": list(self._active), "active_requests": len(self._active),
                    "capacity": self.runner.max_active_requests,
                    "available": self.runner.max_active_requests - len(self._active),
                    "queued_requests": sum(not item.started for item in self._active.values()),
                    "runner_state_count": self._state_count,
                    "runner_state_bytes": self._state_bytes, "closed": self._closed,
                    "accounting_boundary": "last_completed_model_call",
                    "counters": dict(self._counts), "terminal_requests": list(self._terminal)}

    def provenance(self):
        origins = {}
        for name, module in tuple(sys.modules.items()):
            if name == "__main__":
                # `python -m ...server` executes the file as __main__, without
                # inserting its canonical module name into sys.modules.
                name = getattr(getattr(module, "__spec__", None), "name", name)
            if name == "nanovllm" or name.startswith("nanovllm."):
                path = getattr(module, "__file__", None)
                if path and Path(path).is_file():
                    origins[name] = {"path": str(Path(path).resolve()),
                                     "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        versions = {name: str(getattr(sys.modules[name], "__version__", "unknown"))
                    for name in ("torch", "transformers", "triton") if name in sys.modules}
        device = {"device_type": "synthetic", "device_count": 0, "device_model": "test_only"}
        if self.loader_report["dtype"] != "test_only":
            import torch

            device = {"device_type": "cuda", "device_count": torch.cuda.device_count(),
                      "device_model": torch.cuda.get_device_name(0)}
        execution = {}
        if self.loader_report["dtype"] == "fp8":
            from .quant import FP8Linear

            modules = {name: module for name, module in self.runner.model.named_modules()
                       if isinstance(module, FP8Linear)}
            calls = {name: module.native_fp8_calls for name, module in modules.items()}
            execution = {"fp8_execution": {"modules": len(calls), "total_calls": sum(calls.values()),
                         "executed_modules": sum(count > 0 for count in calls.values()), "per_module_calls": calls,
                         "weight_dtype": "float8_e4m3fn", "scale_dtype": "float32",
                         "actual_dtypes_valid": all(module.weight.dtype == torch.float8_e4m3fn
                                                    and module.weight_scale_inv.dtype == torch.float32
                                                    for module in modules.values())}}
        return {"model_id": self.model_id, "precision": self.loader_report["dtype"],
                "loader_report": self.loader_report, "module_origins": origins, "versions": versions,
                **device, **execution}

    def generate(self, prompts, sampling_params=None, use_tqdm=False):
        from nanovllm.sampling_params import SamplingParams

        if not isinstance(prompts, list) or not prompts:
            raise ValueError("prompts must be a nonempty list")
        if type(use_tqdm) is not bool:
            raise ValueError("use_tqdm must be boolean")
        params = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(prompts)
        if len(params) != len(prompts):
            raise ValueError("one sampling configuration is required per prompt")
        resolved = []
        for item in params:
            item = SamplingParams(temperature=0) if item is None else item
            if not isinstance(item, SamplingParams) or item.temperature != 0:
                raise ValueError("dedicated engine supports temperature=0 greedy sampling only")
            resolved.append(item)
        results = []
        for prompt, item in zip(prompts, resolved):
            request = self.submit(prompt, max_tokens=item.max_tokens, ignore_eos=item.ignore_eos)
            try:
                while request.finish_reason is None:
                    request.step()
                results.append(request.result())
            finally:
                request.close()
        return results

    def begin_shutdown(self):
        with self._registry_lock:
            self._closed = True
            for request in self._active.values():
                if not request.cancelled.is_set():
                    request.cancel_reason = "shutdown"
                    request.cancelled.set()

    def close(self):
        self.begin_shutdown()
        with self._model_lock:
            for request in list(self._active.values()):
                self._finish(request, request.cancel_reason)

    exit = close


class Generation:
    def __init__(self, engine, request_id, prompt_ids, max_tokens, ignore_eos):
        self.engine = engine
        self.request_id = request_id
        self.prompt_ids = prompt_ids
        self.max_tokens = max_tokens
        self.ignore_eos = ignore_eos
        self.token_ids = []
        self.text = ""
        self.started = False
        self.finish_reason = None
        self.cancelled = threading.Event()
        self.cancel_reason = "cancelled"

    def result(self):
        return {"text": self.text, "token_ids": self.token_ids.copy(), "finish_reason": self.finish_reason}

    def _cancel_event(self):
        previous = self.text
        try:
            decoded = self.engine.runner.tokenizer.decode(
                self.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )
            if not decoded.startswith(previous):
                raise ValueError("tokenizer changed already streamed text")
            self.text = decoded
        except BaseException:
            self.engine._finish(self, "error")
            raise
        self.engine._finish(self, self.cancel_reason)
        return {"delta": self.text[len(previous):], "token_id": None, **self.result()}

    def step(self, disconnect_check=None):
        with self.engine._model_lock:
            if self.finish_reason is not None:
                return {"delta": "", "token_id": None, **self.result()}
            if self.cancelled.is_set():
                return self._cancel_event()
            if disconnect_check is not None and disconnect_check():
                self.engine._finish(self, "disconnected")
                return {"delta": "", "token_id": None, **self.result()}
            try:
                if not self.started:
                    logits = self.engine.runner.prefill(self.request_id, self.prompt_ids)
                    self.started = True
                else:
                    logits = self.engine.runner.append(self.request_id, [self.token_ids[-1]])
                self.engine._refresh_ownership()
                if self.cancelled.is_set():
                    return self._cancel_event()
                token = int(logits[0, -1].argmax().item())
                self.token_ids.append(token)
                finish = ("stop" if not self.ignore_eos and token in self.engine.eos_token_ids else
                          "length" if len(self.token_ids) == self.max_tokens else None)
                decoded = self.engine.runner.tokenizer.decode(
                    self.token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )
                stable = decoded if finish is not None else decoded.rstrip("\ufffd")
                if not stable.startswith(self.text):
                    raise ValueError("tokenizer changed already streamed text")
                delta = stable[len(self.text):]
                self.text = stable
                if finish is not None:
                    self.engine._finish(self, finish)
                return {"delta": delta, "token_id": token, **self.result()}
            except BaseException:
                self.engine._finish(self, "error")
                raise

    def close(self, reason="cancelled"):
        with self.engine._model_lock:
            self.engine._finish(self, reason)
