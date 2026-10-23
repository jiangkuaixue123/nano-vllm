"""Eager, single-device request ownership for an already loaded native model.

The model owns numerical execution. This runner owns request identity, bounds,
state lifetime and text generation; it never calls Transformers model APIs.
"""

import os
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral


@dataclass
class _Request:
    state: object
    length: int = 0
    graph: object = None
    graph_bucket: int = 0


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
        # CUDA Graph 解码：按 bucket 缓存图并复用，抓一次摊到后续所有请求上。
        self.graph_enabled = os.environ.get("NANOVLLM_CUDA_GRAPH") == "1"
        self.graph_policy = os.environ.get("NANOVLLM_CUDA_GRAPH_POLICY", "buckets")
        if self.graph_policy not in ("buckets", "full_context"):
            raise ValueError("NANOVLLM_CUDA_GRAPH_POLICY must be buckets or full_context")
        if self.graph_enabled and self.graph_policy == "full_context" and max_active_requests != 1:
            raise ValueError("full_context graphs require max_active_requests=1")
        self.graph_headroom = 256
        self._graph_pool: dict[int, object] = {}
        self._graph_busy: set[int] = set()

    @property
    def active_request_ids(self) -> tuple[str, ...]:
        return tuple(self._requests)

    def _tokens(self, token_ids, *, allow_empty=False):
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

    def _graph_capacities(self):
        if self.graph_policy == "full_context":
            return (self.max_context_tokens,)
        from .graph import DecodeGraph

        return DecodeGraph.BUCKETS

    def _graph_capacity_for(self, prompt_length):
        if self.graph_policy == "full_context":
            return self.max_context_tokens
        from .graph import DecodeGraph

        return DecodeGraph.bucket_for(prompt_length + self.graph_headroom)

    def prewarm_graphs(self):
        """启动时把所有桶各抓一遍，capture 成本计入 startup，不进 TTFT。

        返回逐桶的 capture 耗时与常驻显存增量（字节），供 run 记录引用。
        """
        if not self.graph_enabled:
            return {}
        import time

        import torch

        from .graph import DecodeGraph

        report = {"policy": self.graph_policy, "buckets": {}, "started_unix": time.time()}
        before_all = torch.cuda.memory_allocated()
        for capacity in self._graph_capacities():
            if capacity in self._graph_pool:
                continue
            before = torch.cuda.memory_allocated()
            started = time.perf_counter()
            state = self.model.new_state(capacity=capacity)
            graph = DecodeGraph(self.model, state, 0, capacity).capture()
            self._graph_pool[capacity] = graph
            self._graph_busy.discard(capacity)
            report["buckets"][capacity] = {
                "capture_seconds": time.perf_counter() - started,
                "resident_MiB": (torch.cuda.memory_allocated() - before) / 2 ** 20,
            }
        report["total_seconds"] = sum(v["capture_seconds"] for v in report["buckets"].values())
        report["resident_MiB_total"] = (torch.cuda.memory_allocated() - before_all) / 2 ** 20
        report["completed_unix"] = time.time()
        self.prewarm_report = report
        return report

    def _acquire_graph_state(self, capacity):
        """取一张可复用的图（连同它的 state），没有空闲图时返回 None。

        图绑定具体的缓冲区地址，所以同一 bucket 的图在同一时刻只能服务一个请求；
        并发请求各自捕获不再入池，只有顺序复用的请求吃到缓存。
        """
        if capacity in self._graph_busy:
            return None
        graph = self._graph_pool.get(capacity)
        if graph is None:
            return None
        self._graph_busy.add(capacity)
        return graph

    def _release_graph(self, request):
        if request.graph is not None and request.graph_bucket:
            self._graph_busy.discard(request.graph_bucket)
            request.graph = None
            request.graph_bucket = 0

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
        capacity = None
        if self.graph_enabled:
            capacity = self._graph_capacity_for(len(tokens))
        request = _Request(self.model.new_state())
        pooled = None
        if capacity:
            from .graph import DecodeGraph

            pooled = self._acquire_graph_state(capacity)
            if pooled is not None:
                # 复用同一 bucket 的图：把它的 state 清成新请求状态后重新 prefill。
                request.state = pooled.reset_request_state()
            else:
                request.state = self.model.new_state(capacity=capacity)
        self._requests[request_id] = request
        try:
            logits = self.append(request_id, tokens)
            if pooled is not None:
                request.graph = pooled
                request.graph_bucket = capacity
        except BaseException:
            self._release_graph(request)
            raise
        if capacity and request.graph is None and request.length < capacity:
            from .graph import DecodeGraph

            graph = DecodeGraph(self.model, request.state, request.length,
                                capacity).capture()
            request.graph = graph
            request.graph_bucket = capacity
            if capacity not in self._graph_pool:
                self._graph_pool[capacity] = graph
            self._graph_busy.add(capacity)
        return logits

    def _demote_graph(self, request):
        """图容量用尽：把预分配 KV 压成紧凑布局，后续走 eager 路径。

        图里烘焙的是这批预分配缓冲区的地址；state 被换成紧凑 clone 之后，池里那张图
        就永久失效——必须从池里摘掉并清 busy 标记，否则下一次 acquire 会拿到它并
        replay 到旧地址，产生静默的错误输出（不是崩溃）。
        """
        graph = request.graph
        bucket = request.graph_bucket
        request.graph = None
        request.graph_bucket = 0
        if graph is None:
            return
        if bucket:
            self._graph_pool.pop(bucket, None)
            self._graph_busy.discard(bucket)
        state = request.state
        length = request.length
        for layer in state.layers:
            if getattr(layer, "capacity", 0):
                # Linear-attention layers reserve conv/recurrent state, not KV.
                # Preserve that history while switching every layer to eager.
                if layer.key is not None:
                    layer.key = layer.key[:, :, :length].clone()
                    layer.value = layer.value[:, :, :length].clone()
                layer.capacity = 0
        state.capacity = 0
        state.positions = None
        state.length = length

    def append(self, request_id: str, token_ids):
        """Consume more tokens; any failure invalidates and releases this request."""
        self._request_id(request_id)
        if request_id not in self._requests:
            raise KeyError("unknown request_id")
        try:
            request = self._requests[request_id]
            tokens = self._tokens(token_ids)
            if request.graph is not None and len(tokens) == 1:
                position = request.length
                # 边界：桶容量与上下文上限都要先判，越界就永久退回 eager，
                # 不能让图内 index_copy_ 越界（那是崩溃，不是回退）。
                if (position + 1 <= request.graph.capacity
                        and position + 1 <= self.max_context_tokens):
                    request.length += 1
                    return request.graph.step(tokens[0], position)
                self._demote_graph(request)
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
            if self._tensor_factory is None and len(tokens) > 1:
                import torch

                # 只在 prefill（多 token）校验非有限值。decode 单 token 时每步做
                # isfinite(...).all().item() 会插入一次全量规约 + device 同步，
                # 在 CUDA Graph 路径上还会直接阻断 capture；而 decode 的数值风险
                # 由 prefill 校验与每步 argmax 的有限性间接覆盖。
                if not torch.isfinite(logits).all().item():
                    raise ValueError("native model returned nonfinite logits")
            request.length += len(tokens)
            return logits
        except BaseException:
            self.release(request_id)
            raise

    def release(self, request_id: str) -> bool:
        """Drop only the named request's state; repeated release is harmless."""
        request = self._requests.get(request_id)
        if request is None:
            return False
        self._release_graph(request)
        self._requests.pop(request_id, None)
        return True

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
