# Dedicated text entrypoint and local service

The public entrypoint selects the dedicated engine for local `qwen3_5`
configurations; the strict loader admits only the supported Qwen3.8 text
architecture and BF16 or declared block-FP8 representation. Other configurations
retain the legacy engine entrypoint. Importing `LLM` does not import the legacy
Flash Attention runner. Authenticate checkpoint bytes before loading them.

```python
from nanovllm import LLM, SamplingParams

engine = LLM(model_path, max_context_tokens=context_limit, max_active_requests=capacity)
try:
    outputs = engine.generate(
        ["Hello"], SamplingParams(temperature=0, max_tokens=output_limit),
        use_tqdm=False,
    )
finally:
    engine.close()
```

The dedicated constructor accepts `max_context_tokens` and
`max_active_requests`. Legacy constructor options are not aliases for these
limits. Generation accepts text or token-list prompts and greedy sampling only.
Each result contains `text`, `token_ids`, and `finish_reason` (`stop` or `length`).
`ignore_eos=True` is available through the Python and HTTP APIs. Stochastic sampling and
unsupported constructor or HTTP fields fail explicitly. The unrelated legacy
engine continues to require positive sampling temperature.

## HTTP entrypoint

```bash
python -m nanovllm.qwen38.server \
  --model "$MODEL_PATH" \
  --max-context-tokens "$CONTEXT_LIMIT" \
  --max-active-requests "$REQUEST_CAPACITY" \
  --max-connections "$CONNECTION_LIMIT" \
  --socket-timeout "$SOCKET_TIMEOUT" \
  --ready-file "$READY_FILE" --shutdown-file "$SHUTDOWN_FILE"
```

The default bind address is loopback, port 8000. Host, port, body size, socket
timeout, context, admission and connection limits are configurable. Keep at least
two connection slots beyond the active-request limit for control requests.
Connection saturation returns 503; request admission exhaustion returns 429.
One HTTP request is served per connection; chunked request bodies are unsupported.

`POST /v1/completions` accepts the following bounded OpenAI-style contract:

| Field | Support |
| --- | --- |
| `model` | Optional exact loaded model ID |
| `prompt` | Text or a list of nonnegative vocabulary token IDs |
| `input_ids` | Token-list alternative to `prompt`; provide exactly one |
| `max_tokens` | Required positive integer; prompt plus output must fit context |
| `temperature` | Optional; only zero is supported |
| `stream` | Optional boolean, default false |
| `request_id` | Optional identifier using letters, digits, `_`, `.`, `:`, `-` |
| `ignore_eos` | Optional boolean, default false; true generates through EOS up to the requested cap |
| `return_token_ids` | Optional boolean; true also echoes exact prompt IDs in the first stream choice or non-stream response |
| `repetition_penalty` | Optional; only the neutral value 1.0 is supported |
| `logprobs` | Optional; only null is supported |
| `stream_options` | Streaming only; supports boolean `include_usage`, default false |

Non-streaming JSON contains `choices[0].text`, `token_ids`, `finish_reason`, and
token usage. Streaming responses use `text/event-stream`: each `data:` JSON chunk
contains the request ID and a choice with cumulative-decoding text delta and new
token IDs. Existing generated `token_ids` remain present regardless of the new
prompt-echo flag. Normal termination puts its finish reason on the final token
choice, without an additional empty choice that a benchmark could time as a
token. A cancellation with no pending delta uses an empty terminal choice.
With `include_usage=true`, choice chunks carry null usage and one final chunk
has empty `choices` and actual prompt/completion/total token counts. Prompt IDs
are echoed only once. The stream ends with `data: [DONE]`.
Normal termination uses `stop` or `length`;
explicit cancellation uses `cancelled`; service drain uses `shutdown`. Runtime
errors emit a sanitized error object and terminal `error` framing when the
connection is still writable. HTTP runtime failures return 500. Validation
failures return 400 before admission.

`POST /v1/cancel/{request_id}` returns `{"request_id": "...", "cancelled": true}`
for an active request, or 404 for an unknown/finished request. Cancellation is
accompanied by `active_request_ids_before_cancel`, captured under the same lock
as its acknowledgement. Terminal records contain `active_peer_request_ids` from
the instant the request is removed, so isolation checks can distinguish current
peers from historical overlap. The Python `cancel()` API still returns a boolean;
`cancel_with_observation()` returns the HTTP observation object. Cancellation is
observed before or after a model call. In-flight CUDA work cannot be preempted.
Disconnects are checked again after waiting for model ownership, before allocating
request state. Client write-side EOF is treated as disconnection. Each terminal
path releases request state and admission capacity; transport errors cannot affect
another request's state.

`GET /health` reports readiness, model/precision, loaded-weight report, runtime
versions and loaded module origins/hashes. `GET /debug/state` (also `/stats`)
adds active IDs, capacity, queued requests, logical state bytes, outcome counts,
the last 128 terminal records, and actual FP8 call coverage. Module origins are
local runtime diagnostics containing absolute paths; redact them before sharing.
State byte counts sum request-owned KV, convolution and recurrent tensor metadata,
not allocator reservations. Accounting snapshots describe the last completed
model call, allowing control queries during an in-flight call without GPU sync.

SIGINT/SIGTERM stop admission and request cancellation. Incomplete-body/idle
connections are closed immediately during drain. Generation handlers have one
socket-timeout interval to emit terminal framing, after which sockets are closed
to unblock slow writers. CUDA work remains non-preemptible. The optional shutdown
file is written after handler drain and state release. Existing evidence files
are never overwritten.

Model calls are serialized one token step at a time; concurrent requests own
independent states. This functionality does not establish throughput or latency
improvement. CPU fake-model socket tests verify protocol and lifetime behavior;
real FP8 model functional service acceptance is a separate evidence milestone.

### Initial causal prefill with merged FP8 projections

`NANOVLLM_PREFILL_ATTENTION=sdpa_flash` selects PyTorch SDPA's required
`FLASH_ATTENTION` backend for fresh causal prefill only. The option composes
with `NANOVLLM_FP8_PROJECTIONS=merged`; cached continuation and decode retain
the existing position-aware attention. Default remains `native`, and unsupported
backend/device execution raises rather than selecting a fallback.

This is the previously reviewed initial-prefill implementation applied after the
merged-projection change. Merged V is a strided view of packed QKV. The CPU test
covers routing and preservation of that view, but GPU support, numerical/task
quality and HTTP timing must be checked on the combined source. It is not a
claim that PyTorch's backend is the same FlashAttention version used by the
matched vLLM baseline. No performance gain is asserted by this preparation.
