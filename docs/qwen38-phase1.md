# Dedicated Qwen3.8-27B text prototype

This path develops an independent Qwen3.8-27B text engine in the nano-vllm
codebase. InferRSI coordinates experiments externally; the runtime does not
import or require InferRSI. Governing scope and acceptance are recorded in
[InferRSI task](https://github.com/InferMatrixLab/InferRSI/issues/53) and
[FP8 phase-1 acceptance](https://github.com/InferMatrixLab/InferRSI/issues/57).

Status: **unverified; INCONCLUSIVE**. Real FP8 checkpoint, single-H20 numerical
validation and end-to-end qualification are pending. The implementation is a
correctness-first prototype; no performance gain is claimed.

## Execution boundary

- The production loader accepts the exact Qwen/Qwen3.8-27B text configuration
  and its official Qwen/Qwen3.8-27B-FP8 dynamic E4M3 block-128 profile.
  Tiny configurations are an explicit testing API.
- The dedicated model owns every decoder layer and its full-attention KV,
  convolution and FP32 recurrent state. It does not call Transformers model
  `forward` or `generate`. The native path uses eager Torch operators and a
  dedicated Triton FP8 linear kernel.
- One visible CUDA device and greedy generation are supported. The public engine
  serializes model calls while concurrent requests own independent states and
  interleave token steps; this is not vectorized continuous batching.
- Prefix sharing, paged KV integration, CUDA Graph, stochastic sampling, tensor
  parallelism, additional quantization formats, MTP and vision are not implemented in this path.
  The public `LLM` entrypoint selects the dedicated path for this model;
  unrelated configurations retain the original engine. See the
  [engine and HTTP/SSE contract](qwen38-service.md) for supported parameters,
  streaming, cancellation and shutdown behavior.
- Weight loading requires complete text parameter coverage and explicit
  exclusion of vision/MTP tensors. A local index is not model authentication:
  the acceptance workflow first verifies every frozen file against immutable
  upstream identities.

The FP8 path requires SM90. It retains E4M3 weights, quantizes activations
dynamically per token and K128 group, and accumulates scaled products in FP32
before BF16 output. Checkpoint BF16 scales are represented exactly as FP32.
Excluded projections, embeddings, output head and normalization parameters
retain BF16; recurrent state remains FP32. No dense weight-dequantization
fallback is implemented. This describes the implementation, not verified GPU
correctness or performance.

## Run from this checkout

Prepare a Python 3.12 environment with CUDA-capable Torch, Transformers and
Safetensors using the campaign's frozen dependency lock; FP8 also requires
compatible Triton. Tiny reference tests
also require pytest. The dedicated path does not import the legacy Flash
Attention/Triton runner. After preparing the environment, from this checkout:

```bash
<ENV_PYTHON> -m nanovllm.qwen38 --model <MODEL_PATH> \
  --prompt 'Explain how a hash table works.' --max-new-tokens 32 \
  --max-context-tokens 4096 --chat
```

The command deliberately performs greedy generation. It uses the local
checkpoint's tokenizer/chat template and full generation EOS set, without
applying the checkpoint's stochastic sampling defaults. It requires local
files and does not execute remote model code. Context limit includes requested
output tokens. On a Slurm host, submit through an allocation with explicit
resources, root-contained caches and preserved device visibility.

The request API accepts an already loaded model:

```python
from nanovllm.qwen38.loader import load_checkpoint
from nanovllm.qwen38.runner import RequestRunner

model, loading = load_checkpoint(model_path)
runner = RequestRunner(model, max_context_tokens=4096,
                       max_active_requests=2, vocab_size=model.config.vocab_size)
logits_a = runner.prefill('a', prompt_a_ids)
logits_b = runner.prefill('b', prompt_b_ids)
next_logits_a = runner.append('a', [next_token_a])
runner.release('a')
runner.release('b')
```

Each request has a separate state. Failed append operations release that
request; other active requests remain owned by the runner. Release is
idempotent. Reset creates a fresh state for an active request.

## Validation and attribution

`<ENV_PYTHON> -m pytest tests/test_qwen38_*.py` exercises small-model numerical
comparisons, loader corruption checks and request lifecycle behavior. These
checks cannot qualify the downloaded 27B checkpoint or establish GPU speed.
Real model comparisons must retain the frozen workload, validator, exact source
identities, runtime and complete InferRSI results.
Phase 1 also requires real-model HTTP/SSE functional evidence for concurrent
isolation, cancellation, disconnection, errors, shutdown and state/capacity reuse.
CPU socket tests do not establish that service acceptance.

The model math is derived from the Apache-2.0 Transformers Qwen3.5 source.
See [NOTICE-qwen38](../NOTICE-qwen38) and [Apache license](../LICENSES/Apache-2.0.txt).
The original nano-vllm code retains its [MIT license](../LICENSE).

## Optional small-M FP8 GEMM

`NANOVLLM_FP8_GEMM=flashinfer` selects FlashInfer's SM90 block-scaled
FP8 GEMM for fewer than 32 activation rows, including single-token decoding.
Larger inputs retain the existing Triton prefill implementation. This opt-in
requires `flashinfer-python` with `flashinfer.gemm.fp8_blockscale_gemm_sm90`;
missing dependencies or unsupported shapes raise an error. The default remains
`triton`. Weights stay E4M3 and checkpoint block multipliers stay FP32; FlashInfer
quantizes BF16 activations internally and returns BF16 outputs.

The loader reports this mixed backend, and provenance separates FlashInfer and
Triton Python call counts. These counts include warmup/capture and exclude CUDA
Graph replay. Selecting the backend does not establish model quality or service
performance; those require the governing acceptance campaign's evidence.
