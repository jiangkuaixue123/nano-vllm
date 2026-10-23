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

## Optional FP8 GEMM backends

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

`NANOVLLM_FP8_GEMM=flashinfer_deepgemm` additionally uses an independently
installed `deep_gemm` package for M>=32 prefill. It keeps the same FlashInfer
decode path. Activation quantization uses the existing block-128 FP8 semantics
and produces TMA-aligned FP32 scales; the GEMM explicitly disables UE8M0 casting.
Official checkpoint weight bytes and multipliers remain unchanged. This mode
does not import the vLLM runtime and fails if DeepGEMM is unavailable.

### Full-context decode graph (experimental)

With CUDA Graph enabled, `NANOVLLM_CUDA_GRAPH_POLICY=full_context` reserves and
prewarms one graph at the configured `max_context_tokens`. This policy requires
`max_active_requests=1`. Admitted single-token
decode stays in that capacity through the context limit, avoiding the default
bucket policy's mid-request eager demotion. The default remains `buckets`.
This can increase resident KV memory and masked-attention work for short inputs.
GPU numerical behavior and end-to-end effect require separate validation.

### FlashInfer GDN (experimental)

`NANOVLLM_GDN=flashinfer` selects FlashInfer GDN prefill and single-token decode
for the dedicated 128-dimensional heads. The default remains `native` and an
explicit request fails if the optional backend is unavailable. This uses the
installed FlashInfer operator package without importing the vLLM runtime.
Prefill follows the normalization/decay API used by vLLM 0.32 and transposes
the operator's V,K state to the engine's K,V layout. Recurrent state stays FP32;
buffered decode updates fixed storage in place, while unbuffered decode copies
state to preserve the transactional model call.

This changes numerical rounding: prefill normalized Q/K are rounded to the
activation dtype, and fused decode computes gates in FP32 without the native
BF16 sigmoid intermediate. Operator, Graph, full-model quality and end-to-end
qualification are required; enabling it is not a quality or parity claim.

### Merged FP8 projections (experimental)

`NANOVLLM_FP8_PROJECTIONS=merged` combines gate/up, Q/K/V and GDN QKV/Z
projections before allocation. The default is `separate`. Only complete
128-row scale blocks are concatenated; each checkpoint FP8 byte and its original
multiplier load directly into the corresponding slice. No weight requantization
or dense fallback is performed. GDN A/B and the language-model head keep their
existing BF16 paths. In the supported architecture this reduces FP8 linear
modules from 400 to 256. The DeepGEMM path quantizes the input once per group;
the native Triton kernel still quantizes within each output tile.

This follows vLLM's merged projection layout without importing its runtime.
Larger GEMM output shapes can change floating-point accumulation, so structural
CPU tests do not establish numerical equivalence or a service speedup. Operator,
Graph, task-quality and end-to-end comparisons remain required before acceptance.

### Merged BF16 GDN b/a projections (experimental)

For an FP8 checkpoint, `NANOVLLM_GDN_BA_PROJECTIONS=merged` combines each GDN
layer's excluded BF16 b/a weights into one BF16 projection, in `[b, a]` order,
before allocation. The default is `separate`; this option is independent of
`NANOVLLM_FP8_PROJECTIONS`. FP8 exclusions, weight bytes and block scales remain
unchanged. Original checkpoint b/a tensors load directly into their row slices,
and the loader reports `gdn_ba_projection_layout`. Unsupported option values
and requesting this option for a BF16 checkpoint fail explicitly.

This follows the merged b/a layout in the inspected
[vLLM implementation](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L677),
without importing its runtime. Changing GEMM output width can change cuBLAS
reduction order and BF16 outputs. CPU layout/loading checks do not establish
numerical equivalence, Graph correctness, model quality or service speedup;
those remain unverified pending their corresponding GPU and acceptance checks.
