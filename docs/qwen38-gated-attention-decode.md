# Experimental gated decode attention

`NANOVLLM_GATED_DECODE_ATTENTION=triton_split` selects one default-off,
unqualified Triton prototype. The default is `native`. This is source work;
no GPU compilation, operator timing, full-model quality or service result is
established. Status: **INCONCLUSIVE**. Do not infer a service gain from a
component profile or from removing launches.

The only supported metadata is NVIDIA SM90, BF16 contiguous Q `[1,24,1,256]`,
BF16 contiguous K/V `[1,4,10240,256]`, BF16 contiguous gate `[1,1,6144]`, and
one contiguous int64 device position on the same device. All operands are
inference tensors. Q heads `6*j..6*j+5` use KV head `j`. K/V head stride is
2621440 elements and token stride is 256. The existing projection, Q/K norm,
partial RoPE and KV writes remain in their original order. The projection
interleaves query/gate per head; the existing gate reshape materializes its
contiguous flattened head order before this prototype consumes it.

Fresh prefill, cached multi-token continuation, other capacities, layouts,
dtypes and devices use native attention. Metadata selection reports the
reason and never claims execution. A selected import, compilation or launch
failure propagates. The loader reports the constructed selection; FP8 engine
health reports per-module host calls and last metadata route. Those counters
include warmup/capture and do not count Graph replay.

## Fixed kernel design

Stage one launches 4 KV groups × 32 splits, with 320 contiguous tokens per
split. Each split iterates 32-token tiles with a 16-query-head tensor-core
tile, masking to the actual six heads; launch settings are four warps and two
stages. These tile dimensions follow the inspected SGLang grouped kernel.
The fixed split count exposes 128 independent blocks and divides capacity
exactly. It is a design choice, not a measured optimum or an autotune result.
Every live token is included; there is no sparsity, pruning, quantized KV,
context truncation, speculation or model change.

Stage two performs a stable ordered merge of 32 FP32 partial outputs and
log-sum-exp values for each query head, then applies the sigmoid gate. Both
stages load live length from `positions[0] + 1`; no host `.item()`, captured
Python length, page tables, planner, index repack or length-update launch is
needed. Empty splits are overwritten and excluded from the merge. A reset
with a shorter length cannot consume the old KV tail or old partials.

The existing in-bounds KV write precedes launch and provides the position
contract `[0,10239]`. Metadata eligibility does not inspect device values.
Scratch is per-invocation: FP32 partials `[24,32,256]`, FP32 LSE `[24,32]`, and
BF16 output `[1,1,6144]` (801792 bytes combined). Eager tensors or the existing
model-owned Graph pool own these allocations; no shared scratch/state is added.
Warm a complete eligible eager launch before capture. Cold capture is rejected
before KV writes; the readiness marker is device-bound and does not constitute
runtime qualification. The current Graph capture already warms eager decode.

## Numerical boundary

Native attention materializes BF16 QK, BF16 scaled scores, FP32 global softmax,
BF16 globally normalized probabilities, BF16 PV, BF16 sigmoid and BF16 product.
The prototype retains BF16 QK and scaled-score boundaries, scale `1/16`, and the
last three BF16 boundaries. It computes online softmax and casts *unnormalized*
tile probabilities to BF16 for tensor-core PV, then rescales/merges FP32
partials. Its GEMM/reduction order and probability rounding therefore differ
from native. Gate sigmoid uses libdevice exponential and rounded division;
this choice alone does not prove equivalence to the installed Torch build.
No approximate model algorithm is introduced, but bit-exact native output is
not promised. Finite-input assumptions and numerical differences require actual
operator and full-model examination under the existing frozen policy.

The first future qualification must check the same candidate eager versus
Graph exactly, live lengths crossing tile/split boundaries, full capacity,
short reset after dirty tails, stable state addresses and independent request
ownership. Cross-native differences are diagnostics under the existing policy,
not a newly invented tolerance. The original checkpoint, BF16 KV, FP32 GDN
state, full logical context, output-token count and frozen quality gates stay
unchanged. This document adds no harness, GPU submission or acceptance claim.

## Source provenance and limits

Adapted actual source: [SGLang grouped decode stages and wrapper](https://github.com/sgl-project/sglang/blob/406786fd910dd82d03d1ce2c2e4c08b547786bc2/python/sglang/kernels/ops/attention/decode_attention.py#L530-L1019),
Apache-2.0; notices and declared LightLLM ancestry are retained in the kernel
and `NOTICE-qwen38`. Changes remove serving/page/MLA/AMD options, replace
dynamic scheduler splits with fixed contiguous partitions, preserve the native
BF16 score boundaries, and fuse the gate into the merge. No upstream runtime
dependency is added. License text is in `LICENSES/Apache-2.0.txt`.

The separately inspected [llama.cpp attention selector](https://github.com/ggml-org/llama.cpp/blob/6b9ba4f2e8c1436511956f0b3c309e2c9e642071/ggml/src/ggml-cuda/fattn.cu#L631-L670)
excludes its common unquantized vector path for GQA > 4 with D >= 256 or
KV length >= 8192; its alternative F16 preparation does not establish a
BF16-preserving transfer. No llama.cpp code is copied into this prototype.
Prior FA2 unsplit and FlashInfer CUDA-core split-KV negatives remain stopped;
this is a separate direct-stride Triton implementation, not a retry or sweep
of either library configuration.
