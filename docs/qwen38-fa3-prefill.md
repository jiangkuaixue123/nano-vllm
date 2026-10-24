# Optional standalone FA3 initial prefill

`NANOVLLM_PREFILL_ATTENTION=flashinfer_fa3` selects FlashInfer's standalone
SM90 FA3 attention for a fresh request. The setting is captured when each
FullAttention module is constructed. The default remains `native`; the existing
`sdpa_flash` option remains available. Cached multi-token continuation and
single-token decode use the existing native path for every setting.

The adapter accepts NVIDIA SM90, CUDA 12.3 or newer, same-device BF16 tensors,
batch one, 24 query heads, four KV heads and head dimension 256. Query and KV
lengths must match and be nonzero. It checks a unit last stride and positive,
16-byte-aligned head/token strides and data pointers before importing FlashInfer.
Unsupported inputs and backend errors fail explicitly; no alternative attention
backend is selected. There is no new vLLM dependency or package installation.

Q/K have already passed through normalization and partial RoPE. The adapter
passes transpose views to `flashinfer.prefill.single_prefill_with_kv_cache` with
explicit `backend="fa3"`, `kv_layout="NHD"`, causal masking, scale 0.0625, no
additional positional encoding, no FP16 QK reduction, no window or soft cap,
and BF16 output. KV heads are not repeated. The merged projection's strided V
view is retained. Existing KV writes, output sigmoid gate and projection remain
in their original order.

The inspected FlashInfer 0.7.0.post1 single-FA3 C++ binding selects HND strides
but reads KV length/head count from dimensions 0/1 unconditionally. This adapter
therefore always passes NHD views, including for physically head-major Q/K.
The public API allocates a 32 MiB scratch tensor and fresh contiguous NHD output;
at 8192 tokens the BF16 output is 96 MiB. First use can compile an SM90a module
in the configured FlashInfer cache. Preparation time and allocation behavior
must be included in a bounded execution plan; they are not measured here.

`loader_report.full_attention_prefill` records the constructed backend, actual
FullAttention layer count, fresh-prefill scope and native cached continuation.
Existing health/stats provenance forwards that report. It is a configuration
snapshot, not evidence that a kernel executed. A GPU diagnostic must separately
bind the actual package/module, operands, backend call and compiled artifacts.

FA3 changes floating-point reduction order. Cross-backend differences from
native or FlashSDPA are observations, not a bit-equality promise or a newly
chosen error tolerance. The existing task/retrieval quality policy still applies.
Same-candidate eager/Graph comparisons remain exact. Before service timing,
validate production strides and grouped-head mapping, finite BF16 output,
immutability, future-V causal invariance, zero input and actual model operands;
then validate same-candidate full logits/live state, reset and cached continuation.

The metadata tests use mock tensors and do not execute Torch or CUDA arithmetic.
Torch consumer tests exercise dispatch/cache/provenance when Torch is available.
This source change has no GPU correctness, quality or performance result.
