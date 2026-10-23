# Optional whole Q/K L2 normalization

`NANOVLLM_GDN_QK_NORM=triton` is a default-off, construction-bound candidate
for grouped FlashInfer GDN prefill. It requires `NANOVLLM_GDN=flashinfer` and
`NANOVLLM_GDN_PREFILL_QK_LAYOUT=grouped`. Native remains the default. The loader
rejects mixed layer selections and reports prospective configuration rather
than claiming execution. There is no dependency on a post-sum tail option.

A paired kernel reads actual Q/K strides and combines BF16-to-FP32 conversion,
separate FP32 squared terms, FP32 sum, `+1e-6`, reciprocal square root, FP32
multiplication and fresh contiguous BF16 outputs. FP fusion is disabled; actual
emitted square/reduction boundaries still require compiler review. The
16-token/four-warp schedule is a hypothesis for token-minor input and
feature-minor output, not evidence of coalescing or speed.

The Triton reduction tree may differ from Torch. Cross-backend numerical and
logit differences are diagnostics, not a bit-exactness claim or an invented
error tolerance. This candidate needs the unchanged task-quality and retrieval
gates before acceptance. Same-candidate eager/Graph equality, valid outputs,
state ownership and cleanup requirements remain in force. No new quantization,
dtype, KV compression, truncation or query scaling is introduced.

Eligibility is restricted to NVIDIA SM90, inference BF16 tensors with matching
`[1,T,16,128]` shapes/devices, `1 <= T <= 8192`, and conservatively
nonoverlapping positive-stride reads with bounded address span. Unsupported
metadata selects the original native normalization explicitly. Compilation or
execution errors propagate. Outputs have independent storage and canonical
contiguous layout. V, beta, gates, recurrent state, decode and the FlashInfer
core remain unchanged.

Compilation, numerical diagnostics, real first/last-layer replay, generated
layout/resources, full task quality and service performance are NOT_RUN.
The source is an experimental candidate pending independent review and a
separately authorized frozen experiment plan.
