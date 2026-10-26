# Decode input RMS/FP8 fusion

`NANOVLLM_INPUT_RMS_FP8=triton` selects an optional input-normalization and
activation-quantization producer when each decoder layer is constructed. The
default is `native`. It accepts contiguous inference BF16 singleton hidden state
with width 5120 on SM90, the original RMSNorm, and the original FI-backed merged
QKV or QKVZ projection. Fresh prefill, including a one-token prompt, and all
unsupported metadata take the original route before execution. Selected kernel
or consumer errors propagate without a fallback retry.

The producer uses the original norm's cached BF16 `(1 + FP32(weight))` and
epsilon. It computes the full-row FP32 mean square, inverse RMS and weighted
normalized product, then explicitly rounds that product to BF16 before block128
quantization. The declared FI quantizer operations remain `s = amax != 0 ?
448 / amax : 1`, stored `1 / s`, and FP32 BF16-value-times-s before E4M3 conversion.
There is no scale floor, checkpoint conversion or scale reciprocation beyond
the original quantizer contract. Reduction and conversion lowering can differ
from native execution; this source prototype does not establish numerical or
compiled-instruction equivalence.

The 48 GDN layers require two outputs: normalized BF16 for their unchanged BA
projection, and FP8 values/scales for their original QKVZ GEMM. The 16 full
attention layers need only the latter; they omit the normalized BF16 allocation
and store while retaining its rounding boundary in registers. A private
projection handoff bypasses only the already executed QKV/QKVZ call. Attention,
Q/K norms, output projections, residuals, post-attention norm, final norm and
prefill arithmetic are unchanged. The H2 gate/up-only consumer remains separate.

Each invocation allocates Torch-owned FP8 `[1,5120]` values and full FP32
`[40,4]` backing with live `[1,40]` scales at stride `(1,4)`. Padding is undefined.
The public FI FP8-input GEMM consumes the original FP8 checkpoint weights and
FP32 scales, returns BF16, and preserves the original split views. The scale
view owns its backing; no module-global scratch or private FI cache is used.
The current Torch stream orders producer and consumer, and the existing capture
pool owns captured temporaries.

Metadata queries do not create the norm cache or mark execution. Readiness is
recorded only after the complete eager producer/consumer returns. Cold capture
fails before the producer; the existing model Graph path performs two eager
calls before capture. The norm cache keeps its existing behavior, including its
existing treatment of later weight mutation; this option does not repair or
change that policy. Python counters report pair calls and exclude Graph replay.

The source structure adapts the existing H2 Triton producer without residual
addition or mutation. Its quantization contract is the existing FI
`scale_1x128_kernel` and public prequantized GEMM entry. Other projects' fused
norm/quant paths motivate the handoff but are not imported or runtime
dependencies. No native build is added.

Focused CPU controls cover routing, the private consumer, cache/capture
preconditions and buffer ownership. Actual-operand operator diagnostics and a
short service comparison are recorded in the
[campaign acceptance issue](https://github.com/InferMatrixLab/InferRSI/issues/67).
They include same-arm Graph/state/reset checks and cross-native diagnostics;
cross-native FP8 and projection results are not generally bit-exact. The short
service comparison is one measured pair, not formal statistical or task-quality
acceptance. The selector remains experimental and disabled by default while
full campaign validation is outstanding.
