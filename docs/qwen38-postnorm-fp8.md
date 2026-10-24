# Decode post-attention RMS/FP8 fusion

`NANOVLLM_POSTNORM_FP8=triton` selects the optional post-attention add/RMS and
activation-quantization fusion when each MLP is constructed. The default is
`native`. Only SM90 inference BF16 `[1,5120]` or `[1,1,5120]` operands, with an
existing FlashInfer-backed merged gate/up projection, are eligible. Other shapes
and backends retain the original BF16 producer and projection. A selected kernel
or GEMM failure propagates; it never retries after mutating the residual.

The kernel retains the existing full-row BLOCK=8192 RMS expressions and cached
BF16 `(1 + weight)`. Both the residual sum and normalized product explicitly
round to BF16. It writes the residual in place, then quantizes the rounded norm
in groups of 128 without storing a normalized BF16 tensor. The declared FI
quantizer operations are `scale = amax != 0 ? 448 / amax : 1`, stored multiplier
`1 / scale`, and FP32 normalized value times scale before E4M3 conversion. There
is no floor, E8M0 conversion, weight conversion or weight-scale reciprocation.
The implementation uses FP32 round-to-nearest division; correspondence to the
installed CUDA quantizer's actual compiled instructions remains a GPU-review
obligation, as do RMS reduction lowering, casts, nonfinites and subnormals.

Fresh Torch-owned FP8 values have shape `[1,5120]`. FP32 scales have full backing
`[40,4]` and a live `[1,40]` view with stride `(1,4)`. Only the live column is an
output; padding contents are undefined. Do not compact the scale view or replace
its backing with the minimum storage of `empty_strided`. The unchanged public
`fp8_blockscale_gemm_sm90` FP8-input entry consumes these scales and the original
checkpoint weights, emits BF16, and preserves the merged gate/up split views.
PyTorch capture-pool ownership retains captured temporaries; no singleton scratch
or private FI cache modification is used.

`MLP.postnorm_fp8_execution` reports metadata eligibility without preparing a norm
cache or claiming execution. `postnorm_gate_up(residual, delta, norm)` returns
`(residual, gate, up)` through the production bridge. Per-MLP counters and health
metadata count Python producer/consumer calls; they exclude Graph replays. The
first selected eager call warms the actual producer and public FI consumer. Cold
capture is rejected before the residual write. The existing model Graph capture
performs two eager calls first.

Reference mechanisms are FlashInfer's `scale_1x128_kernel` in
`fp8_blockscale_gemm_kernel.cuh`, its public `gemm_base.py` prequantized entry, and
the existing model-owned `_add_rms_kernel`. FlashInfer's similarly named fused
add/RMS/FP8-block API is not used: it normalizes an unrounded residual and applies
a different scale floor. This Python/Triton implementation adds no native build
or default-install compilation. The existing package finder includes its module.

Local metadata controls do not qualify compiled arithmetic or performance.
Prospective evaluation must restore residual operands before every warmup,
eager invocation, capture/replay and timing target, and measure the complete
producer plus original GEMM consumer. Cross-backend differences are diagnostic;
no new tolerance or exact-native acceptance gate is defined here. Same-candidate
Graph, task quality and service acceptance remain separately required. No speed
or end-to-end claim is made by enabling the selector.
