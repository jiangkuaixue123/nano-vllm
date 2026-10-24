# Optional GDN output normalization to FP8 projection input

`NANOVLLM_GDN_OUTNORM_FP8=triton` enables a narrow M1 producer/consumer bridge.
Its default is `native`. Selection requires the existing
`NANOVLLM_GDN_OUTNORM=triton`, SM90, BF16 `[1,1,48,128]` input and gate,
canonical contiguous input, the original BF16 norm weight `[128]`, and the
plain FP8 output projection from 6144 to 5120 using FlashInfer. Unsupported
metadata takes the existing configured norm/projection path. Once selected,
compilation or launch errors propagate without a native retry.

The GDN value-head width and the FP8 quantization-group width are both 128.
The producer writes the original 48-head flattened order directly to FP8
`[1,6144]`; it introduces no head permutation or BF16 output materialization.
The gate can remain a strided view of the merged projection. Each program
handles four heads, matching the existing BF16 output norm's 12-program M1
launch geometry. It preserves that kernel's four adjacent values per lane,
explicit FMA accumulation and lane-zero shuffle reduction, followed by
`libdevice.rsqrt`. The norm result, SiLU gate and product each retain their
existing BF16 rounding boundary; SiLU retains `libdevice.exp` and `tl.div_rn`.
The norm weight comes from the same detached `_fused_weight` cache as before.

The quantization epilogue uses the existing H2/FlashInfer contract: ignore
NaNs for the magnitude maximum; use `448 / amax`, or 1 for zero; emit `1 / scale`
as the FP32 multiplier; convert the rounded BF16 product times scale to E4M3
with round-to-nearest. It does not substitute `amax / 448`, UE8M0, a different
checkpoint format or a new normalization order. Logical scales are `[1,48]`
with stride `(1,4)` over the complete `[48,4]` backing; padding is not output.

The private FP8Linear consumer retains the original FlashInfer GEMM, checkpoint
weights/scales and BF16 output. Full eager producer/consumer execution must
finish before Graph capture. New allocations belong to the capture pool; no
shared scratch, norm parameter, GDN state, KV state or request position is
written by this bridge. Existing state updates and reset logic remain in their
original locations. The existing H2 and MLP paths are unchanged.

`DeltaNet._forward_outnorm(output, gate)` is the integration boundary.
`outnorm_fp8_execution` is a metadata-only query; `outnorm_fp8_calls` counts
completed host calls. Loader `gdn_outnorm_fp8` and service
`gdn_outnorm_fp8_execution` expose configured selection and successful calls.
Warmup and capture count as host calls; Graph replay does not.

Focused CPU tests cover metadata fallback, detached norm-weight handling,
complete scale backing, cold capture, failure propagation, fresh outputs,
consumer object identity and counters. AST comparisons preserve the current
norm reduction/rounding expressions and H2 quantization expressions. They do
not establish target GPU arithmetic, compiler equivalence, latency or task
quality. Any later operator/Graph/service evaluation must use the existing
frozen gates; this source prototype alone establishes no gain.

SGLang's [GDN model dispatch](https://github.com/sgl-project/sglang/blob/406786fd910dd82d03d1ce2c2e4c08b547786bc2/python/sglang/srt/models/qwen3_5.py#L1098-L1120)
and [gated normalization producer](https://github.com/sgl-project/sglang/blob/406786fd910dd82d03d1ce2c2e4c08b547786bc2/python/sglang/kernels/ops/attention/fla/layernorm_gated.py#L185-L219)
provide the direct normalized-output-to-quantized-projection precedent. That
upstream route is AMD/Quark with different scale granularity and arithmetic;
its kernel body is not copied here. llama.cpp's
[QWEN35 GDN graph](https://github.com/ggml-org/llama.cpp/blob/6b9ba4f2e8c1436511956f0b3c309e2c9e642071/src/models/qwen35.cpp#L347-L476)
provides architecture context, not an SM90 FP8 implementation to transplant.
This implementation adapts the engine's own `_triton_gdn_outnorm.py` and
`_triton_rms.py`. Upstream Apache-2.0/MIT source references and this repository's
existing licenses and notices remain applicable to their respective content.
