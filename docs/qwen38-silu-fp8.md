# Optional SwiGLU to FP8 down-projection input

`NANOVLLM_SILU_FP8=triton` enables a narrow producer/consumer fusion. The default
is `native`. The selected path additionally requires the existing
`NANOVLLM_MLP_ACTIVATION=triton`, an SM90 CUDA device, one BF16 gate/up token of
width 17408, and the original plain FP8 down projection from 17408 to 5120 using
the FlashInfer small-M backend. Unsupported metadata retains the configured
activation and projection path. A selected compilation or launch failure is
reported without retrying another backend.

The producer reads the actual strided gate/up views from the merged projection.
It preserves the existing `libdevice.exp` and `tl.div_rn` SiLU expression, BF16
rounding after SiLU, and BF16 rounding after multiplication. It then applies the
existing H2/FlashInfer group-128 quantizer: `448 / amax`, or 1 for a zero group,
the reciprocal output scale, and round-to-nearest E4M3 conversion. It does not
replace this with `amax / 448`, round scales to UE8M0, or change checkpoint
weights. Scales have logical shape `[1,136]`, stride `(1,4)`, and a live
`[136,4]` allocation; only the first column is output, not its padding bytes.

The private FP8Linear consumer uses the same FlashInfer GEMM and BF16 result.
The existing merged gate/up H2 consumer is unchanged. The complete eager
producer/consumer must run before graph capture. Captured allocations belong
to the graph pool; the bridge has no shared scratch or request-state writes.
MLP `silu_fp8_calls`, loader `silu_fp8`, and service `silu_fp8_execution` expose
the route. Counters count completed Python calls, including warmup/capture,
and do not count graph replay.

CPU tests cover routing, full scale backing, cold capture, failure propagation,
and counters. They do not validate GPU arithmetic or performance. The separate
operator probe compares actual model gate/up values, FP8 bytes, live scales,
down-projection results, and eager/graph execution. Quality and service claims
remain subject to the governing acceptance issue and its frozen gates. This
prototype alone establishes no gain.

Design references are SGLang's
[fused group quantization interface](https://github.com/sgl-project/sglang/blob/406786fd910dd82d03d1ce2c2e4c08b547786bc2/python/sglang/kernels/ops/quantization/per_token_group_quant.py#L198-L241)
and [BF16 activation stages](https://github.com/sgl-project/sglang/blob/406786fd910dd82d03d1ce2c2e4c08b547786bc2/python/sglang/kernels/jit/csrc/gemm/per_token_group_quant.cuh#L297-L367),
and llama.cpp's [SwiGLU epilogue](https://github.com/ggml-org/llama.cpp/blob/6b9ba4f2e8c1436511956f0b3c309e2c9e642071/ggml/src/ggml-cuda/mmvf.cu#L340-L380).
These sources motivate the fusion boundary. Their kernel bodies, alternative
weight representations, and numerical shortcuts are not copied. The arithmetic
here is adapted from this engine's existing `_triton_silu.py` and
`_triton_rms.py`; the repository's existing attribution and licenses remain.
