# Optional four-row DeepGEMM activation quantization

`NANOVLLM_DEEPGEMM_QUANTIZATION=row4` selects one fixed four-row by 128-column
Triton quantization tile, with four warps. The default is `native`. Selection
belongs to each constructed FP8 linear, survives projection merging and is
reported under `loader_report.deepgemm_quantization`. Invalid names and mixed
selections fail visibly. Changing the environment after construction does not
change those modules.

The option applies only when `NANOVLLM_FP8_GEMM=flashinfer_deepgemm` reaches its
DeepGEMM branch. FlashInfer still handles fewer than 32 rows. Other GEMM backends
remain unaffected and report `active: false`. Health reports successful Python
DeepGEMM calls by quantizer, with the same scope as existing counters: warmup
and Graph capture count; Graph replay does not rerun the Python counter. Module
origins retain the actual adapter source identity.

The original native quantizer file remains unchanged. The candidate preserves
FP32 `max(abs(x))/448.0`, the separate `max(scale,1e-12)` denominator, division,
E4M3 cast and unfloored stored scale. There is no explicit clamp, E8M0 rounding
or weight transformation. Q stays contiguous; FP32 scales retain column-major
strides `(1,ceil(M/4)*4)`. All input loads, Q stores and scale stores mask
`row<M`: the final scale column's padding is outside its allocated storage.
The GEMM consumer retains the same checkpoint weights/scales, recipe and
`disable_ue8m0_cast=True`.

This is a source-only candidate with unverified CUDA arithmetic and performance.
Four logical rows and four warps do not establish one warp per row, a particular
reduction tree, vector load width, resource use or a speedup. CPU tests execute
the source indexing and finite FP32 expressions with stubs; they deliberately
do not implement FP8 conversion or claim GPU equality.

The inspected [vLLM native register quantizer](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/csrc/quantization/w8a8/fp8/per_token_group_quant.cu)
and [multirow Triton path](https://github.com/vllm-project/vllm/blob/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm/model_executor/layers/quantization/utils/fp8_utils.py)
provide scheduling precedent. Their epsilon, clamping, reciprocal and E8M0
semantics are not substituted for this engine's expression. The
[Hopper resource guidance](https://github.com/InferMatrixLab/InferRSI/blob/04ca833eff7cf83aede933cfa041807305316dab/src/inferrsi/data/skill_library/hardware/cuda/reason-hopper-kernels/references/resources-and-scheduling.md)
motivates inspecting actual register/spill/layout evidence, not inferring it
from source shape. No Blackwell-specific mechanism or autotuning is included.

## Required validation before promotion

Keep the native path until a separately authorized diagnostic establishes all
of the following on the frozen target runtime. Preserve every attempted check,
including compilation errors, mismatches, timeouts and partial records; stop
on any mismatch rather than introduce a tolerance.

1. Compare Q as raw bytes and every live FP32 scale as raw bits for production
   K values 5120, 6144 and 17408, partial rows 1, 3, 5, 31, 33, and aligned rows
   4, 32, 8192. Use distinct row/group maxima at varying lanes; random finite
   inputs; rounding-boundary values; outliers; positive/negative zero; BF16
   subnormals; the denominator-floor transition; largest finite values; and
   isolated/multiple NaNs and both infinities at varying positions. Record
   nonfinite classifications and payload bytes explicitly. No `allclose` or
   dequantized-only comparison can establish this contract.
2. Verify minimal scale allocation, untouched column padding and canaries on
   both outputs, unchanged input storage and fresh output ownership. Include
   contiguous views with a nonzero/misaligned storage offset. Use an authorized
   device memory checker for the row-tail cases; mask inspection is not a
   memory-safety result. Keep undefined padding out of numerical comparisons.
3. Retain the actual native and candidate TTIR, TTGIR, LLVM IR, PTX/cubin and
   available disassembly, launch grid and metadata. Inspect row reduction
   independence, division/conversion lowering, shared memory, barriers,
   registers and spills. The four-row source is not a compiled-layout proof.
4. Capture actual DeepGEMM activation operands at the first and last eligible
   projections for each production K during the accepted prefill workload.
   Bind exact source/runtime/checkpoint identities and operand bytes, strides
   and offsets. Compare quantization, then the unchanged GEMM on the same
   checkpoint operands. Verify small-M FlashInfer calls remain unchanged.
5. Use shared native history for whole-model logits, all live state endpoints,
   same-candidate eager/Graph predictions, reset and cached continuation.
   Confirm stable Graph-owned buffers and actual health/module/counter
   evidence. Also capture and replay the quantizer itself with changing input
   values, checking Q/live-scale bits and owned buffers after replay: the
   model's small-M decode Graph uses FlashInfer and cannot establish this
   quantizer's own Graph behavior. Concrete workloads, coverage and caps belong to the prospective
   diagnostic plan, not an implicit execution permission in this document.

Compilation, numerical, memory-checker, model, Graph and service validation are
pending. Later performance work must preserve the accepted measurement boundary
and frozen campaign gates; operator evidence alone cannot establish service gain.
