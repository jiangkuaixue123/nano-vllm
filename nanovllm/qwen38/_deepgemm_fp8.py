"""SM90 prefill GEMM with unchanged checkpoint weights and FP32 block scales."""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_groups(X, Q, S, K: tl.constexpr, PADDED_M: tl.constexpr):
    row = tl.program_id(0)
    group = tl.program_id(1)
    offsets = row * K + group * 128 + tl.arange(0, 128)
    values = tl.load(X + offsets).to(tl.float32)
    # Preserve the existing native kernel's per-token/block-128 quantization.
    # The scale is not rounded to E8M0 and weights are never requantized.
    scale = tl.max(tl.abs(values), axis=0) / 448.0
    quantized = (values / tl.maximum(scale, 1e-12)).to(tl.float8e4nv)
    tl.store(Q + offsets, quantized)
    tl.store(S + group * PADDED_M + row, scale)


def quantize(inputs: torch.Tensor):
    m, k = inputs.shape
    if inputs.dtype != torch.bfloat16 or not inputs.is_cuda or not inputs.is_contiguous() or k % 128:
        raise ValueError("DeepGEMM quantization requires contiguous CUDA BF16 [M, K128]")
    padded_m = triton.cdiv(m, 4) * 4
    output = torch.empty((m, k), device=inputs.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty_strided((m, k // 128), (1, padded_m), device=inputs.device, dtype=torch.float32)
    _quantize_groups[(m, k // 128)](inputs, output, scales, k, padded_m, num_warps=4)
    return output, scales


def matmul(inputs, weight, scales):
    # An external DeepGEMM package is required. Do not import vLLM or silently
    # fall back to a different implementation when the dependency is absent.
    from deep_gemm import fp8_gemm_nt

    m, k = inputs.shape
    n = weight.shape[0]
    if k % 128 or n % 64:
        raise ValueError("DeepGEMM FP8 requires K divisible by 128 and N by 64")
    quantized, activation_scales = quantize(inputs)
    output = torch.empty((m, n), device=inputs.device, dtype=torch.bfloat16)
    fp8_gemm_nt((quantized, activation_scales), (weight, scales), output,
                recipe=(1, 128, 128), disable_ue8m0_cast=True)
    return output
