"""One fixed BF16 b/a M1 GEMV candidate; native-bit parity is not claimed."""

import torch
import triton
import triton.language as tl


@triton.jit
def _ba_gemv_kernel(X, W, O):
    row = tl.program_id(0)
    k = tl.arange(0, 8192)
    # Keep the original BF16 operands; only the accumulator arithmetic is FP32.
    x = tl.load(X + k, k < 5120, other=0).to(tl.float32)
    w = tl.load(W + row * 5120 + k, k < 5120, other=0).to(tl.float32)
    total = tl.sum(x * w, axis=0)
    tl.store(O + row, total.to(tl.bfloat16, fp_downcast_rounding="rtne"))


def gemv(x, weight):
    """Fresh output on the current stream; no operand copies or scratch buffers."""
    if (tuple(x.shape) not in ((1, 5120), (1, 1, 5120))
            or tuple(weight.shape) != (96, 5120)):
        raise ValueError("BF16 b/a GEMV requires M1/N96/K5120")
    if (x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or x.layout != torch.strided or weight.layout != torch.strided
            or not x.is_contiguous() or tuple(weight.stride()) != (5120, 1)
            or x.is_conj() or x.is_neg() or weight.is_conj() or weight.is_neg()):
        raise ValueError("BF16 b/a GEMV requires contiguous unmodified BF16 operands")
    if (x.device.type != "cuda" or x.device != weight.device
            or torch.version.hip is not None
            or torch.cuda.get_device_capability(x.device) != (9, 0)):
        raise ValueError("BF16 b/a GEMV requires colocated SM90 CUDA operands")
    if torch.is_grad_enabled():
        raise ValueError("BF16 b/a GEMV requires inference/no-grad mode")
    if any(t.data_ptr() % 16 for t in (x, weight)):
        raise ValueError("BF16 b/a GEMV requires aligned operands")
    with torch.cuda.device(x.device):
        output = torch.empty((*x.shape[:-1], 96), device=x.device, dtype=x.dtype)
        _ba_gemv_kernel[(96,)](x, weight, output, num_warps=4,
                               enable_fp_fusion=False)
        stream = torch.cuda.current_stream(x.device)
        x.record_stream(stream)
        weight.record_stream(stream)
    return output
