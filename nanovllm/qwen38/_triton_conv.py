"""Opt-in four-tap BF16 convolution/SiLU for fresh prefill and one-token decode.

The request state has four entries, including the newest token. Convolution
rounds to BF16 before SiLU; this is deliberately different from applying SiLU
directly to the FP32 accumulator. GPU numerical qualification remains required.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _conv_silu_kernel(X, W, H, S, Y, C: tl.constexpr, T: tl.constexpr,
                      XC: tl.constexpr, XT: tl.constexpr,
                      WC: tl.constexpr, WK: tl.constexpr,
                      HC: tl.constexpr, HK: tl.constexpr,
                      SC: tl.constexpr, SK: tl.constexpr,
                      DECODE: tl.constexpr, BC: tl.constexpr, BT: tl.constexpr):
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    if DECODE:
        # All history loads precede the writes. Each program owns disjoint
        # channels, so H and S may alias without a cross-program state race.
        x0 = tl.load(H + c * HC + HK, c < C, other=0)
        x1 = tl.load(H + c * HC + 2 * HK, c < C, other=0)
        x2 = tl.load(H + c * HC + 3 * HK, c < C, other=0)
        x3 = tl.load(X + c * XC, c < C, other=0)
        w0 = tl.load(W + c * WC, c < C, other=0).to(tl.float32)
        w1 = tl.load(W + c * WC + WK, c < C, other=0).to(tl.float32)
        w2 = tl.load(W + c * WC + 2 * WK, c < C, other=0).to(tl.float32)
        w3 = tl.load(W + c * WC + 3 * WK, c < C, other=0).to(tl.float32)
        acc = tl.fma(w0, x0.to(tl.float32), 0.0)
        acc = tl.fma(w1, x1.to(tl.float32), acc)
        acc = tl.fma(w2, x2.to(tl.float32), acc)
        acc = tl.fma(w3, x3.to(tl.float32), acc)
        rounded = acc.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        activated = tl.div_rn(rounded, 1.0 + libdevice.exp(-rounded))
        tl.store(Y + c, activated.to(tl.bfloat16, fp_downcast_rounding="rtne"), c < C)
        tl.store(S + c * SC, x0, c < C)
        tl.store(S + c * SC + SK, x1, c < C)
        tl.store(S + c * SC + 2 * SK, x2, c < C)
        tl.store(S + c * SC + 3 * SK, x3, c < C)
    else:
        t = tl.program_id(1) * BT + tl.arange(0, BT)
        acc = tl.full((BC, BT), 0.0, tl.float32)
        for k in tl.static_range(4):
            source_t = t - 3 + k
            valid = (c[:, None] < C) & (source_t[None, :] >= 0) & (source_t[None, :] < T)
            x = tl.load(X + c[:, None] * XC + source_t[None, :] * XT,
                        valid, other=0).to(tl.float32)
            w = tl.load(W + c * WC + k * WK, c < C, other=0).to(tl.float32)
            # Torch skips out-of-range taps, including 0*Inf/NaN at the left edge.
            acc = tl.where(valid, tl.fma(w[:, None], x, acc), acc)
        rounded = acc.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
        activated = tl.div_rn(rounded, 1.0 + libdevice.exp(-rounded))
        tl.store(Y + c[:, None] * T + t[None, :],
                 activated.to(tl.bfloat16, fp_downcast_rounding="rtne"),
                 (c[:, None] < C) & (t[None, :] < T))
        if tl.program_id(1) == tl.cdiv(T, BT) - 1:
            s = tl.arange(0, 4)
            state_t = T - 4 + s
            tail = tl.load(X + c[:, None] * XC + state_t[None, :] * XT,
                           (c[:, None] < C) & (state_t[None, :] >= 0), other=0)
            tl.store(S + c[:, None] * SC + s[None, :] * SK, tail, c[:, None] < C)


def _fresh_state(raw):
    # Zero padding clones the full raw tensor before slicing its tail. Infer
    # both clone layouts on Meta so singleton strides survive without copying
    # or allocating the full prompt on the device. Only four tokens are live.
    layout = torch.empty_strided(raw.shape, raw.stride(), dtype=raw.dtype, device="meta")
    layout = layout.clone()[:, :, -4:].clone()
    return torch.empty_strided(layout.shape, layout.stride(), dtype=raw.dtype, device=raw.device)


def causal_conv_silu(raw, weight, history=None, *, state_out=None):
    """Return NCL output and four-entry state; overwrite only supplied state_out.

    Fresh calls ignore previous storage. Decode reads history entries 1..3 and
    appends raw. The model supplies state_out only for its preallocated state;
    unbuffered calls return fresh storage and leave the old state untouched.
    """
    if raw.ndim != 3 or raw.shape[0] != 1 or min(raw.shape[1:]) < 1:
        raise ValueError("Triton GDN convolution requires nonempty [1, channels, tokens]")
    channels, count = raw.shape[1:]
    if weight.shape != (channels, 1, 4):
        raise ValueError("Triton GDN convolution requires [channels, 1, 4] weights")
    if history is not None and (count != 1 or history.shape != (1, channels, 4)):
        raise ValueError("Triton GDN convolution supports only single-token cached decode")
    if state_out is not None and state_out.shape != (1, channels, 4):
        raise ValueError("Triton GDN convolution requires a four-entry output state")
    tensors = [raw, weight, *([history] if history is not None else []),
               *([state_out] if state_out is not None else [])]
    if (raw.device.type != "cuda" or torch.version.hip is not None
            or any(t.device != raw.device or t.dtype != torch.bfloat16 for t in tensors)):
        raise ValueError("Triton GDN convolution requires BF16 on one NVIDIA CUDA device")
    if torch.is_grad_enabled() or any(s <= 0 for t in tensors for s in t.stride()):
        raise ValueError("Triton GDN convolution requires inference mode and positive strides")
    if state_out is None:
        # Positive padding and cached cat use dense NCL.
        state_out = (_fresh_state(raw) if history is None and count >= 4 else
                     torch.empty((1, channels, 4), dtype=raw.dtype, device=raw.device))
    output = torch.empty(raw.shape, dtype=raw.dtype, device=raw.device)
    history_strides = history.stride()[1:] if history is not None else (0, 0)
    bc, bt = (128, 1) if history is not None else (32, 32)
    with torch.cuda.device(raw.device):
        _conv_silu_kernel[(triton.cdiv(channels, bc), triton.cdiv(count, bt))](
            raw, weight, history if history is not None else raw, state_out, output,
            channels, count, *raw.stride()[1:], weight.stride(0), weight.stride(2),
            *history_strides, *state_out.stride()[1:], history is not None, bc, bt,
            num_warps=4, enable_fp_fusion=False,
        )
    return output, state_out
