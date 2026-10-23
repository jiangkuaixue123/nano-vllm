"""Optional FlashInfer GDN bridge; native request state stays FP32 [B,H,K,V].

Prefill follows vLLM 0.32's fi_chunk_gated_delta_rule adapter: normalize Q/K
before the call and pass exp(log_decay). FlashInfer prefill state is [B,H,V,K],
whereas its nontranspose decode API consumes the native [B,H,K,V] layout.
These fused reductions change rounding; this is not an exact native oracle.
"""

import torch


def prefill(query, key, value, decay, beta, initial):
    # Optional CUDA dependency: importing the ordinary CPU model must not load it.
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    def normalized(x):
        work = x.float()
        return (work * torch.rsqrt((work * work).sum(-1, keepdim=True) + 1e-6)).to(x.dtype)

    length = query.shape[1]
    state = None if initial is None else initial.transpose(-1, -2).contiguous()
    output, updated = chunk_gated_delta_rule(
        q=normalized(query).squeeze(0).contiguous(),
        k=normalized(key).squeeze(0).contiguous(),
        v=value.squeeze(0).contiguous(),
        g=decay.squeeze(0).float().exp().contiguous(),
        beta=beta.squeeze(0).float().contiguous(),
        initial_state=state,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, length], dtype=torch.int64, device=query.device),
        use_qk_l2norm_in_kernel=False,
        backend="flashinfer",
    )
    return output.unsqueeze(0), updated.transpose(-1, -2).contiguous()


def decode(query, key, value, initial, a_log, a, dt_bias, b):
    from flashinfer.gdn_decode import gated_delta_rule_decode

    # Caller supplies writable request-owned state. This API updates it in place.
    return gated_delta_rule_decode(
        q=query.contiguous(), k=key.contiguous(), v=value.contiguous(),
        state=initial, A_log=a_log.float(), a=a.contiguous(),
        dt_bias=dt_bias, b=b.contiguous(), use_qk_l2norm=True,
        backend="flashinfer",
    )
