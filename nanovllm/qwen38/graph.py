"""CUDA Graph 解码单步：KV 预分配原地写 + 设备端 position + 图内 logits。

被 capture 的区域只含张量运算，不含任何 host 同步；每步只有一次 graph.replay 和
两个 8 字节驻留缓冲写入（token id / position），不再有每步 H2D 分配。

同一个 bucket 的图按 runner 缓存复用：图绑定的是 KV / 卷积态 / 递推态的物理地址，
所以复用时不重新 capture，而是把这张图持有的 state 交给下一个请求，由重置钩子把
请求私有的 conv / recurrent 清空后重新 prefill 进同一批缓冲区。省掉的是每个请求
一次的 warmup + capture（8K 桶上约 0.15 s，直接体现在 TTFT 上）。
"""
import torch

# 所有 DecodeGraph 共用一张捕获内存池，与默认池隔离：否则 capture 之后的 eager 分配
# （例如下一个请求的 prefill）会复用图内缓冲，replay 直接读到被覆盖的数据。
_CAPTURE_POOL = None
_CAPTURE_POOL_OWNER = None


def capture_pool(owner):
    """全模块共用的 capture 内存池，并绑定唯一 owner（一个模型/引擎）。

    同一进程里第二个带图引擎复用这张池会让前一张图的地址被覆盖、replay 读到别人的
    数据；那种情况在 allocator 里表现为一句难懂的 "use_count > 0" 断言，所以这里
    提前显式报错。
    """
    global _CAPTURE_POOL, _CAPTURE_POOL_OWNER
    if _CAPTURE_POOL is None:
        _CAPTURE_POOL = torch.cuda.graph_pool_handle()
        _CAPTURE_POOL_OWNER = owner
    elif _CAPTURE_POOL_OWNER != owner:
        raise RuntimeError(
            "CUDA Graph 捕获池已被另一个引擎占用：同一进程内不支持多个开启 "
            "NANOVLLM_CUDA_GRAPH 的引擎（共享 capture pool 会让 replay 读到彼此被"
            "覆盖的缓冲区，表现为输出静默错误）。请给每个引擎独立进程，或只用其中一个"
            "开启 CUDA Graph。"
        )
    return _CAPTURE_POOL


class DecodeGraph:
    BUCKETS = (256, 1024, 2048, 4096, 8448)

    @classmethod
    def bucket_for(cls, needed):
        for b in cls.BUCKETS:
            if needed <= b:
                return b
        return None

    def __init__(self, model, state, length, capacity):
        self.model = model
        self.state = state
        self.length = length
        self.capacity = capacity
        dev = model.model.embed_tokens.weight.device
        self.ids = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(1, dtype=torch.long, device=dev)
        self.logits = None
        self.graph = None

    def capture(self, dummy_token=0):
        state = self.state
        saved = [
            (None if l.key is None else l.key.clone(),
             None if l.value is None else l.value.clone(),
             None if l.conv is None else l.conv.clone(),
             None if l.recurrent is None else l.recurrent.clone())
            for l in state.layers
        ]
        state.positions = self.pos
        self.ids.fill_(dummy_token)
        self.pos.fill_(self.length)
        with torch.inference_mode():
            self.model(self.ids, state=state)          # 预热一份，供 capture 前的分配
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, pool=capture_pool(id(self.model))):
                self.logits = self.model(self.ids, state=state)
        # capture 真实跑了一步，把 KV / 卷积态 / 递推态恢复到抓取前
        with torch.inference_mode():
            for layer, (k, v, c, r) in zip(state.layers, saved):
                if layer.key is not None and k is not None:
                    layer.key.copy_(k)
                if layer.value is not None and v is not None:
                    layer.value.copy_(v)
                if layer.conv is not None and c is not None:
                    layer.conv.copy_(c)
                if layer.recurrent is not None and r is not None:
                    layer.recurrent.copy_(r)
        self.pos.fill_(self.length)
        return self

    def reset_request_state(self):
        """把这张图持有的 state 清成“新请求”的状态，供复用。

        KV 缓冲区不动（下一次 prefill 会按 position 原地覆盖需要的位置，剩余位置由
        解码期的 mask 屏蔽），只清掉请求私有的卷积态与递推态，并把长度归零。
        """
        with torch.inference_mode():
            for layer in self.state.layers:
                # 缓冲保留（图里绑定了这些地址），值清掉、fresh 打回首步语义。
                layer.fresh = True
                if layer.recurrent is not None:
                    layer.recurrent.zero_()
                if layer.conv is not None:
                    layer.conv.zero_()
        self.state.length = 0
        self.state.positions = None
        return self.state

    def step(self, token, position):
        self.ids.fill_(token)
        self.pos.fill_(position)
        self.graph.replay()
        return self.logits
