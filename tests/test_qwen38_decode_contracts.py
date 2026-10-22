"""decode 契约测试：增量文本拼接与 CUDA Graph 边界回退。

B1：byte-level BPE 下中文/emoji 跨 token 时，流式增量拼接必须等于非流式解码，
    且 finish 那一步不得重复下发已发过的文本。
B2：Graph 桶容量用尽（输出超过 headroom）时必须回退 eager，不能越界。
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm.qwen38.engine import _IncrementalText  # noqa: E402

MODEL_DIR = os.environ.get("QWEN38_MODEL", "/home/david_cwq/jcz/models/Qwen3.8-27B")

needs_tokenizer = pytest.mark.skipif(
    not (Path(MODEL_DIR) / "tokenizer.json").exists(),
    reason="real Qwen tokenizer not available",
)


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True,
                                        trust_remote_code=False)


@needs_tokenizer
@pytest.mark.parametrize("text", [
    "你好，世界。这是一段中文。",
    "混合 emoji 😀🚀🎯 与中文的句子",
    "多字节连续：🎉🎉🎉然后是汉字",
    "plain ascii baseline text",
])
def test_incremental_text_matches_full_decode(text):
    tokenizer = _load_tokenizer()
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) >= 4

    streamer = _IncrementalText(tokenizer)
    streamed = ""
    for token_id in ids:
        streamed += streamer.push(token_id)
    streamed += streamer.finish()

    reference = tokenizer.decode(ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
    assert streamed == reference, f"streamed={streamed!r} reference={reference!r}"
    assert streamer.text == reference


@needs_tokenizer
def test_finish_does_not_replay_text():
    """finish 的 delta 不得重复已下发内容（旧实现的静默重复就出在这里）。"""
    tokenizer = _load_tokenizer()
    ids = tokenizer.encode("汉字与 emoji 😀 混合", add_special_tokens=False)

    streamer = _IncrementalText(tokenizer)
    chunks = [streamer.push(t) for t in ids]
    tail = streamer.finish()

    assert all(chunk is not None for chunk in chunks)
    assert len(tail) < len("".join(chunks)) + 1
    joined = "".join(chunks) + tail
    reference = tokenizer.decode(ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
    assert joined == reference


def test_demote_graph_clears_pool_and_busy():
    """降级必须同时：紧凑化 KV、摘掉池里的失效图、清 busy、graph_bucket 归零。"""
    import torch

    from nanovllm.qwen38.runner import RequestRunner, _Request

    runner = RequestRunner(None, max_context_tokens=64)

    class _Layer:
        def __init__(self, capacity):
            self.capacity = capacity
            self.key = torch.arange(capacity, dtype=torch.float32).reshape(1, 1, capacity, 1)
            self.value = self.key.clone()
            self.conv = None
            self.recurrent = None

    class _State:
        def __init__(self, layers):
            self.layers = layers
            self.capacity = 64
            self.positions = torch.zeros(1, dtype=torch.long)
            self.length = 0

    request = _Request(state=_State([_Layer(64), _Layer(64)]))
    request.length = 5
    request.graph = object()
    request.graph_bucket = 64
    runner._graph_pool[64] = request.graph
    runner._graph_busy.add(64)

    runner._demote_graph(request)

    assert request.graph is None
    assert request.graph_bucket == 0
    assert 64 not in runner._graph_pool, "失效的图必须从池里摘掉"
    assert 64 not in runner._graph_busy, "busy 标记必须清掉，否则该桶永远不复用"
    assert request.state.capacity == 0
    assert request.state.positions is None
    assert request.state.length == 5
    for layer in request.state.layers:
        assert layer.capacity == 0
        assert layer.key.shape[2] == 5
        assert layer.value.shape[2] == 5
        assert layer.key.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


@needs_tokenizer
def test_incremental_decode_stays_within_window():
    """push 100 个 token，任何一次 decode 的输入长度都不得超过 WINDOW（常数代价）。"""
    tokenizer = _load_tokenizer()
    base = tokenizer.encode("汉字与 emoji 😀 混合的流式解码测试", add_special_tokens=False)
    ids = (base * (100 // len(base) + 2))[:100]

    seen = []

    class _Counting:
        def decode(self, token_ids, **kwargs):
            seen.append(len(token_ids))
            return tokenizer.decode(token_ids, **kwargs)

    streamer = _IncrementalText(_Counting())
    streamed = "".join(streamer.push(t) for t in ids)
    streamed += streamer.finish()

    assert seen, "decode 未被调用"
    assert max(seen) <= _IncrementalText.WINDOW, f"出现全量解码：max={max(seen)}"
    reference = tokenizer.decode(ids, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)
    assert streamed == reference


@needs_tokenizer
@pytest.mark.skipif(not os.environ.get("QWEN38_GPU_CONTRACT"),
                    reason="GPU integration test (set QWEN38_GPU_CONTRACT=1)")
def test_graph_output_beyond_headroom_falls_back_to_eager():
    """小 prompt、max_tokens 超过 256 headroom、开图：不报错且与非图逐 id 一致。"""
    os.environ["NANOVLLM_CUDA_GRAPH"] = "1"
    from nanovllm.qwen38.engine import DedicatedEngine

    prompt = list(range(11, 43))
    common = dict(model_path=MODEL_DIR, max_context_tokens=2048, max_active_requests=1)
    graph_engine = DedicatedEngine(common["model_path"], max_context_tokens=2048,
                                   max_active_requests=1)
    ref_engine = DedicatedEngine(common["model_path"], max_context_tokens=2048,
                                 max_active_requests=1)
    ref_engine.runner.graph_enabled = False

    graph_ids = _greedy(graph_engine, prompt, 300)
    ref_ids = _greedy(ref_engine, prompt, 300)
    assert graph_ids == ref_ids


def _greedy(engine, prompt, max_tokens):
    request = engine.submit(prompt, max_tokens=max_tokens, ignore_eos=True)
    while request.finish_reason is None:
        request.step()
    return list(request.token_ids)
