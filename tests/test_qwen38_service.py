"""Real HTTP sockets with a synthetic model; no GPU/service quality claims."""

import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_qwen38_runner import Logits, Row

from nanovllm.qwen38.engine import DedicatedEngine
from nanovllm.qwen38.runner import RequestRunner
from nanovllm.qwen38.server import CompletionServer
from nanovllm.sampling_params import SamplingParams


class TensorMetadata:
    def numel(self):
        return 8

    def element_size(self):
        return 4


class Model:
    def __init__(self):
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.block_prompt = None
        self.fail_prompt = None
        self.failure_type = RuntimeError
        self.block_after_first = False
        self.calls = []

    def new_state(self):
        return SimpleNamespace(tokens=[], layers=[SimpleNamespace(recurrent=TensorMetadata())])

    def __call__(self, inputs, *, state):
        fresh = not state.tokens
        if fresh and inputs[0] == self.block_prompt:
            self.entered.set()
            assert self.resume.wait(3), "test forgot to release its blocked model"
        if fresh and inputs[0] == self.fail_prompt:
            raise self.failure_type("synthetic private model failure")
        if not fresh and self.block_after_first and state.tokens == [1]:
            self.entered.set()
            assert self.resume.wait(3)
        self.calls.append((id(state), list(inputs[0])))
        rows = []
        for token in inputs[0]:
            state.tokens.append(token)
            values = [0.] * 8
            values[sum(state.tokens) % 8] = 1.
            rows.append(Row(values))
        time.sleep(0.002)
        return Logits(rows, 8)


class Tokenizer:
    def encode(self, text, **kwargs):
        return [1] if text else []

    def decode(self, tokens, **kwargs):
        return "".join(chr(0x4E00 + token) for token in tokens)


def engine(*, capacity=2, eos=(), tokenizer=None):
    runner = RequestRunner(Model(), tokenizer or Tokenizer(), max_context_tokens=32,
                           max_active_requests=capacity, vocab_size=8, tensor_factory=lambda x: x)
    return DedicatedEngine.from_runner_for_testing(runner, eos_token_ids=eos)


@contextmanager
def serving(model=None, **kwargs):
    model = model or engine()
    server = CompletionServer(("127.0.0.1", 0), model, max_connections=6, socket_timeout=0.5, **kwargs)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield model, server.server_address
    finally:
        model.runner.model.resume.set()
        model.begin_shutdown()
        server.shutdown()
        server.server_close()
        model.close()
        thread.join(2)
        assert not thread.is_alive()
        assert_clean(model)


def request(address, path="/v1/completions", body=None, method="POST"):
    connection = http.client.HTTPConnection(*address, timeout=3)
    connection.request(method, path, body=json.dumps(body) if body is not None else None,
                       headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    return connection, response


def complete(address, prompt=(1,), **kwargs):
    connection, response = request(address, body={"input_ids": list(prompt), "max_tokens": 4, **kwargs})
    try:
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def stream(address, prompt=(1,), **kwargs):
    connection, response = request(address, body={"input_ids": list(prompt), "max_tokens": 4,
                                                 "stream": True, **kwargs})
    assert response.status == 200
    return connection, response


def events(response):
    result = []
    while True:
        line = response.readline()
        if not line:
            raise AssertionError("SSE ended without DONE")
        if line.startswith(b"data: "):
            data = line[6:].strip()
            if data == b"[DONE]":
                return result
            result.append(json.loads(data))


def assert_clean(model):
    state = model.stats()
    assert state["active_requests"] == state["runner_state_count"] == state["runner_state_bytes"] == 0
    assert state["available"] == state["capacity"]


def wait_clean(model):
    deadline = time.monotonic() + 2
    while model.stats()["active_requests"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert_clean(model)


def test_real_http_stream_matches_nonstream_and_serial_entry():
    with serving() as (model, address):
        serial = model.generate([[1]], SamplingParams(temperature=0, max_tokens=4))[0]
        status, plain = complete(address)
        connection, response = stream(address)
        chunks = events(response)
        connection.close()
        assert status == 200
        assert plain["choices"][0]["token_ids"] == serial["token_ids"]
        assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == serial["text"]
        assert [token for chunk in chunks for token in chunk["choices"][0]["token_ids"]] == serial["token_ids"]
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"
        assert_clean(model)


def test_unicode_buffer_and_literal_replacement_are_flushed_on_terminal():
    class ByteTokenizer(Tokenizer):
        def decode(self, tokens, **kwargs):
            return ["", "\ufffd", "你", "你🙂", "你🙂\ufffd"][len(tokens)]

    with serving(engine(tokenizer=ByteTokenizer())) as (model, address):
        _, plain = complete(address)
        connection, response = stream(address)
        chunks = events(response)
        connection.close()
        assert "".join(chunk["choices"][0]["text"] for chunk in chunks) == plain["choices"][0]["text"] == "你🙂\ufffd"
        assert_clean(model)


def test_eos_empty_completion_text_and_normal_cleanup():
    class SpecialTokenizer(Tokenizer):
        def decode(self, tokens, **kwargs):
            return ""

    with serving(engine(eos=(1,), tokenizer=SpecialTokenizer())) as (model, address):
        _, result = complete(address)
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["choices"][0]["token_ids"] == [1]
        assert result["choices"][0]["text"] == ""
        assert_clean(model)


@pytest.mark.parametrize("ignore_eos, expected_count, reason", [(False, 1, "stop"), (True, 4, "length")])
def test_benchmark_stream_counts_real_tokens_and_emits_one_choice_per_token(ignore_eos, expected_count, reason):
    with serving(engine(eos=(1,))) as (model, address):
        connection, response = stream(address, ignore_eos=ignore_eos, return_token_ids=True,
                                      repetition_penalty=1.0, logprobs=None,
                                      stream_options={"include_usage": True})
        chunks = events(response)
        connection.close()
        choices = [chunk["choices"][0] for chunk in chunks if chunk["choices"]]
        ids = [token for choice in choices for token in choice["token_ids"]]
        assert len(ids) == len(choices) == expected_count
        assert choices[0]["prompt_token_ids"] == [1]
        assert all("prompt_token_ids" not in choice for choice in choices[1:])
        assert all(choice["finish_reason"] is None for choice in choices[:-1])
        assert choices[-1]["finish_reason"] == reason
        assert all(chunk["usage"] is None for chunk in chunks[:-1])
        assert chunks[-1]["choices"] == []
        assert chunks[-1]["usage"] == {"prompt_tokens": 1, "completion_tokens": expected_count,
                                       "total_tokens": 1 + expected_count}
        assert_clean(model)


def test_nonstream_token_echo_and_usage_use_actual_input_and_output():
    with serving(engine(eos=(1,))) as (model, address):
        status, result = complete(address, ignore_eos=True, return_token_ids=True)
        assert status == 200
        assert result["choices"][0]["prompt_token_ids"] == [1]
        assert len(result["choices"][0]["token_ids"]) == 4
        assert result["usage"] == {"prompt_tokens": 1, "completion_tokens": 4, "total_tokens": 5}
        assert_clean(model)


@pytest.mark.parametrize("options", [{}, {"include_usage": False}])
def test_optional_usage_is_absent_when_disabled(options):
    with serving() as (model, address):
        connection, response = stream(address, return_token_ids=False, stream_options=options)
        chunks = events(response)
        connection.close()
        assert len(chunks) == 4
        assert all("usage" not in chunk for chunk in chunks)
        assert all("prompt_token_ids" not in chunk["choices"][0] for chunk in chunks)
        assert_clean(model)


@pytest.mark.parametrize("fields", [
    {"ignore_eos": 1}, {"return_token_ids": "true"},
    {"repetition_penalty": True}, {"repetition_penalty": 1.1}, {"logprobs": 0},
    {"stream_options": None}, {"stream_options": {"include_usage": 1}},
    {"stream_options": {"continuous_usage_stats": True}},
    {"stream": False, "stream_options": {"include_usage": True}},
])
def test_invalid_benchmark_options_rejected_before_model_work(fields):
    with serving() as (model, address):
        connection, response = request(address, body={"input_ids": [1], "max_tokens": 4,
                                                     "stream": True, **fields})
        assert response.status == 400
        response.read()
        connection.close()
        assert not model.runner.model.calls
        assert_clean(model)


def test_concurrent_requests_match_serial_and_are_isolated():
    with serving() as (model, address):
        baseline = model.generate([[1], [3]], SamplingParams(temperature=0, max_tokens=4))
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(complete, address, (token,), request_id=f"parallel-{token}") for token in (1, 3)]
            outputs = [item.result() for item in futures]
        assert [item[1]["choices"][0]["token_ids"] for item in outputs] == [item["token_ids"] for item in baseline]
        assert_clean(model)


def test_full_admission_keeps_cancel_health_usable_and_queued_cancel_allocates_no_state():
    model = engine()
    model.runner.model.block_prompt = [1]
    with serving(model) as (_, address):
        first, first_response = stream(address, request_id="running")
        assert model.runner.model.entered.wait(1)
        second, second_response = stream(address, (3,), request_id="queued")
        assert complete(address, (4,))[0] == 429
        health, health_response = request(address, "/health", method="GET")
        assert health_response.status == 200
        health_response.read()
        health.close()
        cancel, cancelled = request(address, "/v1/cancel/queued")
        assert cancelled.status == 200 and json.loads(cancelled.read())["cancelled"]
        cancel.close()
        model.runner.model.resume.set()
        assert events(second_response)[-1]["choices"][0]["finish_reason"] == "cancelled"
        assert events(first_response)[-1]["choices"][0]["finish_reason"] == "length"
        first.close()
        second.close()
        ledger = {item["request_id"]: item for item in model.stats()["terminal_requests"]}
        assert ledger["queued"]["prefilled"] is False
        assert complete(address)[0] == 200
        assert_clean(model)


@pytest.mark.parametrize("queued", [False, True])
def test_disconnect_releases_running_or_queued_request_and_capacity(queued):
    model = engine()
    model.runner.model.block_prompt = [1]
    with serving(model) as (_, address):
        first, first_response = stream(address, request_id="running")
        assert model.runner.model.entered.wait(1)
        if queued:
            second, second_response = stream(address, (3,), request_id="gone")
        else:
            second, second_response = first, first_response
        second_response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
        second_response.close()
        second.close()
        model.runner.model.resume.set()
        if queued:
            events(first_response)
            first.close()
        wait_clean(model)
        ledger = {item["request_id"]: item for item in model.stats()["terminal_requests"]}
        item = ledger["gone" if queued else "running"]
        assert item["finish_reason"] == "disconnected"
        if queued:
            assert item["prefilled"] is False
        assert complete(address)[0] == 200


@pytest.mark.parametrize("failure_type", [RuntimeError, OSError, ValueError, TypeError])
def test_model_error_stream_framing_cleanup_and_subsequent_reuse(failure_type):
    model = engine()
    model.runner.model.fail_prompt = [7]
    model.runner.model.failure_type = failure_type
    with serving(model) as (_, address):
        status, failed = complete(address, (7,))
        assert status == 500
        assert "private" not in json.dumps(failed)
        connection, response = stream(address, (7,), request_id="failure")
        chunks = events(response)
        connection.close()
        assert "error" in chunks[0]
        assert "private" not in json.dumps(chunks)
        assert chunks[-1]["choices"][0]["finish_reason"] == "error"
        assert_clean(model)
        assert complete(address)[0] == 200


@pytest.mark.parametrize("body", [[], {"prompt": "x", "max_tokens": 1, "temperature": 1},
                                     {"prompt": "", "max_tokens": 1},
                                     {"input_ids": [8], "max_tokens": 1},
                                     {"input_ids": [1], "max_tokens": 32},
                                     {"prompt": "x", "max_tokens": 1, "stop": "x"}])
def test_invalid_and_overlimit_requests_do_not_allocate(body):
    with serving() as (model, address):
        connection, response = request(address, body=body)
        assert response.status == 400
        response.read()
        connection.close()
        assert_clean(model)
        assert not model.runner.model.calls


def test_shutdown_cancels_active_stream_and_rejects_new_admission():
    model = engine()
    model.runner.model.block_prompt = [1]
    with serving(model) as (_, address):
        connection, response = stream(address, request_id="closing")
        assert model.runner.model.entered.wait(1)
        model.begin_shutdown()
        model.runner.model.resume.set()
        assert events(response)[-1]["choices"][0]["finish_reason"] == "shutdown"
        connection.close()
        model.close()
        assert_clean(model)
        assert model.stats()["closed"]


def test_request_error_does_not_release_other_request_and_state_bytes_are_metadata():
    model = engine()
    first = model.submit([1], max_tokens=4, request_id="owned")
    first.step()
    assert model.stats()["runner_state_bytes"] == 32
    with pytest.raises(ValueError, match="already active"):
        model.submit([3], max_tokens=1, request_id="owned")
    assert model.stats()["active_request_ids"] == ["owned"]
    first.close()
    first.close()
    assert_clean(model)


def test_active_cancellation_does_not_cancel_survivor_and_releases_state():
    model = engine()
    model.runner.model.block_after_first = True
    with serving(model) as (_, address):
        first, first_response = stream(address, request_id="cancel-active")
        assert model.runner.model.entered.wait(1)
        assert model.stats()["runner_state_bytes"] == 32
        second, second_response = stream(address, (3,), request_id="survivor")
        cancel, response = request(address, "/v1/cancel/cancel-active")
        assert response.status == 200
        response.read()
        cancel.close()
        model.runner.model.resume.set()
        cancelled_events = events(first_response)
        survivor_events = events(second_response)
        first.close()
        second.close()
        assert cancelled_events[-1]["choices"][0]["finish_reason"] == "cancelled"
        assert [token for item in cancelled_events for token in item["choices"][0]["token_ids"]] == [1]
        assert survivor_events[-1]["choices"][0]["finish_reason"] == "length"
        assert_clean(model)


@pytest.mark.parametrize("request_id", [False, 0, "", [], "bad/id"])
def test_invalid_request_ids_do_not_become_generated_ids(request_id):
    with serving() as (model, address):
        assert complete(address, request_id=request_id)[0] == 400
        assert_clean(model)


def test_slow_body_does_not_hold_shutdown_open():
    model = engine()
    server = CompletionServer(("127.0.0.1", 0), model, max_connections=4, socket_timeout=0.1)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    client = socket.create_connection(server.server_address)
    client.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1000\r\n\r\n{")
    time.sleep(0.02)
    model.begin_shutdown()
    server.shutdown()
    closing = threading.Thread(target=server.server_close)
    closing.start()
    closing.join(0.5)
    try:
        assert not closing.is_alive(), "incomplete body blocked service shutdown"
    finally:
        client.close()
        closing.join(1)
        model.close()
        thread.join(1)
    assert_clean(model)


def test_real_sigterm_writes_drained_shutdown_evidence(tmp_path):
    ready, final = tmp_path / "ready.json", tmp_path / "shutdown.json"
    source = """
import sys, threading
import nanovllm
from test_qwen38_service import engine
from nanovllm.qwen38.server import main
model = engine()
model.runner.model.block_prompt = [1]
threading.Timer(1, model.runner.model.resume.set).start()
nanovllm.LLM = lambda *args, **kwargs: model
sys.argv = ['server', '--model', 'synthetic', '--port', '0', '--max-context-tokens', '32',
            '--max-active-requests', '2', '--socket-timeout', '2',
            '--ready-file', sys.argv[1], '--shutdown-file', sys.argv[2]]
main()
"""
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(root), str(root / "tests")))}
    process = subprocess.Popen([sys.executable, "-c", source, str(ready), str(final)], env=environment)
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        address = json.loads(ready.read_text())["address"]
        connection, response = stream(address, request_id="signal-active")
        idle = socket.create_connection(address)
        idle.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1000\r\n\r\n{")
        process.terminate()
        assert events(response)[-1]["choices"][0]["finish_reason"] == "shutdown"
        connection.close()
        idle.close()
        assert process.wait(timeout=4) == 0
        result = json.loads(final.read_text())
        assert result["closed"] and result["active_requests"] == result["runner_state_count"] == result["runner_state_bytes"] == 0
        assert result["available"] == result["capacity"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)


def test_oversized_body_and_connection_limit_leave_engine_usable():
    with serving(max_body_bytes=64) as (model, address):
        assert complete(address, request_id="x" * 128)[0] == 400
        assert_clean(model)
    with serving() as (model, address):
        sockets = [socket.create_connection(address) for _ in range(6)]
        try:
            time.sleep(0.03)
            connection, response = request(address, "/health", method="GET")
            assert response.status == 503
            response.read()
            connection.close()
        finally:
            for connection in sockets:
                connection.close()
        time.sleep(0.03)
        assert complete(address)[0] == 200
        assert_clean(model)


def test_slow_stream_writer_timeout_releases_state_and_capacity():
    class LargeTokenizer(Tokenizer):
        def decode(self, tokens, **kwargs):
            return "x" * (4 * 1024 * 1024 * len(tokens))

    with serving(engine(tokenizer=LargeTokenizer())) as (model, address):
        connection, response = stream(address, request_id="slow-reader")
        # Deliberately do not consume the body. The bounded socket write must
        # terminate and release the request even while its client stays open.
        wait_clean(model)
        response.close()
        connection.close()
        ledger = model.stats()["terminal_requests"][-1]
        assert ledger["finish_reason"] == "disconnected"


def test_non_greedy_sampling_fails_before_model_execution():
    model = engine()
    with pytest.raises(ValueError, match="greedy"):
        model.generate([[1]], SamplingParams(temperature=1))
    assert not model.runner.model.calls
    assert_clean(model)


def test_acknowledged_cancel_wins_last_token_terminal_race(monkeypatch):
    model = engine()
    at_terminal, resume = threading.Event(), threading.Event()
    original = model._finish

    def barrier(request, reason):
        if reason == "length":
            at_terminal.set()
            assert resume.wait(2)
        original(request, reason)

    monkeypatch.setattr(model, "_finish", barrier)
    with serving(model) as (_, address), ThreadPoolExecutor(1) as pool:
        future = pool.submit(complete, address, request_id="last-token", max_tokens=1)
        try:
            assert at_terminal.wait(1)
            connection, response = request(address, "/v1/cancel/last-token")
            assert response.status == 200 and json.loads(response.read())["cancelled"]
            connection.close()
        finally:
            resume.set()
        assert future.result()[1]["choices"][0]["finish_reason"] == "cancelled"
        assert model.stats()["terminal_requests"][-1]["finish_reason"] == "cancelled"


def test_shutdown_preserves_previously_accepted_cancel_reason():
    model = engine()
    pending = model.submit([1], max_tokens=1, request_id="cancel-first")
    assert model.cancel("cancel-first")
    model.begin_shutdown()
    assert pending.step()["finish_reason"] == "cancelled"
    model.close()
    assert_clean(model)


def test_cancel_observation_and_terminal_peers_capture_exact_registry_boundary():
    model = engine()
    victim = model.submit([1], max_tokens=4, request_id="victim")
    peer = model.submit([3], max_tokens=4, request_id="peer")
    acknowledgement = model.cancel_with_observation("victim")
    assert acknowledgement == {"request_id": "victim", "cancelled": True,
                               "active_request_ids_before_cancel": ["victim", "peer"]}
    peer.close()
    victim.step()
    # The acknowledgement retains its own immutable list after peers terminate;
    # terminal metadata describes the later boundary, not the old observation.
    assert acknowledgement["active_request_ids_before_cancel"] == ["victim", "peer"]
    ledger = model.stats()["terminal_requests"]
    assert ledger[0]["active_peer_request_ids"] == ["victim"]
    assert ledger[1]["active_peer_request_ids"] == []
    assert model.cancel("unknown") is False
    assert model.cancel_with_observation("unknown") == {
        "request_id": "unknown", "cancelled": False, "active_request_ids_before_cancel": [],
    }
    assert_clean(model)


def test_disconnect_terminal_records_peer_at_release_and_http_cancel_returns_snapshot():
    with serving() as (model, address):
        victim = model.submit([1], max_tokens=4, request_id="victim")
        peer = model.submit([3], max_tokens=4, request_id="peer")
        victim.step()
        victim.close("disconnected")
        terminal = model.stats()["terminal_requests"][-1]
        assert terminal["finish_reason"] == "disconnected"
        assert terminal["active_peer_request_ids"] == ["peer"]
        connection, response = request(address, "/v1/cancel/peer")
        assert response.status == 200
        assert json.loads(response.read()) == {"request_id": "peer", "cancelled": True,
                                               "active_request_ids_before_cancel": ["peer"]}
        connection.close()
        peer.step()
        assert_clean(model)
