"""Bounded loopback HTTP/SSE completion service for the dedicated text engine.

Only greedy text completion is supported. Connections are closed after one
request. Cancellation takes effect at model call boundaries, never mid-kernel.
"""

from __future__ import annotations

import argparse
import json
import math
import select
import signal
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .engine import CapacityError, EngineClosedError


class CompletionServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True

    def __init__(self, address, engine, *, max_connections=16, max_body_bytes=1048576, socket_timeout=30):
        if type(max_connections) is not int or max_connections < engine.runner.max_active_requests + 2:
            raise ValueError("max_connections must reserve at least two control slots beyond active requests")
        if type(max_body_bytes) is not int or max_body_bytes <= 0 or not math.isfinite(socket_timeout) or socket_timeout <= 0:
            raise ValueError("HTTP bounds must be positive")
        self.engine = engine
        self.max_body_bytes = max_body_bytes
        self.socket_timeout = socket_timeout
        self._slots = threading.BoundedSemaphore(max_connections)
        self._connections_lock = threading.Lock()
        self._connections = {}
        super().__init__(address, CompletionHandler)

    def process_request(self, request, client_address):
        request.settimeout(self.socket_timeout)
        if not self._slots.acquire(blocking=False):
            try:
                request.settimeout(min(self.socket_timeout, 1))
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            except OSError:
                pass
            finally:
                self.shutdown_request(request)
            return
        try:
            with self._connections_lock:
                self._connections[request] = False
            super().process_request(request, client_address)
        except BaseException:
            with self._connections_lock:
                self._connections.pop(request, None)
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._connections_lock:
                self._connections.pop(request, None)
            self._slots.release()

    def _close_connections(self, *, idle_only=False):
        with self._connections_lock:
            sockets = [connection for connection, generating in self._connections.items()
                       if not idle_only or not generating]
        for connection in sockets:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def server_close(self):
        self.engine.begin_shutdown()
        # Slow headers/bodies have no generation to drain. Active streams get
        # one socket-timeout interval to send terminal framing before transport
        # shutdown unblocks writers. CUDA calls remain non-preemptible.
        self._close_connections(idle_only=True)
        deadline = threading.Timer(self.socket_timeout, self._close_connections)
        deadline.daemon = True
        deadline.start()
        try:
            super().server_close()
        finally:
            deadline.cancel()

    def handle_error(self, request, client_address):
        # Client addresses and arbitrary exceptions do not belong in service logs.
        pass


class CompletionHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        pass

    def _json(self, status, value):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(payload)
        self.wfile.flush()

    def _error(self, status, message):
        self._json(status, {"error": {"message": message, "type": "request_error"}})

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "closed" if self.server.engine.stats()["closed"] else "ready",
                             **self.server.engine.provenance()})
        elif self.path in {"/debug/state", "/stats"}:
            self._json(200, {**self.server.engine.stats(), **self.server.engine.provenance()})
        else:
            self._error(404, "unknown route")

    def _body(self):
        if self.headers.get("Transfer-Encoding") is not None:
            raise ValueError("chunked request bodies are unsupported")
        lengths = self.headers.get_all("Content-Length", [])
        if len(lengths) != 1 or not lengths[0].isdigit():
            raise ValueError("one Content-Length header is required")
        length = int(lengths[0])
        if length <= 0 or length > self.server.max_body_bytes:
            raise ValueError("request body exceeds configured bounds")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ValueError("incomplete request body")

        def unique(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate JSON fields")
                value[key] = item
            return value

        body = json.loads(raw, object_pairs_hook=unique,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON value")))
        if not isinstance(body, dict):
            raise TypeError("request must be a JSON object")
        return body

    def _disconnected(self):
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except OSError:
            return True

    def _sse(self, value):
        payload = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, allow_nan=False)
        self.wfile.write(("data: " + payload + "\n\n").encode())
        self.wfile.flush()

    def _chunk(self, request, text, token_ids, finish_reason=None):
        return {"id": request.request_id, "request_id": request.request_id,
                "object": "text_completion", "model": self.server.engine.model_id,
                "choices": [{"index": 0, "text": text, "token_ids": token_ids,
                             "finish_reason": finish_reason}]}

    def do_POST(self):
        request = None
        streaming = False
        headers_sent = False
        try:
            if self.path.startswith("/v1/cancel/"):
                request_id = self.path.removeprefix("/v1/cancel/")
                accepted = self.server.engine.cancel(request_id)
                self._json(200 if accepted else 404, {"request_id": request_id, "cancelled": accepted})
                return
            if self.path != "/v1/completions":
                self._error(404, "unknown route")
                return
            body = self._body()
            allowed = {"model", "prompt", "input_ids", "max_tokens", "temperature", "stream", "request_id"}
            if set(body) - allowed:
                raise ValueError("unsupported completion fields")
            if body.get("model", self.server.engine.model_id) != self.server.engine.model_id:
                raise ValueError("unsupported model")
            if ("prompt" in body) == ("input_ids" in body):
                raise ValueError("provide exactly one prompt or input_ids")
            if isinstance(body.get("temperature", 0), bool) or body.get("temperature", 0) != 0:
                raise ValueError("only temperature=0 greedy completion is supported")
            streaming = body.get("stream", False)
            if type(streaming) is not bool:
                raise ValueError("stream must be boolean")
            prompt = body.get("prompt", body.get("input_ids"))
            if "input_ids" in body and not isinstance(prompt, list):
                raise ValueError("input_ids must be a token list")
            if self._disconnected():
                return
            request = self.server.engine.submit(prompt, max_tokens=body.get("max_tokens"), request_id=body.get("request_id"))
            with self.server._connections_lock:
                self.server._connections[self.connection] = True
            if streaming:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.send_header("X-Request-ID", request.request_id)
                self.end_headers()
                self.wfile.flush()
                self.close_connection = True
                headers_sent = True
            while request.finish_reason is None:
                if self._disconnected():
                    request.close("disconnected")
                    return
                event = request.step(disconnect_check=self._disconnected)
                if streaming and (event["token_id"] is not None or event["delta"]):
                    self._sse(self._chunk(request, event["delta"], [] if event["token_id"] is None else [event["token_id"]]))
            if streaming:
                self._sse(self._chunk(request, "", [], request.finish_reason))
                self._sse("[DONE]")
            else:
                result = self._chunk(request, request.text, request.token_ids, request.finish_reason)
                result["usage"] = {"prompt_tokens": len(request.prompt_ids),
                                   "completion_tokens": len(request.token_ids),
                                   "total_tokens": len(request.prompt_ids) + len(request.token_ids)}
                self._json(200, result)
        except (BrokenPipeError, ConnectionError, TimeoutError, OSError):
            if request is not None and request.finish_reason == "error":
                if headers_sent:
                    self._sse({"error": {"message": "generation failed", "type": "generation_error"}})
                    self._sse(self._chunk(request, "", [], "error"))
                    self._sse("[DONE]")
                else:
                    self._error(500, "generation failed")
            elif request is not None:
                request.close("disconnected")
        except EngineClosedError:
            self._error(503, "engine is closed")
        except (ValueError, TypeError, CapacityError) as exc:
            if request is not None:
                request.close("error")
            if headers_sent:
                self._sse({"error": {"message": "generation failed", "type": "generation_error"}})
                self._sse(self._chunk(request, "", [], "error"))
                self._sse("[DONE]")
            elif request is not None:
                self._error(500, "generation failed")
            else:
                self._error(429 if isinstance(exc, CapacityError) else 400, str(exc))
        except Exception:  # noqa: BLE001 - isolate third-party runtime failures per request.
            if request is not None:
                request.close("error")
            if headers_sent:
                self._sse({"error": {"message": "generation failed", "type": "generation_error"}})
                self._sse(self._chunk(request, "", [], "error"))
                self._sse("[DONE]")
            else:
                self._error(500, "generation failed")
        finally:
            if request is not None:
                request.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-context-tokens", type=int, required=True)
    parser.add_argument("--max-active-requests", type=int, required=True)
    parser.add_argument("--max-connections", type=int, default=16)
    parser.add_argument("--max-body-bytes", type=int, default=1048576)
    parser.add_argument("--socket-timeout", type=float, default=30)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--shutdown-file", type=Path)
    args = parser.parse_args()
    for output in (args.ready_file, args.shutdown_file):
        if output is not None and output.exists():
            parser.error("refusing to overwrite a service evidence file")
    from nanovllm import LLM

    engine = LLM(args.model, max_context_tokens=args.max_context_tokens,
                 max_active_requests=args.max_active_requests)
    server = CompletionServer((args.host, args.port), engine, max_connections=args.max_connections,
                              max_body_bytes=args.max_body_bytes, socket_timeout=args.socket_timeout)

    def drain():
        engine.begin_shutdown()
        server.shutdown()

    shutdown_started = False

    def shutdown(signum, frame):
        nonlocal shutdown_started
        if not shutdown_started:
            shutdown_started = True
            # No engine lock in a signal handler, and shutdown never runs on
            # the serve_forever thread that it must wait for.
            threading.Thread(target=drain, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    if args.ready_file is not None:
        with args.ready_file.open("x") as handle:
            json.dump({"address": list(server.server_address), **engine.provenance(), **engine.stats()}, handle)
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        engine.begin_shutdown()
        server.server_close()
        engine.close()
        if args.shutdown_file is not None:
            with args.shutdown_file.open("x") as handle:
                json.dump({**engine.provenance(), **engine.stats()}, handle)


if __name__ == "__main__":
    main()
