"""A local stand-in for the OpenAI Chat Completions API, for tests (no network).

Two models:

* ``fake-chat`` accepts everything and returns log-probabilities;
* ``fake-reasoner`` behaves like a reasoning model: it rejects ``max_tokens``
  (use ``max_completion_tokens``), any ``temperature`` other than 1,
  ``logprobs``, and ``n``.

``FakeOpenAI.fail_next`` makes the next request return HTTP 429 that many times.
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

WORDS = ("check", "bet", "fold", "call", "raise")


def legal_from(messages: list[dict]) -> list[str]:
    user = messages[-1]["content"]
    m = re.search(r"Legal actions: ([a-z, ]+)\.", user)
    if not m:
        m = re.search(r"Choose one of: ([a-z, ]+)\.", user)
    return [w.strip() for w in m.group(1).split(",")] if m else list(WORDS[:2])


def _error(param: str, code: str, message: str) -> dict:
    return {"error": {"message": message, "type": "invalid_request_error", "param": param, "code": code}}


class FakeOpenAI:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.fail_next = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _reply(self, status: int, payload: dict, headers: dict | None = None) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.headers.get("Authorization") != "Bearer test-key":
                    self._reply(401, {"error": {"message": "bad key"}})
                    return
                if self.path.endswith("/models"):
                    self._reply(200, {"data": [{"id": "fake-reasoner"}, {"id": "fake-chat"}]})
                else:
                    self._reply(404, {"error": {"message": "not found"}})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                server.requests.append(body)
                if server.fail_next > 0:
                    server.fail_next -= 1
                    self._reply(429, {"error": {"message": "slow down"}}, {"retry-after": "0"})
                    return
                status, payload = server.respond(body)
                self._reply(status, payload)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat_requests(self) -> int:
        return len(self.requests)

    # ----------------------------------------------------------- responses
    def respond(self, body: dict) -> tuple[int, dict]:
        model = body["model"]
        if model == "fake-reasoner":
            if "max_tokens" in body:
                return 400, _error("max_tokens", "unsupported_parameter",
                                   "Unsupported parameter: 'max_tokens' is not supported with this model. "
                                   "Use 'max_completion_tokens' instead.")
            if "temperature" in body and body["temperature"] != 1:
                return 400, _error("temperature", "unsupported_value",
                                   "Unsupported value: 'temperature' does not support 0 with this model.")
            for p in ("logprobs", "top_logprobs", "n"):
                if p in body:
                    return 400, _error(p, "unsupported_parameter", f"Unsupported parameter: '{p}'.")
        elif model != "fake-chat":
            return 404, {"error": {"message": f"model {model} does not exist"}}
        legal = legal_from(body["messages"])
        n = int(body.get("n", 1))
        prompt_tokens = sum(len(m["content"]) for m in body["messages"]) // 4
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 0}
        system = body["messages"][0]["content"]
        choices = []
        if body.get("logprobs"):
            top = [
                {"token": legal[0], "logprob": -0.5108256237659907},  # 0.6
                {"token": legal[-1].capitalize(), "logprob": -1.2039728043259361},  # 0.3
                {"token": "I", "logprob": -2.3025850929940455},  # 0.1, not a legal answer
            ]
            choices.append({"index": 0, "message": {"role": "assistant", "content": legal[0]},
                            "logprobs": {"content": [{"token": legal[0], "logprob": -0.51, "top_logprobs": top}]},
                            "finish_reason": "length"})
            usage["completion_tokens"] = 1
        elif "JSON" in system:
            probs = {legal[0]: 0.25, legal[-1]: 0.75}
            choices.append({"index": 0, "message": {"role": "assistant", "content": f"Sure: {json.dumps(probs)}"},
                            "finish_reason": "stop"})
            usage["completion_tokens"] = 12
        else:
            answers = [legal[-1].capitalize() + ".", legal[0], legal[-1], "Hmm, let me think"]
            for k in range(n):
                choices.append({"index": k, "message": {"role": "assistant", "content": answers[k % len(answers)]},
                                "finish_reason": "stop"})
            usage["completion_tokens"] = 3 * n
        if model == "fake-reasoner":
            usage["completion_tokens"] += 50 * len(choices)
            usage["completion_tokens_details"] = {"reasoning_tokens": 50 * len(choices)}
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return 200, {"id": "chatcmpl-fake", "object": "chat.completion", "model": model, "choices": choices,
                     "usage": usage}
