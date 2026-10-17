"""OpenAI models as poker strategies, through the Chat Completions HTTP API
(plain ``requests``; no SDK).

Three ways to read a strategy from a model:

* ``logprobs``: one output token with ``logprobs``/``top_logprobs``. The
  probability of an action is the total probability of first tokens that spell
  its word (any case, surrounding spaces or punctuation ignored). Exact up to
  the top-k cut-off; the share of probability that landed on legal answers is
  reported as the legal mass. Needs a model that returns log-probabilities.
* ``sample``: n answers at temperature 1; the strategy is the empirical
  frequency of each legal first word. Works for reasoning models. Invalid
  answers are counted and excluded. With n samples the frequencies are noisy,
  and noise itself makes a strategy more exploitable: the measured number is an
  estimate of the exploitability of the sampled strategy, not of the model's
  exact one.
* ``stated``: the model is asked for its mixed strategy as JSON.

Requests are cached on disk by a hash of the request, so an interrupted run
resumes without paying twice. Token use is counted from each response's
``usage``; ``max_total_tokens`` stops a run before it can exceed a budget.
Parameters a model rejects (for example ``temperature`` or ``max_tokens`` on
reasoning models) are dropped or renamed and the request retried; the
adjustment is remembered per model.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..games.game import Game
from ..prompts import Prompt, parse_action_word, render
from .base import Extraction, Policy

MODES = ("logprobs", "sample", "stated")
DROPPABLE = ("temperature", "top_p", "n", "reasoning_effort", "presence_penalty", "frequency_penalty", "seed",
             "logprobs", "top_logprobs")
RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504)


class OpenAIError(RuntimeError):
    pass


class BudgetExceeded(OpenAIError):
    pass


class UnsupportedParameter(OpenAIError):
    def __init__(self, param: str, message: str) -> None:
        super().__init__(f"the model rejected required parameter '{param}': {message}")
        self.param = param


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Rough count without a tokenizer: about four characters per token, plus
    a few tokens of overhead per message."""
    return int(sum(len(m["content"]) for m in messages) / 4) + 4 * len(messages) + 3


def output_cap(body: dict) -> int:
    cap = body.get("max_completion_tokens", body.get("max_tokens", 256))
    return int(cap) * int(body.get("n", 1))


@dataclass
class Usage:
    requests: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    retries: int = 0
    adjustments: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {"requests": self.requests, "cache_hits": self.cache_hits, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "reasoning_tokens": self.reasoning_tokens,
                "total_tokens": self.total_tokens, "retries": self.retries, "adjustments": self.adjustments}


class OpenAIClient:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, cache_dir: str | os.PathLike = "cache/openai",
                 max_total_tokens: int | None = None, session=None, timeout: float = 120.0, max_retries: int = 6,
                 echo=print, sleep=time.sleep) -> None:
        import requests

        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.max_total_tokens = max_total_tokens
        self.session = session or requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self.echo = echo
        self.sleep = sleep
        self.usage = Usage()
        self._quirks_path = self.cache_dir / "model_params.json"
        self.quirks: dict = json.loads(self._quirks_path.read_text()) if self._quirks_path.exists() else {}

    # ---------------------------------------------------------------- http
    def _headers(self) -> dict:
        if not self.api_key:
            raise OpenAIError("OPENAI_API_KEY is not set")
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _send(self, method: str, path: str, body: dict | None = None) -> dict:
        import requests

        url = f"{self.base_url}{path}"
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.request(method, url, headers=self._headers(), json=body, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt == self.max_retries:
                    raise OpenAIError(f"network error after {attempt + 1} attempts: {exc}") from None
                self.usage.retries += 1
                self.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if resp.status_code in RETRY_STATUS and attempt < self.max_retries:
                self.usage.retries += 1
                wait = resp.headers.get("retry-after")
                try:
                    pause = float(wait) if wait else delay
                except ValueError:
                    pause = delay
                self.sleep(min(max(pause, 0.1), 120))
                delay = min(delay * 2, 60)
                continue
            try:
                data = resp.json()
            except ValueError:
                data = {"error": {"message": resp.text[:500]}}
            if resp.status_code >= 400:
                err = data.get("error", {}) if isinstance(data, dict) else {}
                raise _HTTPError(resp.status_code, err if isinstance(err, dict) else {"message": str(err)})
            return data
        raise OpenAIError("unreachable")

    def models(self) -> list[str]:
        try:
            data = self._send("GET", "/models")
        except _HTTPError as exc:
            raise OpenAIError(f"listing models failed ({exc.status}): {exc.message}") from None
        return sorted(m["id"] for m in data.get("data", []))

    # ------------------------------------------------------- request logic
    def _apply_quirks(self, body: dict) -> dict:
        body = dict(body)
        q = self.quirks.get(body.get("model", ""), {})
        for p in q.get("drop", []):
            body.pop(p, None)
        if q.get("rename") == "max_completion_tokens" and "max_tokens" in body:
            body["max_completion_tokens"] = body.pop("max_tokens")
        if q.get("rename") == "max_tokens" and "max_completion_tokens" in body:
            body["max_tokens"] = body.pop("max_completion_tokens")
        return body

    def _remember(self, model: str, change: dict) -> None:
        q = self.quirks.setdefault(model, {})
        if "drop" in change:
            q.setdefault("drop", [])
            if change["drop"] not in q["drop"]:
                q["drop"].append(change["drop"])
        if "rename" in change:
            q["rename"] = change["rename"]
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._quirks_path.write_text(json.dumps(self.quirks, indent=1, sort_keys=True) + "\n")

    @staticmethod
    def _offending_param(err: dict, body: dict) -> str | None:
        param = err.get("param")
        if isinstance(param, str) and param in body:
            return param
        message = str(err.get("message", ""))
        for name in sorted(body, key=len, reverse=True):
            if name in ("model", "messages"):
                continue
            if re.search(rf"['\"`]{re.escape(name)}['\"`]", message) or f" {name} " in f" {message} ":
                return name
        return None

    def cache_key(self, body: dict, extra: str = "") -> str:
        blob = json.dumps(body, sort_keys=True, separators=(",", ":")) + "|" + extra
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def chat(self, body: dict, required: tuple[str, ...] = (), cache_extra: str = "") -> dict:
        key = self.cache_key(body, cache_extra)
        path = self._cache_path(key)
        if path.exists():
            self.usage.cache_hits += 1
            return json.loads(path.read_text())["response"]
        sent = self._apply_quirks(body)
        for missing in required:
            if missing not in sent:
                raise UnsupportedParameter(missing, f"{body.get('model')} is known not to accept it")
        for _ in range(8):
            if self.max_total_tokens is not None:
                projected = self.usage.total_tokens + estimate_prompt_tokens(sent["messages"]) + output_cap(sent)
                if projected > self.max_total_tokens:
                    raise BudgetExceeded(
                        f"stopping before the token budget is exceeded: {self.usage.total_tokens} tokens used, "
                        f"next request could use up to {projected - self.usage.total_tokens}, "
                        f"budget {self.max_total_tokens}. Finished requests are cached; rerun with a larger "
                        "--max-total-tokens to continue.")
            try:
                response = self._send("POST", "/chat/completions", sent)
            except _HTTPError as exc:
                if exc.status != 400:
                    raise OpenAIError(f"HTTP {exc.status}: {exc.message}") from None
                param = self._offending_param(exc.error, sent)
                model = body.get("model", "")
                if param in required:
                    raise UnsupportedParameter(param, exc.message) from None
                if param == "max_tokens":
                    sent["max_completion_tokens"] = sent.pop("max_tokens")
                    change = {"rename": "max_completion_tokens"}
                elif param == "max_completion_tokens":
                    sent["max_tokens"] = sent.pop("max_completion_tokens")
                    change = {"rename": "max_tokens"}
                elif param in DROPPABLE:
                    sent.pop(param)
                    change = {"drop": param}
                else:
                    raise OpenAIError(f"HTTP 400: {exc.message}") from None
                self._remember(model, change)
                note = f"{model}: {exc.message[:120]} -> {change}"
                self.usage.adjustments.append(note)
                self.echo(f"[openai] {note}")
                continue
            self.usage.requests += 1
            usage = response.get("usage") or {}
            self.usage.prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.usage.completion_tokens += int(usage.get("completion_tokens", 0))
            details = usage.get("completion_tokens_details") or {}
            self.usage.reasoning_tokens += int(details.get("reasoning_tokens", 0) or 0)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"request": body, "sent": sent, "response": response}) + "\n")
            os.replace(tmp, path)
            return response
        raise OpenAIError("too many parameter adjustments for one request")


class _HTTPError(Exception):
    def __init__(self, status: int, error: dict) -> None:
        self.status = status
        self.error = error
        self.message = str(error.get("message", ""))
        super().__init__(f"{status}: {self.message}")


# ------------------------------------------------------------- parsing
def _norm_token(token: str) -> str:
    return token.strip().strip("*_`\"'.,;:!?()[]").lower()


def probs_from_top_logprobs(top: list[dict], legal: tuple[str, ...]) -> tuple[np.ndarray, float, float]:
    """Probability of each legal action from the first token's top
    log-probabilities. A token that is a strict prefix (two letters or more) of
    exactly one legal word counts for that word. Returns (unnormalized
    probabilities, legal mass, mass from prefix matches)."""
    p = np.zeros(len(legal))
    prefix_mass = 0.0
    for entry in top:
        tok = _norm_token(entry.get("token", ""))
        if not tok:
            continue
        prob = math.exp(float(entry["logprob"]))
        if tok in legal:
            p[legal.index(tok)] += prob
            continue
        if len(tok) >= 2:
            owners = [k for k, w in enumerate(legal) if w.startswith(tok)]
            if len(owners) == 1:
                p[owners[0]] += prob
                prefix_mass += prob
    return p, float(p.sum()), prefix_mass


def parse_stated(text: str, legal: tuple[str, ...]) -> np.ndarray | None:
    """The first JSON object in ``text`` as probabilities over ``legal``
    (normalized; negative values treated as zero), or None if unusable."""
    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.S):
        try:
            data = json.loads(match.group(0))
        except ValueError:
            continue
        if not isinstance(data, dict):
            continue
        p = np.zeros(len(legal))
        found = False
        for k, v in data.items():
            word = _norm_token(str(k))
            if word in legal:
                try:
                    value = float(str(v).strip().rstrip("%"))
                except ValueError:
                    continue
                if math.isfinite(value):
                    p[legal.index(word)] = max(value, 0.0)
                    found = True
        if found and p.sum() > 0:
            return p / p.sum()
    return None


# -------------------------------------------------------------- policy
class OpenAIPolicy(Policy):
    """Extraction only: an OpenAI model's strategy for every infoset."""

    def __init__(self, game: Game, client: OpenAIClient, model: str, mode: str = "logprobs", template: str = "A",
                 samples: int = 8, top_logprobs: int = 20, max_completion_tokens: int | None = None,
                 reasoning_effort: str | None = None, echo=print) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.game = game
        self.client = client
        self.model = model
        self.mode = mode
        self.template = template
        self.samples = samples
        self.top_logprobs = top_logprobs
        self.max_completion_tokens = max_completion_tokens or {"logprobs": 1, "sample": 16, "stated": 200}[mode]
        self.reasoning_effort = reasoning_effort
        self.echo = echo

    def prompt(self, info) -> Prompt:
        return render(self.game, info, self.template, answer="json" if self.mode == "stated" else "word")

    def request_body(self, info, n: int | None = None) -> dict:
        p = self.prompt(info)
        body: dict = {"model": self.model, "messages": p.messages()}
        if self.mode == "logprobs":
            body.update({"max_tokens": 1, "logprobs": True, "top_logprobs": self.top_logprobs})
        elif self.mode == "sample":
            body.update({"max_completion_tokens": self.max_completion_tokens, "temperature": 1.0})
            if (n or self.samples) > 1:
                body["n"] = n or self.samples
        else:
            body.update({"max_completion_tokens": self.max_completion_tokens, "temperature": 0.0})
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def plan(self) -> dict:
        """Requests and token estimates for a dry run (before any call)."""
        prompt_tokens = 0
        output = 0
        for info in self.game.infosets:
            body = self.request_body(info)
            prompt_tokens += estimate_prompt_tokens(body["messages"])
            output += output_cap(body)
        return {"model": self.model, "mode": self.mode, "game": self.game.name, "template": self.template,
                "requests": self.game.num_infosets, "samples_per_infoset": self.samples if self.mode == "sample" else 1,
                "estimated_prompt_tokens": prompt_tokens, "max_output_tokens": output}

    def _one(self, info) -> tuple[np.ndarray, dict]:
        legal = self.prompt(info).actions
        uniform = np.full(len(legal), 1.0 / len(legal))
        if self.mode == "logprobs":
            resp = self.client.chat(self.request_body(info), required=("logprobs", "top_logprobs"))
            try:
                top = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            except (KeyError, IndexError, TypeError):
                raise UnsupportedParameter("logprobs", "the response has no log-probabilities") from None
            p, mass, prefix = probs_from_top_logprobs(top, legal)
            if mass <= 0:
                return uniform, {"legal_mass": 0.0, "fallback_uniform": 1, "prefix_mass": 0.0}
            return p / mass, {"legal_mass": mass, "fallback_uniform": 0, "prefix_mass": prefix}
        if self.mode == "stated":
            resp = self.client.chat(self.request_body(info))
            text = (resp["choices"][0].get("message") or {}).get("content") or ""
            p = parse_stated(text, legal)
            if p is None:
                return uniform, {"invalid": 1, "fallback_uniform": 1}
            return p, {"invalid": 0, "fallback_uniform": 0}
        # sample
        body = self.request_body(info)
        resp = self.client.chat(body)
        choices = list(resp.get("choices", []))
        k = 0
        while len(choices) < self.samples:  # the model ignored or rejected n: ask one at a time
            single = self.request_body(info, n=1)
            more = self.client.chat(single, cache_extra=f"sample-{k}")
            choices.extend(more.get("choices", [])[:1])
            k += 1
        counts = np.zeros(len(legal))
        invalid = truncated = 0
        for choice in choices[: self.samples]:
            text = (choice.get("message") or {}).get("content") or ""
            word = parse_action_word(text, legal)
            if word is None:
                invalid += 1
                if choice.get("finish_reason") == "length" and not text.strip():
                    truncated += 1
            else:
                counts[legal.index(word)] += 1
        valid = counts.sum()
        stats = {"valid": int(valid), "invalid": invalid, "truncated": truncated,
                 "fallback_uniform": int(valid == 0)}
        return (counts / valid if valid else uniform), stats

    def extract(self) -> Extraction:
        start = time.time()
        strategy = np.empty(self.game.action_slots)
        rows = []
        mass = []
        n = self.game.num_infosets
        for k, info in enumerate(self.game.infosets):
            p, stats = self._one(info)
            strategy[info.offset : info.offset + info.action_count] = p
            rows.append(stats)
            mass.append(stats.get("legal_mass", float("nan")))
            if (k + 1) % 25 == 0 or k + 1 == n:
                u = self.client.usage
                self.echo(f"[openai] {self.model} {self.mode} {self.game.name}/{self.template}: {k + 1}/{n} infosets, "
                          f"{u.requests} requests, {u.cache_hits} cached, {u.total_tokens} tokens")
        totals: dict = {}
        for r in rows:
            for key, v in r.items():
                if key != "legal_mass":
                    totals[key] = totals.get(key, 0) + v
        legal_mass = np.array(mass) if self.mode == "logprobs" else None
        return Extraction(strategy=strategy, legal_mass=legal_mass, seconds=time.time() - start,
                          extra={"per_infoset": rows, "totals": totals, "usage": self.client.usage.as_dict()})

    def describe(self) -> dict:
        return {"backend": "openai", "model": self.model, "mode": self.mode, "template": self.template,
                "samples": self.samples if self.mode == "sample" else 1, "top_logprobs": self.top_logprobs,
                "max_completion_tokens": self.max_completion_tokens, "reasoning_effort": self.reasoning_effort}
