"""The OpenAI backend against a local fake server (no network)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent))

from fake_openai import FakeOpenAI  # noqa: E402

from nashtune.experiments.openai_plan import dry_run, load_plan, run_plan  # noqa: E402
from nashtune.games import make_game  # noqa: E402
from nashtune.policy.openai_backend import (  # noqa: E402
    BudgetExceeded,
    OpenAIClient,
    OpenAIPolicy,
    UnsupportedParameter,
    parse_stated,
    probs_from_top_logprobs,
)


def local_session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False  # never send localhost traffic to a proxy
    return s


class OpenAITestCase(unittest.TestCase):
    def setUp(self):
        self.server = FakeOpenAI()
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "cache"
        self.messages: list[str] = []

    def tearDown(self):
        self.server.close()
        self.tmp.cleanup()

    def client(self, **kw) -> OpenAIClient:
        return OpenAIClient(api_key="test-key", base_url=self.server.base_url, cache_dir=self.cache,
                            session=local_session(), echo=self.messages.append, sleep=lambda s: None, **kw)


class TestModes(OpenAITestCase):
    def test_list_models(self):
        self.assertEqual(self.client().models(), ["fake-chat", "fake-reasoner"])

    def test_logprobs_mode(self):
        game = make_game("kuhn")
        policy = OpenAIPolicy(game, self.client(), "fake-chat", mode="logprobs", echo=self.messages.append)
        ext = policy.extract()
        game.check_strategy(ext.strategy)
        for info in game.infosets:
            p = ext.strategy[info.offset : info.offset + info.action_count]
            np.testing.assert_allclose(p, [2 / 3, 1 / 3])  # 0.6 and 0.3, renormalized
        np.testing.assert_allclose(ext.legal_mass, 0.9)
        self.assertEqual(self.server.chat_requests(), game.num_infosets)
        sent = self.server.requests[0]
        self.assertEqual((sent["max_tokens"], sent["logprobs"], sent["top_logprobs"]), (1, True, 20))

    def test_sample_mode(self):
        game = make_game("kuhn")
        policy = OpenAIPolicy(game, self.client(), "fake-chat", mode="sample", samples=4, echo=self.messages.append)
        ext = policy.extract()
        info = game.infoset_by_key("1:Q:-:b")  # fold, call
        np.testing.assert_allclose(ext.strategy[info.offset : info.offset + 2], [1 / 3, 2 / 3])
        self.assertEqual(ext.extra["totals"]["invalid"], game.num_infosets)  # one "Hmm" per infoset
        self.assertEqual(self.server.requests[0]["n"], 4)

    def test_stated_mode(self):
        game = make_game("leduc")
        policy = OpenAIPolicy(game, self.client(), "fake-chat", mode="stated", echo=self.messages.append)
        ext = policy.extract()
        for info in game.infosets[:30]:
            p = ext.strategy[info.offset : info.offset + info.action_count]
            self.assertAlmostEqual(p[0], 0.25)
            self.assertAlmostEqual(p[-1], 0.75)
        self.assertIn("JSON", self.server.requests[0]["messages"][0]["content"])

    def test_cache_makes_reruns_free(self):
        game = make_game("kuhn")
        first = OpenAIPolicy(game, self.client(), "fake-chat", mode="logprobs", echo=self.messages.append).extract()
        count = self.server.chat_requests()
        client = self.client()
        second = OpenAIPolicy(game, client, "fake-chat", mode="logprobs", echo=self.messages.append).extract()
        self.assertEqual(self.server.chat_requests(), count)
        self.assertEqual(client.usage.cache_hits, game.num_infosets)
        self.assertEqual(client.usage.total_tokens, 0)
        np.testing.assert_array_equal(first.strategy, second.strategy)

    def test_token_guard(self):
        game = make_game("leduc")
        client = self.client(max_total_tokens=1500)
        policy = OpenAIPolicy(game, client, "fake-chat", mode="logprobs")
        with self.assertRaises(BudgetExceeded):
            policy.extract()
        self.assertLessEqual(client.usage.total_tokens, 1500)
        self.assertGreater(client.usage.requests, 0)

    def test_usage_accounting(self):
        game = make_game("kuhn")
        client = self.client()
        OpenAIPolicy(game, client, "fake-reasoner", mode="sample", samples=2, echo=self.messages.append).extract()
        self.assertEqual(client.usage.reasoning_tokens, 50 * 2 * game.num_infosets)
        self.assertEqual(client.usage.completion_tokens, (50 + 3) * 2 * game.num_infosets)


class TestParameterRejection(OpenAITestCase):
    def test_reasoning_model_sample_mode(self):
        """n is rejected: dropped, remembered, and replaced by single requests."""
        game = make_game("kuhn")
        client = self.client()
        policy = OpenAIPolicy(game, client, "fake-reasoner", mode="sample", samples=3, reasoning_effort="low", echo=self.messages.append)
        ext = policy.extract()
        game.check_strategy(ext.strategy)
        info = game.infoset_by_key("1:Q:-:b")
        np.testing.assert_allclose(ext.strategy[info.offset : info.offset + 2], [0.0, 1.0])  # always "Call."
        quirks = json.loads((self.cache / "model_params.json").read_text())
        self.assertIn("n", quirks["fake-reasoner"]["drop"])
        # After the first rejection no request carries n again.
        accepted = [r for r in self.server.requests[1:] if r["model"] == "fake-reasoner"]
        self.assertTrue(all("n" not in r for r in accepted))
        self.assertTrue(any("n" in note for note in client.usage.adjustments))

    def test_max_tokens_renamed_and_temperature_dropped(self):
        game = make_game("kuhn")
        client = self.client()
        policy = OpenAIPolicy(game, client, "fake-reasoner", mode="stated", echo=self.messages.append)
        ext = policy.extract()
        game.check_strategy(ext.strategy)
        last = self.server.requests[-1]
        self.assertNotIn("temperature", last)
        self.assertIn("max_completion_tokens", last)

    def test_logprobs_rejected_is_a_clear_error(self):
        game = make_game("kuhn")
        client = self.client()
        policy = OpenAIPolicy(game, client, "fake-reasoner", mode="logprobs")
        with self.assertRaises(UnsupportedParameter) as ctx:
            policy.extract()
        self.assertIn("logprobs", str(ctx.exception))
        quirks = json.loads((self.cache / "model_params.json").read_text())
        self.assertEqual(quirks["fake-reasoner"]["rename"], "max_completion_tokens")

    def test_retry_after_rate_limit(self):
        self.server.fail_next = 2
        client = self.client()
        OpenAIPolicy(make_game("kuhn"), client, "fake-chat", mode="logprobs", echo=self.messages.append).extract()
        self.assertEqual(client.usage.retries, 2)

    def test_unknown_model(self):
        from nashtune.policy.openai_backend import OpenAIError

        with self.assertRaises(OpenAIError):
            OpenAIPolicy(make_game("kuhn"), self.client(), "no-such-model", mode="logprobs").extract()


class TestParsing(unittest.TestCase):
    def test_top_logprobs(self):
        top = [{"token": " Call", "logprob": np.log(0.5)}, {"token": "fold.", "logprob": np.log(0.2)},
               {"token": "ra", "logprob": np.log(0.1)}, {"token": "**", "logprob": np.log(0.1)}]
        p, mass, prefix = probs_from_top_logprobs(top, ("fold", "call", "raise"))
        np.testing.assert_allclose(p, [0.2, 0.5, 0.1])
        self.assertAlmostEqual(mass, 0.8)
        self.assertAlmostEqual(prefix, 0.1)

    def test_stated(self):
        legal = ("fold", "call", "raise")
        np.testing.assert_allclose(parse_stated('{"Fold": 1, "call": 3}', legal), [0.25, 0.75, 0.0])
        np.testing.assert_allclose(parse_stated('text {"raise": "40%", "call": "60%"} more', legal), [0, 0.6, 0.4])
        self.assertIsNone(parse_stated("call", legal))
        self.assertIsNone(parse_stated('{"check": 1}', legal))


class TestPlan(OpenAITestCase):
    def write_plan(self):
        plan = {"budget_usd": 1.0, "runs": [
            {"model": "$CHAT_MODEL", "mode": "logprobs", "game": "kuhn", "template": "A", "expected_output_tokens": 1},
            {"model": "$CHAT_MODEL", "mode": "logprobs", "game": "kuhn", "template": "B", "expected_output_tokens": 1},
            {"model": "$REASONING_MODEL", "mode": "sample", "game": "kuhn", "template": "A", "samples": 2,
             "max_completion_tokens": 100, "expected_output_tokens": 60},
        ]}
        path = Path(self.tmp.name) / "plan.json"
        path.write_text(json.dumps(plan))
        return path

    def test_dry_run_and_run(self):
        plan = load_plan(self.write_plan(), "fake-chat", "fake-reasoner")
        prices = {"fake-chat": {"input": 1.0, "output": 4.0}, "fake-reasoner": {"input": 2.0, "output": 8.0}}
        out = Path(self.tmp.name) / "openai"
        est = dry_run(plan, prices, out, echo=self.messages.append)
        self.assertEqual(est["missing_prices"], [])
        self.assertGreater(est["bound_usd"], est["likely_usd"])
        self.assertEqual(self.server.chat_requests(), 0)  # a dry run sends nothing
        kw = {"api_key": "test-key", "base_url": self.server.base_url, "session": local_session(),
              "sleep": lambda s: None}
        results = run_plan(plan, prices, out, echo=self.messages.append, client_kwargs=kw)
        self.assertEqual(len(results), 3)
        ledger = json.loads((out / "spend.json").read_text())
        self.assertEqual(set(ledger), {"fake-chat", "fake-reasoner"})
        self.assertTrue((out / "fake-chat_logprobs_kuhn_B.json").exists())
        count = self.server.chat_requests()
        self.assertEqual(run_plan(plan, prices, out, echo=self.messages.append, client_kwargs=kw), [])
        self.assertEqual(self.server.chat_requests(), count)

    def test_missing_model_skips_its_runs(self):
        plan = load_plan(self.write_plan(), "fake-chat", None)
        self.assertEqual(len(plan["runs"]), 2)


if __name__ == "__main__":
    unittest.main()
