"""The batching and bookkeeping behind the hf backend, checked with a NumPy
stand-in for a language model (no torch needed)."""

from __future__ import annotations

import unittest

import numpy as np

from nashtune.games import make_game
from nashtune.policy.hf import render_chat
from nashtune.policy.scoring import (
    chunk_by_rows,
    collate_left,
    kept_index,
    plan_scoring,
    reduce_segments,
    surface_forms,
)
from nashtune.prompts import render

VOCAB = 64


def fake_logits(prefix: list[int]) -> np.ndarray:
    """Next-token logits that depend on the whole prefix (a seeded hash)."""
    seed = (len(prefix) * 1_000_003 + sum((i + 1) * t for i, t in enumerate(prefix))) % (2**32)
    return np.random.default_rng(seed).normal(size=VOCAB)


def log_softmax(x):
    x = x - x.max()
    return x - np.log(np.exp(x).sum())


def fake_batched_model(ids: np.ndarray, mask: np.ndarray, positions: np.ndarray, keep: int) -> np.ndarray:
    """What a causal LM does on a left-padded batch: position q of a row sees
    only that row's real tokens up to q. Returns logits for the last ``keep``
    positions."""
    out = np.zeros((ids.shape[0], keep, VOCAB))
    width = ids.shape[1]
    for b in range(ids.shape[0]):
        real = [int(t) for t, m in zip(ids[b], mask[b]) if m]
        for k in range(keep):
            q = width - keep + k
            n_real = int(positions[b, q]) + 1 if mask[b, q] else 0
            out[b, k] = fake_logits(real[:n_real])
    return out


def reference(prompt: list[int], forms: list[list[list[int]]]) -> tuple[np.ndarray, float]:
    """Sequential, unbatched: log P(form | prompt) summed over forms per action."""
    actions = []
    for per_action in forms:
        lps = []
        for form in per_action:
            lp = 0.0
            for t, tok in enumerate(form):
                lp += log_softmax(fake_logits(prompt + form[:t]))[tok]
            lps.append(lp)
        actions.append(np.logaddexp.reduce(lps))
    actions = np.array(actions)
    total = np.logaddexp.reduce(actions)
    return actions - total, float(np.exp(total))


class TestScoringPlan(unittest.TestCase):
    def check(self, prompts, forms):
        plan = plan_scoring(prompts, forms)
        ids, mask, pos = collate_left(plan.rows, pad_id=0)
        logits = fake_batched_model(ids, mask, pos, plan.keep)
        entry = np.array([log_softmax(logits[r, k])[t] for r, k, t in
                          zip(plan.entry_row, kept_index(plan), plan.entry_token)])
        logs, mass = reduce_segments(entry, plan, [len(f) for f in forms])
        for j in range(len(prompts)):
            ref_logs, ref_mass = reference(prompts[j], forms[j])
            np.testing.assert_allclose(logs[j], ref_logs, atol=1e-12)
            self.assertAlmostEqual(mass[j], ref_mass, places=12)
        return plan

    def test_single_token_answers_use_one_row_per_infoset(self):
        rng = np.random.default_rng(0)
        prompts = [list(rng.integers(1, VOCAB, size=n)) for n in (7, 3, 11)]
        forms = [[[[5], [6]], [[9]]], [[[5]], [[9], [10]], [[12]]], [[[2]], [[3]]]]
        plan = self.check(prompts, forms)
        self.assertEqual(len(plan.rows), 3)
        self.assertEqual(plan.keep, 1)

    def test_multi_token_answers(self):
        rng = np.random.default_rng(1)
        prompts = [list(rng.integers(1, VOCAB, size=n)) for n in (5, 9)]
        forms = [[[[5, 7, 8], [6]], [[9, 9]]], [[[1, 2]], [[1, 3], [4]], [[12, 13, 14, 15]]]]
        plan = self.check(prompts, forms)
        self.assertEqual(plan.keep, 4)
        # Forms that share their inputs share a row: [1, 2] and [1, 3] both feed prompt + [1].
        self.assertEqual(len(plan.rows), 2 + 4)

    def test_collate_left(self):
        ids, mask, pos = collate_left([[5, 6, 7], [8]], pad_id=0)
        np.testing.assert_array_equal(ids, [[5, 6, 7], [0, 0, 8]])
        np.testing.assert_array_equal(mask, [[1, 1, 1], [0, 0, 1]])
        np.testing.assert_array_equal(pos, [[0, 1, 2], [0, 0, 0]])

    def test_chunks(self):
        self.assertEqual(chunk_by_rows([1, 1, 1, 1, 1], 2), [[0, 1], [2, 3], [4]])
        self.assertEqual(chunk_by_rows([3, 1, 5, 1], 4), [[0, 1], [2], [3]])

    def test_surface_forms(self):
        self.assertEqual(surface_forms("call"), ["call", "Call"])
        self.assertEqual(surface_forms("call", "lower"), ["call"])
        self.assertEqual(surface_forms("fold", "both", leading_space=True), [" fold", " Fold"])
        with self.assertRaises(ValueError):
            surface_forms("call", "upper")


class FakeTokenizer:
    """Character-level tokenizer with a chat template, like the HF interface."""

    def __init__(self, system_ok: bool = True, template: bool = True):
        self.chat_template = "fake" if template else None
        self.system_ok = system_ok
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        self.calls.append(kwargs)
        if not self.system_ok and messages[0]["role"] == "system":
            raise ValueError("System role not supported")
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=True):
        ids = [ord(c) for c in text]
        return {"input_ids": ([1] if add_special_tokens else []) + ids}


class TestChatRendering(unittest.TestCase):
    def test_chat_template_and_thinking_switch(self):
        tok = FakeTokenizer()
        prompt = render(make_game("kuhn"), 0, "A")
        ids, chat = render_chat(tok, prompt)
        text = "".join(chr(i) for i in ids)
        self.assertTrue(chat)
        self.assertTrue(text.startswith("<system>You are playing Kuhn poker"))
        self.assertTrue(text.endswith("Your action?<assistant>"))
        self.assertEqual(tok.calls[0], {"enable_thinking": False})

    def test_system_message_merged_when_rejected(self):
        tok = FakeTokenizer(system_ok=False)
        ids, chat = render_chat(tok, render(make_game("kuhn"), 0, "A"))
        text = "".join(chr(i) for i in ids)
        self.assertTrue(text.startswith("<user>You are playing Kuhn poker"))
        self.assertIn("Legal actions: check, bet.", text)

    def test_no_chat_template(self):
        tok = FakeTokenizer(template=False)
        ids, chat = render_chat(tok, render(make_game("kuhn"), 0, "A"))
        self.assertFalse(chat)
        self.assertEqual(ids[0], 1)  # special tokens kept for a plain prompt
        self.assertTrue("".join(chr(i) for i in ids[1:]).endswith("\nAnswer:"))


if __name__ == "__main__":
    unittest.main()
