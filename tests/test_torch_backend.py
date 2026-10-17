"""The torch/transformers/peft backend, with a tiny randomly initialized Qwen2
model and tokenizers built inside the test (no downloads).

Skipped when torch, transformers, peft or tokenizers is missing. CI installs
CPU torch and runs these on Linux and macOS.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAVE_TORCH = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "peft", "tokenizers"))
if os.environ.get("NASHTUNE_REQUIRE_TORCH") and not HAVE_TORCH:
    raise ImportError("NASHTUNE_REQUIRE_TORCH is set but torch, transformers, peft or tokenizers is missing")

if HAVE_TORCH:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from nashtune.games import make_game
from nashtune.grpo import BatchBuilder, log_softmax, loss_and_grad
from nashtune.prompts import ACTION_WORDS, render
from nashtune.train import TrainConfig, Trainer, quiet

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def corpus() -> list[str]:
    texts = ["<|system|> <|user|> <|assistant|>"]
    for name in ("kuhn", "leduc"):
        game = make_game(name)
        for template in ("A", "B"):
            for info in game.infosets[:40]:
                p = render(game, info, template)
                texts += [p.system, p.user]
    for word in ACTION_WORDS.values():
        texts += [word, word.capitalize()]
    return texts


def word_tokenizer():
    """One token per word: every answer is a single token."""
    pre = pre_tokenizers.Whitespace()
    words = set()
    for text in corpus():
        words.update(w for w, _ in pre.pre_tokenize_str(text))
    vocab = {w: i for i, w in enumerate(["[UNK]", "[PAD]", "[EOS]"] + sorted(words))}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    fast.chat_template = CHAT_TEMPLATE
    return fast


def bpe_tokenizer():
    """A small BPE vocabulary: answers span several tokens."""
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(corpus(), trainer=trainers.BpeTrainer(vocab_size=90, special_tokens=["[UNK]", "[PAD]", "[EOS]"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    fast.chat_template = CHAT_TEMPLATE
    return fast


def tiny_model(vocab_size: int, seed: int = 0):
    torch.manual_seed(seed)
    config = Qwen2Config(vocab_size=vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                         tie_word_embeddings=True)
    model = Qwen2ForCausalLM(config)
    # Larger output weights than the default init so the answers' probabilities differ visibly.
    with torch.no_grad():
        model.get_input_embeddings().weight.normal_(0.0, 0.5)
    return model.eval()


def make_policy(tokenizer, game="kuhn", train=False, seed=0, **kw):
    from nashtune.policy.hf import HFPolicy

    return HFPolicy(make_game(game), "tiny-qwen2", model=tiny_model(len(tokenizer), seed), tokenizer=tokenizer,
                    device="cpu", dtype="float32", train=train, seed=seed, echo=quiet, **kw)


def reference_logpi(policy, i: int) -> np.ndarray:
    """One unpadded sequence per answer form, log-probabilities summed by hand."""
    prompt = policy.prompt_ids[i]
    actions = []
    with torch.no_grad():
        for forms in policy.form_ids[i]:
            per_form = []
            for form in forms:
                ids = torch.tensor([prompt + form])
                logits = policy.model(input_ids=ids).logits[0].double()
                lp = torch.log_softmax(logits, dim=-1)
                per_form.append(sum(float(lp[len(prompt) + t - 1, tok]) for t, tok in enumerate(form)))
            actions.append(np.logaddexp.reduce(per_form))
    actions = np.array(actions)
    return actions - np.logaddexp.reduce(actions)


@unittest.skipUnless(HAVE_TORCH, "needs torch, transformers, peft and tokenizers")
class TestTorchLoss(unittest.TestCase):
    def test_torch_loss_matches_numpy(self):
        from nashtune.policy.torch_loss import loss_from_logp_torch

        rng = np.random.default_rng(0)
        sizes = [2, 3, 3, 2, 3]
        builder = BatchBuilder(sizes, 0.2)
        for i, n in enumerate(sizes):
            builder.set_old(i, log_softmax(rng.normal(size=n)))
            builder.set_ref(i, log_softmax(rng.normal(size=n)))
            for _ in range(5):
                builder.add_sample(i, int(rng.integers(n)), float(rng.normal()), float(rng.random()))
            builder.add_kl(i, 0.3)
            builder.add_ce(i, rng.dirichlet(np.ones(n)), 0.2)
        batch = builder.build()
        logs = [log_softmax(lp + rng.normal(scale=0.4, size=len(lp))) for lp in batch.old_logp]
        np_loss, np_grads, _ = loss_and_grad(logs, batch)
        t_logs = [torch.tensor(lp, dtype=torch.float64, requires_grad=True) for lp in logs]
        loss = loss_from_logp_torch(t_logs, batch)
        loss.backward()
        self.assertAlmostEqual(float(loss), np_loss, places=10)
        for t, g in zip(t_logs, np_grads):
            np.testing.assert_allclose(t.grad.numpy(), g, atol=1e-10)


@unittest.skipUnless(HAVE_TORCH, "needs torch, transformers, peft and tokenizers")
class TestScoring(unittest.TestCase):
    def test_single_token_answers_batched_equals_unbatched(self):
        policy = make_policy(word_tokenizer(), batch_rows=3)
        self.assertTrue(policy.single_token_answers)
        ids = list(range(policy.game.num_infosets))
        logs = policy.logpi(ids)
        for i in ids:
            np.testing.assert_allclose(logs[i], reference_logpi(policy, i), atol=2e-5)

    def test_multi_token_answers_batched_equals_unbatched(self):
        policy = make_policy(bpe_tokenizer(), batch_rows=5)
        self.assertFalse(policy.single_token_answers)
        ids = [0, 1, 2, 5, 7, 11]
        logs = policy.logpi(ids)
        for k, i in enumerate(ids):
            np.testing.assert_allclose(logs[k], reference_logpi(policy, i), atol=2e-5)

    def test_extraction_is_a_strategy_with_legal_mass(self):
        policy = make_policy(word_tokenizer())
        ext = policy.extract()
        policy.game.check_strategy(ext.strategy)
        self.assertEqual(ext.legal_mass.shape, (policy.game.num_infosets,))
        self.assertTrue(np.all((ext.legal_mass > 0) & (ext.legal_mass <= 1)))
        # Deterministic.
        np.testing.assert_array_equal(ext.strategy, policy.extract().strategy)

    def test_device_and_dtype_choice(self):
        from nashtune.policy.hf import pick_device, pick_dtype

        self.assertEqual(pick_device("cpu").type, "cpu")
        self.assertEqual(pick_dtype("auto", pick_device("cpu")), torch.float32)
        self.assertEqual(pick_dtype("bfloat16", pick_device("cpu")), torch.bfloat16)


@unittest.skipUnless(HAVE_TORCH, "needs torch, transformers, peft and tokenizers")
class TestTraining(unittest.TestCase):
    def config(self, **kw):
        base = {"game": "kuhn", "reward": "br", "steps": 3, "group_size": 4, "refresh_every": 1, "print_every": 0,
                "seed": 0}
        base.update(kw)
        return TrainConfig(**base)

    def test_grpo_steps_change_the_policy(self):
        policy = make_policy(word_tokenizer(), train=True, lr=1e-2, batch_rows=4)
        self.assertTrue(policy.params)
        before = policy.extract().strategy
        trainer = Trainer(policy, self.config(), echo=quiet)
        trainer.run()
        after = policy.extract().strategy
        self.assertGreater(np.abs(after - before).max(), 1e-4)
        self.assertTrue(all(np.isfinite(r["loss"]) for r in trainer.rows if "loss" in r))

    def test_sft_and_subset_steps(self):
        policy = make_policy(word_tokenizer(), train=True, lr=1e-2, batch_rows=4)
        Trainer(policy, self.config(reward="sft", infosets_per_step=5, refresh_every=2), echo=quiet).run()
        policy = make_policy(word_tokenizer(), train=True, lr=1e-2, batch_rows=4)
        Trainer(policy, self.config(reward="exact", kl_coef=0.1, updates_per_batch=2), echo=quiet).run()

    def test_resume_matches_uninterrupted_run(self):
        tok = word_tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            full = make_policy(tok, train=True, lr=1e-2, batch_rows=4)
            Trainer(full, self.config(steps=4, checkpoint_every=2), out_dir=Path(tmp) / "full", echo=quiet).run()
            part = make_policy(tok, train=True, lr=1e-2, batch_rows=4)
            trainer = Trainer(part, self.config(steps=4, checkpoint_every=2), out_dir=Path(tmp) / "part", echo=quiet)
            trainer.train_step()
            trainer.train_step()
            trainer.save_checkpoint()
            resumed = make_policy(tok, train=True, lr=1e-2, batch_rows=4)
            again = Trainer(resumed, self.config(steps=4, checkpoint_every=2), out_dir=Path(tmp) / "part", echo=quiet)
            self.assertEqual(again.step, 2)
            again.run()
            np.testing.assert_allclose(resumed.extract().strategy, full.extract().strategy, atol=1e-6)

    def test_saved_adapter_reloads(self):
        tok = word_tokenizer()
        policy = make_policy(tok, train=True, lr=1e-2, batch_rows=4)
        Trainer(policy, self.config(steps=2), echo=quiet).run()
        with tempfile.TemporaryDirectory() as tmp:
            policy.save(Path(tmp))
            loaded = make_policy(tok, adapter=str(Path(tmp) / "adapter"))
            np.testing.assert_allclose(loaded.extract().strategy, policy.extract().strategy, atol=1e-6)

    def test_bench_leaves_the_policy_unchanged(self):
        from nashtune.policy.hf import bench

        policy = make_policy(word_tokenizer(), train=True, lr=1e-2, batch_rows=4)
        before = policy.extract().strategy
        out = bench(policy, n_infosets=4, echo=quiet)
        self.assertIn("seconds_per_train_step", out)
        np.testing.assert_allclose(policy.extract().strategy, before, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
