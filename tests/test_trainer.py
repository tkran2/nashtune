from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nashtune.games import best_response, exploitability, make_game
from nashtune.games.analysis import expected_value
from nashtune.rewards import refresh, sample_deals, simulate
from nashtune.train import LOG_COLUMNS, TrainConfig, make_tabular_trainer, quiet


class TestRewards(unittest.TestCase):
    def test_br_rewards_are_conditional_values_against_the_best_response(self):
        g = make_game("kuhn")
        s = g.normalize(np.random.default_rng(0).random(g.action_slots) + 0.05)
        info = refresh(g, s)
        e = exploitability(g, s)
        self.assertAlmostEqual(info.expl.milli, e.milli, places=10)
        # Check one infoset by hand: player 1 holding Q facing a bet. The
        # opponent (player 0) best-responds; Q(call) is the average showdown
        # result against the hands player 0 bets with, weighted by how often.
        _, br0 = best_response(g, s, 0)
        key = g.infoset_by_key("1:Q:-:b")
        bet_j = br0[g.infoset_by_key("0:J:-:").offset + 1]
        bet_k = br0[g.infoset_by_key("0:K:-:").offset + 1]
        if bet_j + bet_k > 0:
            self.assertTrue(info.defined[key.index])
            call = (bet_j * 2 + bet_k * -2) / (bet_j + bet_k)
            self.assertAlmostEqual(info.q[key.offset + 1], call, places=12)
            self.assertAlmostEqual(info.q[key.offset + 0], -1.0, places=12)  # folding loses the ante
        else:
            self.assertFalse(info.defined[key.index])

    def test_selfplay_rollouts_estimate_the_value(self):
        g = make_game("leduc")
        rng = np.random.default_rng(1)
        s = g.normalize(rng.random(g.action_slots) + 0.1)

        def probs_fn(ids):
            return [s[g.infosets[i].offset : g.infosets[i].offset + g.infosets[i].action_count] for i in ids]

        deals = sample_deals(g, 4000, rng)
        roll = simulate(g, deals, 4, probs_fn, rng)
        self.assertEqual(len(roll.payoff), 16000)
        est = roll.payoff.mean()
        se = roll.payoff.std() / np.sqrt(len(roll.payoff))
        self.assertLess(abs(est - expected_value(g, s)), 4 * se)
        # Every decision in a rollout belongs to the right player and infoset of that deal.
        for r, player, i, a in roll.decisions[:200]:
            info = g.infosets[i]
            self.assertEqual(info.player, player)
            self.assertLess(a, info.action_count)

    def test_rollouts_of_one_deal_share_the_cards(self):
        g = make_game("leduc")
        rng = np.random.default_rng(2)
        u = g.uniform_strategy()

        def probs_fn(ids):
            return [u[g.infosets[i].offset : g.infosets[i].offset + g.infosets[i].action_count] for i in ids]

        roll = simulate(g, sample_deals(g, 5, rng), 8, probs_fn, rng)
        cards: dict = {}
        for r, player, i, a in roll.decisions:
            info = g.infosets[i]
            deal = int(roll.deal[r])
            entry = cards.setdefault(deal, {})
            entry.setdefault(player, set()).add(info.private_rank)
            if info.public_rank >= 0:
                entry.setdefault("public", set()).add(info.public_rank)
        for entry in cards.values():
            for values in entry.values():
                self.assertEqual(len(values), 1)


class TestTrainer(unittest.TestCase):
    def run_trainer(self, out=None, **kw):
        cfg = TrainConfig(**{"game": "kuhn", "steps": 40, "group_size": 4, "seed": 3, "print_every": 0, **kw})
        tr = make_tabular_trainer(cfg, lr=0.05, out_dir=out, echo=quiet)
        tr.run()
        return tr

    def test_br_reduces_exploitability(self):
        for reward in ("br", "exact"):
            tr = self.run_trainer(reward=reward, steps=150)
            first = tr.rows[0]["exploitability_ma"]
            last = tr.rows[-1]["exploitability_ma"]
            self.assertLess(last, 0.2 * first, reward)

    def test_selfplay_and_sft_run(self):
        tr = self.run_trainer(reward="selfplay", deals_per_step=8)
        self.assertEqual(len(tr.rows), 41)
        self.assertGreater(tr.rows[0]["samples"], 0)
        tr = self.run_trainer(reward="sft", steps=300)
        self.assertLess(tr.rows[-1]["exploitability_ma"], 5.0)

    def test_subset_refresh_kl_and_weighting(self):
        tr = self.run_trainer(reward="br", infosets_per_step=4, refresh_every=5, kl_coef=0.1,
                              weighting="reach", updates_per_batch=2)
        measured = [r for r in tr.rows if "exploitability_ma" in r]
        self.assertEqual(len(measured), 40 // 5 + 1)
        self.assertTrue(all(r["batch_infosets"] <= 4 for r in tr.rows[:-1]))
        self.assertGreater(tr.rows[-1]["kl_ref"], 0)

    def test_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = self.run_trainer(out=Path(tmp) / "full", steps=30, checkpoint_every=10, refresh_every=4,
                                    infosets_per_step=6, kl_coef=0.05)
            # Interrupted run: stop after 20 steps (a checkpoint), then resume to 30.
            part_dir = Path(tmp) / "part"
            cfg = TrainConfig(game="kuhn", steps=20, group_size=4, seed=3, print_every=0, checkpoint_every=10,
                              refresh_every=4, infosets_per_step=6, kl_coef=0.05)
            tr = make_tabular_trainer(cfg, lr=0.05, out_dir=part_dir, echo=quiet)
            while tr.step < 20:
                tr.train_step()
                if tr.step % 10 == 0:
                    tr.save_checkpoint()
            cfg.steps = 30
            resumed = make_tabular_trainer(cfg, lr=0.05, out_dir=part_dir, echo=quiet)
            self.assertEqual(resumed.step, 20)
            resumed.run()
            np.testing.assert_array_equal(resumed.policy.logits, full.policy.logits)
            a = [r.get("exploitability_ma") for r in resumed.rows]
            b = [r.get("exploitability_ma") for r in full.rows]
            self.assertEqual(a, b)

    def test_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            self.run_trainer(out=out, steps=12, checkpoint_every=5)
            with open(out / "log.csv") as fh:
                rows = list(csv.reader(fh))
            self.assertEqual(rows[0], LOG_COLUMNS)
            self.assertEqual(len(rows), 1 + 13)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["steps"], 12)
            self.assertTrue((out / "final_strategy.txt").exists())
            kept = sorted(p.name for p in (out / "checkpoints").glob("step_*"))
            self.assertEqual(kept, ["step_000010", "step_000012"])
            # Running again finds the finished run and does nothing.
            again = self.run_trainer(out=out, steps=12, checkpoint_every=5)
            self.assertEqual(again.step, 12)

    def test_bad_config(self):
        with self.assertRaises(ValueError):
            TrainConfig(reward="ppo").validate()
        with self.assertRaises(ValueError):
            TrainConfig(group_size=1).validate()


if __name__ == "__main__":
    unittest.main()
