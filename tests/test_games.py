from __future__ import annotations

import unittest

import numpy as np

from nashtune.games import GAME_VALUES, CFRPlus, exploitability, game_value, make_game, solve_to
from nashtune.games.analysis import expected_value, expected_value_terminal_sum
from nashtune.games.game import CHANCE, DECISION, TERMINAL


def kuhn_equilibrium(alpha: float) -> np.ndarray:
    """Kuhn's hand-derived family: first player bluffs J with alpha, bets K with
    3 alpha, never bets Q, calls with Q alpha + 1/3. The second player bets K,
    bluffs J 1/3, calls with Q 1/3."""
    game = make_game("kuhn")
    s = np.zeros(game.action_slots)

    def put(key, probs):
        info = game.infoset_by_key(key)
        for action, p in probs.items():
            s[info.offset + info.actions.index(action)] = p

    put("0:J:-:", {"k": 1 - alpha, "b": alpha})
    put("0:Q:-:", {"k": 1.0, "b": 0.0})
    put("0:K:-:", {"k": 1 - 3 * alpha, "b": 3 * alpha})
    put("0:J:-:kb", {"f": 1.0, "c": 0.0})
    put("0:Q:-:kb", {"f": 2 / 3 - alpha, "c": alpha + 1 / 3})
    put("0:K:-:kb", {"f": 0.0, "c": 1.0})
    put("1:J:-:k", {"k": 2 / 3, "b": 1 / 3})
    put("1:Q:-:k", {"k": 1.0, "b": 0.0})
    put("1:K:-:k", {"k": 0.0, "b": 1.0})
    put("1:J:-:b", {"f": 1.0, "c": 0.0})
    put("1:Q:-:b", {"f": 2 / 3, "c": 1 / 3})
    put("1:K:-:b", {"f": 0.0, "c": 1.0})
    return s


class TestGameTrees(unittest.TestCase):
    def test_infoset_counts_match_zerosum(self):
        expected = {"kuhn": (12, 24, 55), "leduc": (288, 672, 1936), "leduc4": (504, 1176, 4741),
                    "leduc5": (780, 1820, 9376)}
        for name, (infosets, actions, nodes) in expected.items():
            counts = make_game(name).counts()
            self.assertEqual(counts["infosets"], infosets, name)
            self.assertEqual(counts["actions"], actions, name)
            self.assertEqual(counts["nodes"], nodes, name)
            self.assertEqual(counts["infosets_player0"], counts["infosets_player1"])

    def test_chance_probabilities_sum_to_one(self):
        for name in ("kuhn", "leduc", "leduc4"):
            g = make_game(name)
            for node in np.flatnonzero(g.node_type == CHANCE):
                start, n = g.first_child[node], g.child_count[node]
                self.assertAlmostEqual(float(g.chance_prob[start : start + n].sum()), 1.0, places=12)

    def test_tree_structure(self):
        g = make_game("leduc")
        self.assertEqual(g.node_type[0], CHANCE)
        for node in range(g.num_nodes):
            if g.node_type[node] == TERMINAL:
                self.assertEqual(g.child_count[node], 0)
            if g.node_type[node] == DECISION:
                info = g.infosets[g.node_infoset[node]]
                self.assertEqual(info.player, g.node_player[node])
                self.assertEqual(info.action_count, g.child_count[node])
        # A raise is only allowed after a bet, and at most two bets per round.
        for info in g.infosets:
            self.assertIn(info.actions, ("kb", "fcr", "fc"))
            for rnd in info.history.split("/"):
                self.assertLessEqual(rnd.count("b") + rnd.count("r"), 2)

    def test_key_format(self):
        g = make_game("leduc")
        info = g.infoset_by_key("0:K:J:kbc/kb")
        self.assertEqual((info.player, info.actions, info.round), (0, "fcr", 1))
        self.assertEqual(g.rank_names, ["J", "Q", "K"])
        self.assertEqual(make_game("leduc4").rank_names, ["J", "Q", "K", "A"])
        self.assertEqual(make_game("leduc5").rank_names, ["T", "J", "Q", "K", "A"])

    def test_unknown_game(self):
        with self.assertRaises(ValueError):
            make_game("holdem")

    def test_kuhn_equilibrium_family_is_unexploitable(self):
        g = make_game("kuhn")
        for alpha in (0.0, 0.1, 0.2, 1 / 3):
            s = kuhn_equilibrium(alpha)
            g.check_strategy(s)
            e = exploitability(g, s)
            self.assertAlmostEqual(e.value, -1 / 18, places=12)
            self.assertLess(abs(e.exploitability), 1e-12)

    def test_value_two_ways(self):
        rng = np.random.default_rng(3)
        for name in ("kuhn", "leduc", "leduc4"):
            g = make_game(name)
            s = g.normalize(rng.random(g.action_slots) + 0.01)
            self.assertAlmostEqual(expected_value(g, s), expected_value_terminal_sum(g, s), places=12)

    def test_cfr_plus_converges_and_brackets_the_value(self):
        g = make_game("leduc")
        solver = CFRPlus(g)
        solver.run(600)
        e = exploitability(g, solver.average_strategy())
        self.assertLess(e.milli, 1.5)
        self.assertLessEqual(e.value_lower, GAME_VALUES["leduc"] + 1e-12)
        self.assertGreaterEqual(e.value_upper, GAME_VALUES["leduc"] - 1e-12)

    def test_cfr_plus_kuhn(self):
        g = make_game("kuhn")
        s, iterations, gap = solve_to(g, 0.05)
        self.assertLessEqual(gap, 0.05)
        self.assertAlmostEqual(exploitability(g, s).value, -1 / 18, places=4)

    def test_game_value_table(self):
        self.assertAlmostEqual(game_value("kuhn"), -1 / 18, places=15)
        # leduc4's tabulated value (ZeroSum's LP) lies inside a CFR+ bracket.
        g = make_game("leduc4")
        s, _, _ = solve_to(g, 2.0)
        e = exploitability(g, s)
        self.assertLessEqual(e.value_lower, game_value("leduc4"))
        self.assertGreaterEqual(e.value_upper, game_value("leduc4"))


if __name__ == "__main__":
    unittest.main()
