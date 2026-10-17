from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from nashtune.games import format_strategy, make_game, parse_strategy, read_strategy, write_strategy


class TestStrategyFiles(unittest.TestCase):
    def test_round_trip_is_exact(self):
        g = make_game("leduc")
        s = g.normalize(np.random.default_rng(0).random(g.action_slots))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "s.txt"
            write_strategy(g, s, path)
            np.testing.assert_array_equal(read_strategy(g, path), s)

    def test_format(self):
        g = make_game("kuhn")
        text = format_strategy(g, g.uniform_strategy())
        lines = text.splitlines()
        self.assertEqual(lines[0], "# zerosum strategy for kuhn")
        self.assertEqual(lines[2], "0:J:-: k 0.5")
        self.assertEqual(len(lines), 2 + g.action_slots)

    def test_errors(self):
        g = make_game("kuhn")
        good = format_strategy(g, g.uniform_strategy())
        with self.assertRaisesRegex(ValueError, "missing"):
            parse_strategy(g, "\n".join(good.splitlines()[:-1]))
        with self.assertRaisesRegex(ValueError, "different game"):
            parse_strategy(g, good + "0:A:-: k 1\n")
        with self.assertRaisesRegex(ValueError, "do not sum"):
            parse_strategy(g, good.replace("0:J:-: k 0.5", "0:J:-: k 0.7"))
        with self.assertRaisesRegex(ValueError, "bad strategy line"):
            parse_strategy(g, good.replace("0:J:-: k 0.5", "0:J:-: x 0.5"))
        with self.assertRaises(ValueError):
            write_strategy(g, np.zeros(g.action_slots), "unused.txt")


if __name__ == "__main__":
    unittest.main()
