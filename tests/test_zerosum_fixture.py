"""NashTune against ZeroSum, an independent C++ implementation of the same games.

The fixture (tests/fixtures/zerosum/) holds fixed random strategies in ZeroSum's
file format and the numbers ``zerosum eval`` printed for them, plus the
information-set order from ``zerosum export`` and the counts from
``zerosum info``. It was produced by scripts/make_zerosum_fixture.py.

Set ZEROSUM_BIN=/path/to/zerosum to also run the binary live.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from nashtune.games import exploitability, make_game, read_strategy, write_strategy

FIXTURE = Path(__file__).parent / "fixtures" / "zerosum"


def close(a: float, b: float, rel: float = 1e-9, abs_tol: float = 1e-11) -> bool:
    # ZeroSum prints 10 significant digits.
    return abs(a - b) <= max(abs_tol, rel * abs(b))


class TestZeroSumFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((FIXTURE / "zerosum_results.json").read_text())

    def test_games_present(self):
        self.assertEqual(sorted(self.data["games"]), ["kuhn", "leduc", "leduc4", "leduc5"])

    def test_infoset_order_and_actions_match(self):
        for name, entry in self.data["games"].items():
            game = make_game(name)
            ours = [[i.key, i.actions, i.player] for i in game.infosets]
            self.assertEqual(ours, entry["infosets"], name)

    def test_node_counts_match(self):
        for name, entry in self.data["games"].items():
            counts = make_game(name).counts()
            line = entry["info"][1]  # "nodes N (chance C, decision D, terminal T)"
            expected = f"nodes {counts['nodes']} (chance {counts['chance']}, decision {counts['decision']}, terminal {counts['terminal']})"
            self.assertEqual(line, expected)

    def test_values_and_exploitability_match(self):
        checked = 0
        for name, entry in self.data["games"].items():
            game = make_game(name)
            for row in entry["strategies"]:
                strategy = read_strategy(game, FIXTURE / row["file"])
                e = exploitability(game, strategy)
                with self.subTest(game=name, file=row["file"]):
                    self.assertTrue(close(e.value, row["value_to_first_player"]))
                    self.assertTrue(close(e.best_vs_p1, row["best_response_vs_second_player"]))
                    self.assertTrue(close(e.best_vs_p0, row["best_response_vs_first_player"]))
                    self.assertTrue(close(e.milli, row["exploitability_ma_per_hand"]))
                    self.assertTrue(close(e.value_lower, row["value_lower"]))
                    self.assertTrue(close(e.value_upper, row["value_upper"]))
                checked += 1
        self.assertGreaterEqual(checked, 14)

    @unittest.skipUnless(os.environ.get("ZEROSUM_BIN"), "set ZEROSUM_BIN to run the ZeroSum binary live")
    def test_live_binary(self):
        binary = os.environ["ZEROSUM_BIN"]
        rng = np.random.default_rng(99)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("kuhn", "leduc", "leduc4"):
                game = make_game(name)
                strategy = game.normalize(rng.random(game.action_slots) ** 3 + 1e-3)
                path = Path(tmp) / f"{name}.txt"
                write_strategy(game, strategy, path)
                out = subprocess.run([binary, "eval", name, "--strategy", str(path)], check=True,
                                     capture_output=True, text=True).stdout
                fields = dict(line.split()[:2] for line in out.splitlines())
                e = exploitability(game, read_strategy(game, path))
                self.assertTrue(close(e.value, float(fields["value_to_first_player"])))
                self.assertTrue(close(e.milli, float(fields["exploitability_ma_per_hand"])))


if __name__ == "__main__":
    unittest.main()
