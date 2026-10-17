"""The memo's language-model findings: collapse detection and determinism counts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nashtune.report import base_model_findings, strategy_determinism, training_diagnostics


def log_rows(measured: dict[int, tuple[float, float]], steps: int, dead_from: int | None) -> list[dict]:
    """Rows like a trainer's log.csv: measurements every 10 steps, one row per update step."""
    rows = []
    for step in range(steps + 1):
        r = {"step": str(step), "exploitability_ma": "", "entropy": "", "batch_infosets": "", "skipped_groups": ""}
        if step in measured:
            r["exploitability_ma"], r["entropy"] = (str(v) for v in measured[step])
        if step < steps:
            r["batch_infosets"] = "20"
            r["skipped_groups"] = "20" if dead_from is not None and step >= dead_from else "3"
        rows.append(r)
    return rows


class Diagnostics(unittest.TestCase):
    def test_collapse_is_detected(self):
        measured = {0: (2289.0, 0.6), 10: (1196.0, 0.27), 20: (1463.0, 0.008), 30: (1466.6, 0.0003),
                    40: (1466.6, 0.0003), 50: (1466.6, 0.0001)}
        d = training_diagnostics(log_rows(measured, 50, dead_from=20))
        self.assertEqual(d["best"], 1196.0)
        self.assertEqual(d["best_step"], 10)
        self.assertEqual(d["final"], 1466.6)
        self.assertEqual(d["first_low_entropy_step"], 20)
        self.assertEqual(d["no_signal_steps"], 30)
        self.assertEqual(d["steps"], 50)
        self.assertEqual(d["flat_from"], 20)  # 1,463 is within 0.5% of 1,466.6

    def test_healthy_run_has_signal(self):
        measured = {0: (2289.0, 0.6), 10: (1500.0, 0.5), 20: (900.0, 0.45), 30: (700.0, 0.4)}
        d = training_diagnostics(log_rows(measured, 30, dead_from=None))
        self.assertEqual(d["no_signal_steps"], 0)
        self.assertIsNone(d["first_low_entropy_step"])
        self.assertEqual(d["flat_from"], 30)
        self.assertEqual(d["tail_range"], (700.0, 900.0))

    def test_strategy_determinism(self):
        text = "# zerosum strategy for leduc\n# infoset action probability\n" \
               "0:J:-: k 0.9999999\n0:J:-: b 0.0000001\n1:Q:-: k 0.5\n1:Q:-: b 0.5\n"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.txt"
            p.write_text(text)
            det, n, ent = strategy_determinism(p)
        self.assertEqual((det, n), (1, 2))
        self.assertAlmostEqual(ent, (0.6931471805599453 + 1.7e-6) / 2, places=5)
        self.assertIsNone(strategy_determinism(Path("/nonexistent/strategy.txt")))


class BaseFindings(unittest.TestCase):
    def test_uniform_comparison_and_prompt_swing(self):
        def row(model, game, template, ma, uni):
            return {"label": None, "model": f"Qwen/Qwen2.5-{model}-Instruct", "game": game, "template": template,
                    "exploitability_ma": ma, "uniform_exploitability_ma": uni}
        rows = [row("0.5B", "leduc", "A", 2389.0, 2373.6), row("7B", "leduc", "A", 1245.0, 2373.6),
                row("0.5B", "leduc", "B", 3673.0, 2373.6), row("7B", "leduc", "B", 1708.0, 2373.6)]
        text = "\n".join(base_model_findings(rows))
        self.assertIn("0.5B was no better than uniform random play", text)
        self.assertIn("a factor of up to 1.5", text)
        self.assertIn("0.5B 2,389 → 3,673", text)


if __name__ == "__main__":
    unittest.main()
