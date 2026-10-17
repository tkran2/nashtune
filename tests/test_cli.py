"""The command line, end to end (NumPy only)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "zerosum"


def run(*args: str, cwd: Path = ROOT) -> str:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, "-m", "nashtune", *args], cwd=cwd, env=env, capture_output=True,
                         text=True, timeout=600)
    if out.returncode != 0:
        raise AssertionError(f"nashtune {' '.join(args)} failed:\n{out.stdout}\n{out.stderr}")
    return out.stdout


class TestCommandLine(unittest.TestCase):
    def test_info_and_eval_match_zerosum(self):
        self.assertIn("information_sets 288", run("info", "leduc"))
        out = run("eval", "leduc", "--strategy", str(FIXTURE / "leduc_dirichlet1.txt"))
        fields = dict(line.split()[:2] for line in out.splitlines())
        expected = json.loads((FIXTURE / "zerosum_results.json").read_text())["games"]["leduc"]["strategies"][0]
        self.assertEqual(fields["exploitability_ma_per_hand"], f"{expected['exploitability_ma_per_hand']:.10g}")

    def test_prompts(self):
        out = run("prompts", "leduc", "--infoset", "0:K:J:kbc/kb", "--template", "B")
        self.assertIn("Choose one of: raise, call, fold.", out)

    def test_solve_train_dynamics_report_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            run("solve", "kuhn", "--iterations", "200", "--save", str(tmp / "k.txt"))
            self.assertIn("exploitability_ma_per_hand", run("eval", "kuhn", "--strategy", str(tmp / "k.txt")))
            run("train", "--game", "kuhn", "--reward", "selfplay", "--steps", "10", "--out", str(tmp / "run"))
            self.assertTrue((tmp / "run" / "summary.json").exists())
            run("dynamics", "--quick", "--out", str(tmp / "results" / "dynamics"))
            run("report", "--results", str(tmp / "results"), "--out", str(tmp / "reports"))
            memo = (tmp / "reports" / "NashTune_Research_Memo.md").read_text()
            self.assertIn("## 2. Reward design", memo)
            self.assertIn("Pending Mac run", memo)
            self.assertTrue((tmp / "reports" / "figures" / "dynamics_main.png").exists())
            run("bundle", "--out", str(tmp / "b.zip"), cwd=tmp)
            self.assertTrue((tmp / "b.zip").exists())

    def test_openai_plan_dry_run_needs_no_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            prices = Path(tmp) / "prices.json"
            prices.write_text(json.dumps({"m1": {"input": 1.0, "output": 2.0}}))
            out = run("openai-plan", "--chat-model", "m1", "--prices", str(prices), "--dry-run", "--out", tmp)
            self.assertIn("Remaining runs", out)


if __name__ == "__main__":
    unittest.main()
