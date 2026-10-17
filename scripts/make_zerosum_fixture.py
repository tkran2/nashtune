"""Regenerate tests/fixtures/zerosum/ from a ZeroSum binary.

ZeroSum (github.com/tkran2/zerosum) is a separate C++ implementation of the
same games. This script writes several fixed random strategies in its file
format, asks ``zerosum eval`` for their value and best-response payoffs, and
records ``zerosum info`` counts and the information-set order from
``zerosum export``. The tests then require NashTune to reproduce every number.

    python3 scripts/make_zerosum_fixture.py /path/to/zerosum/build/zerosum

The committed fixture was produced this way; you do not need to rerun it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nashtune.games import make_game, write_strategy  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "zerosum"
GAMES = {"kuhn": 3, "leduc": 3, "leduc4": 3, "leduc5": 1}


def random_strategy(game, kind: str, rng: np.random.Generator) -> np.ndarray:
    s = np.empty(game.action_slots)
    for info in game.infosets:
        n = info.action_count
        if kind == "dirichlet1":
            p = rng.dirichlet(np.ones(n))
        elif kind == "dirichlet03":
            p = rng.dirichlet(np.full(n, 0.3))
        else:  # "pure_mix": a third pure, a third uniform (ties for the best response), a third random
            u = rng.random()
            if u < 1 / 3:
                p = np.zeros(n)
                p[rng.integers(n)] = 1.0
            elif u < 2 / 3:
                p = np.full(n, 1.0 / n)
            else:
                p = rng.dirichlet(np.ones(n))
        s[info.offset : info.offset + n] = p
    return s


def run(binary: str, *args: str) -> str:
    return subprocess.run([binary, *args], check=True, capture_output=True, text=True).stdout


def parse_eval(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if parts[0] == "game_value_between":
            out["value_lower"], out["value_upper"] = float(parts[1]), float(parts[2])
        else:
            out[parts[0]] = float(parts[1])
    return out


def main() -> None:
    binary = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    fixture = {"source": "zerosum eval/info/export (C++), github.com/tkran2/zerosum", "games": {}}
    kinds = ["dirichlet1", "dirichlet03", "pure_mix"]
    for name, count in GAMES.items():
        game = make_game(name)
        info = run(binary, "info", name).split("\n")
        exported = json.loads(run(binary, "export", name))
        entry = {
            "info": info,
            "infosets": [[i["key"], i["actions"], i["player"]] for i in exported["infosets"]],
            "strategies": [],
        }
        rng = np.random.default_rng(20261015 + len(name))
        for k in range(count):
            kind = kinds[k]
            path = OUT / f"{name}_{kind}.txt"
            write_strategy(game, random_strategy(game, kind, rng), path)
            entry["strategies"].append({"file": path.name, **parse_eval(run(binary, "eval", name, "--strategy", str(path)))})
        # A near-equilibrium strategy written by ZeroSum's own CFR+ solver.
        path = OUT / f"{name}_zerosum_cfrplus.txt"
        run(binary, "solve", name, "--algo", "cfr+", "--iterations", "300", "--save", str(path))
        entry["strategies"].append({"file": path.name, **parse_eval(run(binary, "eval", name, "--strategy", str(path)))})
        fixture["games"][name] = entry
        print(name, [round(s["exploitability_ma_per_hand"], 4) for s in entry["strategies"]])
    (OUT / "zerosum_results.json").write_text(json.dumps(fixture, indent=1) + "\n")


if __name__ == "__main__":
    main()
