"""Experiment A: how the choice of reward shapes GRPO training dynamics.

Runs the tabular policy (one logit per decision) on Kuhn and Leduc with three
reward designs, and CFR+ as the reference:

* ``selfplay``: each decision rewarded with the final payoff of a sampled hand;
* ``br``: each sampled action rewarded with its exact value against the
  opponent's best response (exploitability descent);
* ``exact``: the same values, expectation over actions taken exactly.

Everything is seeded; the plan, settings and timings go into manifest.json.
Results: curves.csv (exploitability every ``record_every`` steps) and
final.csv (one row per run).
"""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__
from ..games import exploitability, make_game
from ..games.cfr import CFRPlus
from ..train import TrainConfig, make_tabular_trainer, quiet

THRESHOLDS = {"kuhn": (50.0, 10.0, 1.0), "leduc": (500.0, 100.0, 30.0)}


@dataclass(frozen=True)
class RunSpec:
    experiment: str
    game: str
    method: str  # br | exact | selfplay | cfr+
    seed: int
    steps: int
    lr: float = 0.05
    group_size: int = 8
    kl_coef: float = 0.0
    adv_norm: str = "std"
    weighting: str = "uniform"
    deals_per_step: int = 32
    infosets_per_step: int = 0
    refresh_every: int = 1
    track_average: bool = False
    init_scale: float = 1.0

    def key(self) -> str:
        return (f"{self.experiment}|{self.game}|{self.method}|s{self.seed}|lr{self.lr}|G{self.group_size}|"
                f"kl{self.kl_coef}|{self.adv_norm}|{self.weighting}|D{self.deals_per_step}|"
                f"K{self.infosets_per_step}|R{self.refresh_every}")


SETTING_FIELDS = ["lr", "group_size", "kl_coef", "adv_norm", "weighting", "deals_per_step", "infosets_per_step",
                  "refresh_every"]


def plan(quick: bool = False) -> list[RunSpec]:
    """The full study, or a seconds-long version of it for the tests."""
    if quick:
        steps = {"kuhn": 30, "leduc": 20}
        seeds = [0, 1]
        sweep_seeds = [0]
        sweep_steps = 20
    else:
        steps = {"kuhn": 2000, "leduc": 3000}
        seeds = [0, 1, 2, 3, 4]
        sweep_seeds = [0, 1, 2]
        sweep_steps = 2000
    runs: list[RunSpec] = []
    for game in ("kuhn", "leduc"):
        runs.append(RunSpec("main", game, "cfr+", 0, steps[game]))
        for method in ("br", "exact", "selfplay"):
            for seed in seeds:
                runs.append(RunSpec("main", game, method, seed, steps[game], track_average=True))
    leduc = "leduc"
    for seed in sweep_seeds:
        for g in (2, 4, 16):
            runs.append(RunSpec("group_size", leduc, "br", seed, sweep_steps, group_size=g))
            runs.append(RunSpec("group_size", leduc, "selfplay", seed, sweep_steps, group_size=g))
        for kl in (0.01, 0.1):
            runs.append(RunSpec("kl", leduc, "br", seed, sweep_steps, kl_coef=kl))
        for method in ("br", "exact"):
            runs.append(RunSpec("adv_norm", leduc, method, seed, sweep_steps, adv_norm="none"))
        for w in ("reach", "cfreach"):
            runs.append(RunSpec("weighting", leduc, "br", seed, sweep_steps, weighting=w))
        # Reach weighting, exact expectation and no normalization: plain gradient
        # descent on exploitability in the softmax parameters.
        runs.append(RunSpec("weighting", leduc, "exact", seed, sweep_steps, weighting="reach", adv_norm="none"))
        for lr in (0.01, 0.2):
            for method in ("br", "exact", "selfplay"):
                runs.append(RunSpec("lr", leduc, method, seed, sweep_steps, lr=lr))
        runs.append(RunSpec("selfplay_deals", leduc, "selfplay", seed, sweep_steps, deals_per_step=128))
        # The levers a language model needs: a random 32 of the 288 infosets per
        # step, and best responses recomputed only every 10 steps.
        for refresh_every in (1, 10):
            runs.append(RunSpec("subset", leduc, "br", seed, sweep_steps, infosets_per_step=32,
                                refresh_every=refresh_every))
    return runs


def _thresholds(game: str, steps: np.ndarray, values: np.ndarray) -> dict:
    out = {}
    for t in THRESHOLDS.get(game, ()):
        hit = np.flatnonzero(values <= t)
        out[f"steps_to_{t:g}ma"] = int(steps[hit[0]]) if hit.size else ""
    return out


def run_one(spec: RunSpec, record_every: int = 10) -> tuple[list[dict], dict]:
    start = time.time()
    game = make_game(spec.game)
    curve: list[dict] = []
    if spec.method == "cfr+":
        solver = CFRPlus(game)
        points = [(0, exploitability(game, solver.average_strategy()).milli)]
        while solver.iteration < spec.steps:
            solver.run(record_every)
            points.append((solver.iteration, exploitability(game, solver.average_strategy()).milli))
        for step, value in points:
            curve.append({"step": step, "exploitability_ma": value})
    else:
        cfg = TrainConfig(game=spec.game, reward=spec.method, steps=spec.steps, group_size=spec.group_size,
                          infosets_per_step=spec.infosets_per_step, deals_per_step=spec.deals_per_step,
                          refresh_every=spec.refresh_every, kl_coef=spec.kl_coef, weighting=spec.weighting,
                          adv_norm=spec.adv_norm, seed=spec.seed, print_every=0, track_average=spec.track_average)
        trainer = make_tabular_trainer(cfg, lr=spec.lr, init="random", init_scale=spec.init_scale, echo=quiet)
        trainer.run()
        for row in trainer.rows:
            if "exploitability_ma" in row and (row["step"] % record_every == 0 or row["step"] == spec.steps):
                curve.append({k: row.get(k, "") for k in ("step", "exploitability_ma", "avg_exploitability_ma",
                                                           "mean_reward", "kl_ref", "entropy")})
    steps = np.array([r["step"] for r in curve])
    values = np.array([r["exploitability_ma"] for r in curve], dtype=float)
    tail = values[steps >= 0.9 * spec.steps]
    final = {
        **asdict(spec), "final_ma": float(values[-1]), "tail_mean_ma": float(tail.mean()),
        "tail_max_ma": float(tail.max()), "min_ma": float(values.min()), "initial_ma": float(values[0]),
        **_thresholds(spec.game, steps, values), "seconds": round(time.time() - start, 2),
    }
    if spec.track_average and curve and curve[-1].get("avg_exploitability_ma", "") != "":
        final["final_avg_ma"] = float(curve[-1]["avg_exploitability_ma"])
    return curve, final


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns: list[str] = []
    for row in rows:
        for k in row:
            if k not in columns:
                columns.append(k)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    os.replace(tmp, path)


def _run_spec(args):
    spec, record_every = args
    return spec, run_one(spec, record_every)


def run_all(out_dir: str | os.PathLike, quick: bool = False, jobs: int = 1, record_every: int = 10,
            echo=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = plan(quick)
    # Resumable: results of finished runs are kept in a JSON-lines file.
    done_path = out / "runs.jsonl"
    done: dict[str, tuple[list[dict], dict]] = {}
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            try:
                rec = json.loads(line)
            except ValueError:
                continue  # a line cut off by an interruption: that run is redone
            done[rec["key"]] = (rec["curve"], rec["final"])
    todo = [s for s in specs if s.key() not in done]
    echo(f"[dynamics] {len(specs)} runs, {len(done)} already done, {len(todo)} to go")
    start = time.time()

    def record(spec, result):
        curve, final = result
        done[spec.key()] = (curve, final)
        with open(done_path, "a") as fh:
            fh.write(json.dumps({"key": spec.key(), "curve": curve, "final": final}) + "\n")
        echo(f"[dynamics] {len(done):3d}/{len(specs)} {spec.experiment:14s} {spec.game:6s} {spec.method:8s} "
             f"seed {spec.seed}  final {final['final_ma']:9.2f} mA  ({final['seconds']:.0f} s)")

    if jobs > 1 and todo:
        from multiprocessing import get_context

        with get_context("spawn").Pool(jobs) as pool:
            for spec, result in pool.imap_unordered(_run_spec, [(s, record_every) for s in todo]):
                record(spec, result)
    else:
        for spec in todo:
            record(spec, run_one(spec, record_every))

    curves, finals = [], []
    for spec in specs:
        curve, final = done[spec.key()]
        finals.append(final)
        for row in curve:
            curves.append({"experiment": spec.experiment, "game": spec.game, "method": spec.method,
                           "seed": spec.seed, **{k: getattr(spec, k) for k in SETTING_FIELDS}, **row})
    _write_csv(out / "curves.csv", curves)
    _write_csv(out / "final.csv", finals)
    manifest = {
        "experiment": "A: reward-design dynamics (tabular softmax policy)",
        "nashtune_version": __version__, "quick": quick, "record_every": record_every,
        "runs": len(specs), "python": sys.version.split()[0], "numpy": np.__version__,
        "machine": platform.platform(), "processor": platform.machine(),
        "wall_seconds_this_invocation": round(time.time() - start, 1),
        "total_run_seconds": round(sum(f["seconds"] for f in finals), 1),
        "defaults": {
            "policy": "softmax table, logits initialized N(0, 1) from the run's seed",
            "optimizer": "Adam, lr 0.05 unless the run says otherwise, betas (0.9, 0.999)",
            "group_size": 8, "clip_eps": 0.2, "updates_per_batch": 1, "kl_coef": 0.0,
            "advantages": "(r - mean) / (std + 1e-6) within each group, population std",
            "infosets_per_step": "all (br, exact)", "deals_per_step": "32 (selfplay)",
            "refresh_every": 1, "weighting": "uniform",
            "cfr+": "alternating updates, regret floor at zero, linear averaging; one iteration per step",
        },
        "thresholds_ma": THRESHOLDS,
        "plan": [asdict(s) for s in specs],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest
