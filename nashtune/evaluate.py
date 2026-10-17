"""Measure a policy exactly and write the result files.

For a policy (a language model or anything else implementing ``Policy``):
the complete strategy in ZeroSum's format, a JSON summary (exploitability per
seat and averaged, value, legal mass, timing, settings) and a per-infoset CSV.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .games import exploitability, game_value, solve_to, write_strategy
from .games.analysis import Exploitability
from .games.game import Game
from .policy.base import Extraction, Policy


def summarize(game: Game, ext: Extraction) -> dict:
    e: Exploitability = exploitability(game, ext.strategy)
    seat0, seat1 = e.seat_milli(game_value(game.name))
    out = {
        "game": game.name, "infosets": game.num_infosets, "exploitability_ma": e.milli,
        "seat0_ma": seat0, "seat1_ma": seat1, "value": e.value, "best_vs_p1": e.best_vs_p1,
        "best_vs_p0": e.best_vs_p0, "game_value": game_value(game.name),
        "uniform_exploitability_ma": exploitability(game, game.uniform_strategy()).milli,
        "extraction_seconds": ext.seconds,
    }
    if ext.legal_mass is not None and np.isfinite(ext.legal_mass).any():
        lm = ext.legal_mass[np.isfinite(ext.legal_mass)]
        out.update({"legal_mass_mean": float(lm.mean()), "legal_mass_min": float(lm.min()),
                    "legal_mass_median": float(np.median(lm))})
    return out


def slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "_" for c in text).strip("_")


def write_evaluation(game: Game, ext: Extraction, out_prefix: str | Path, meta: dict) -> dict:
    """Write ``<prefix>.json``, ``<prefix>_strategy.txt`` and ``<prefix>_infosets.csv``."""
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary = {**meta, **summarize(game, ext), "nashtune_version": __version__,
               "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    extra = dict(ext.extra)
    per_infoset = extra.pop("per_infoset", None)
    summary.update({k: v for k, v in extra.items()})
    write_strategy(game, ext.strategy, f"{prefix}_strategy.txt")
    reference, _, _ = solve_to(game, 0.5)
    with open(f"{prefix}_infosets.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["infoset", "actions", "probabilities", "equilibrium_probabilities", "legal_mass", "notes"])
        for k, info in enumerate(game.infosets):
            s = slice(info.offset, info.offset + info.action_count)
            lm = "" if ext.legal_mass is None else f"{ext.legal_mass[k]:.6g}"
            notes = "" if per_infoset is None else json.dumps(per_infoset[k], sort_keys=True)
            w.writerow([info.key, info.actions, " ".join(f"{p:.6f}" for p in ext.strategy[s]),
                        " ".join(f"{p:.4f}" for p in reference[s]), lm, notes])
    tmp = Path(f"{prefix}.json.tmp")
    tmp.write_text(json.dumps(summary, indent=1, default=float) + "\n")
    tmp.replace(f"{prefix}.json")
    return summary


def evaluate(policy: Policy, out_prefix: str | Path, meta: dict, echo=print) -> dict:
    start = time.time()
    ext = policy.extract()
    summary = write_evaluation(policy.game, ext, out_prefix, {**meta, **policy.describe()})
    echo(f"[eval] {meta.get('label', '')} {policy.game.name}: exploitability {summary['exploitability_ma']:.1f} mA "
         f"(seat 0 {summary['seat0_ma']:.1f}, seat 1 {summary['seat1_ma']:.1f}; uniform play "
         f"{summary['uniform_exploitability_ma']:.1f})"
         + (f", legal mass {summary['legal_mass_mean']:.3f}" if "legal_mass_mean" in summary else "")
         + f"  [{time.time() - start:.0f} s]")
    return summary
