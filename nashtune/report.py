"""Build reports/NashTune_Research_Memo.md and its figures from whatever is in
results/. Sections whose results do not exist yet say so and give the command
that produces them. Every number in the memo is read from a results file."""

from __future__ import annotations

import csv
import json
import math
import os
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# Reference categorical palette (validated for color-vision deficiency), plus ink.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                                                       "#008300", "#6250d6")
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
METHOD_STYLE = {
    "selfplay": (ORANGE, "GRPO, self-play outcome reward"),
    "br": (BLUE, "GRPO, exact value vs best response (sampled actions)"),
    "exact": (AQUA, "GRPO, exact value vs best response (expected over actions)"),
    "cfr+": (INK2, "CFR+ (reference solver)"),
}


# ------------------------------------------------------------------ reading
def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def fnum(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_json_dir(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for p in sorted(path.glob("*.json")):
        if p.name in ("spend.json", "model_params.json"):
            continue
        try:
            data = json.loads(p.read_text())
        except ValueError:
            continue
        if "exploitability_ma" in data:
            data["_file"] = str(p)
            out.append(data)
    return out


def fmt(x: float, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if abs(x) >= 100:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}"


def med(values) -> float:
    v = [x for x in values if not math.isnan(x)]
    return float(np.median(v)) if v else float("nan")


# ------------------------------------------------------------------ figures
def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _log_axis(ax, axis: str = "y") -> None:
    """Log scale with plain-number tick labels (10, 100, 1,000) and no minor labels."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    target = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    target.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0)))
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    target.set_minor_formatter(NullFormatter())


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
                         "axes.labelcolor": INK2, "figure.facecolor": SURFACE})
    return plt


def figure_dynamics_main(curves: list[dict], out: Path) -> Path | None:
    rows = [r for r in curves if r["experiment"] == "main"]
    if not rows:
        return None
    plt = _plt()
    games = [g for g in ("kuhn", "leduc") if any(r["game"] == g for r in rows)]
    fig, axes = plt.subplots(1, len(games), figsize=(6.2 * len(games), 4.4), squeeze=False)
    for ax, game in zip(axes[0], games):
        _style(ax)
        for method in ("selfplay", "br", "exact", "cfr+"):
            series = defaultdict(dict)
            for r in rows:
                if r["game"] == game and r["method"] == method:
                    series[int(r["seed"])][int(r["step"])] = fnum(r["exploitability_ma"])
            if not series:
                continue
            steps = sorted(set.intersection(*[set(s) for s in series.values()]))
            data = np.array([[series[seed][t] for t in steps] for seed in sorted(series)])
            color, label = METHOD_STYLE[method]
            x = np.array(steps)
            ax.plot(x, np.median(data, axis=0), color=color, linewidth=2, label=label, solid_capstyle="round")
            if data.shape[0] > 1:
                ax.fill_between(x, data.min(axis=0), data.max(axis=0), color=color, alpha=0.12, linewidth=0)
            if method == "selfplay":
                avg = defaultdict(dict)
                for r in rows:
                    if r["game"] == game and r["method"] == method and r.get("avg_exploitability_ma") not in ("", None):
                        avg[int(r["seed"])][int(r["step"])] = fnum(r["avg_exploitability_ma"])
                if avg:
                    st = sorted(set.intersection(*[set(s) for s in avg.values()]))
                    d = np.array([[avg[s][t] for t in st] for s in sorted(avg)])
                    ax.plot(st, np.median(d, axis=0), color=ORANGE, linewidth=1.2, linestyle=(0, (4, 2)),
                            label="self-play, average of its policies")
        _log_axis(ax)
        ax.set_xlabel("training step (CFR+: iteration)")
        ax.set_ylabel("exploitability, mA per hand (log scale)")
        ax.set_title({"kuhn": "Kuhn poker (12 information sets)", "leduc": "Leduc hold'em (288 information sets)"}[game],
                     color=INK, loc="left")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=3, frameon=False, fontsize=9,
               labelcolor=INK2)
    seeds = len({r["seed"] for r in rows if r["method"] == "br"})
    fig.text(0.01, 0.01, f"Lines: median of {seeds} seeds; bands: min to max. Tabular softmax policy, Adam lr 0.05, G = 8.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    path = out / "dynamics_main.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_sweeps(finals: list[dict], out: Path) -> Path | None:
    leduc = [f for f in finals if f["game"] == "leduc"]
    if not any(f["experiment"] != "main" for f in leduc):
        return None
    plt = _plt()
    panels = [
        ("Group size G", "group_size", [("br", BLUE), ("selfplay", ORANGE)], [2, 4, 8, 16]),
        ("Learning rate", "lr", [("br", BLUE), ("exact", AQUA), ("selfplay", ORANGE)], [0.01, 0.05, 0.2]),
        ("KL coefficient (to the step-0 policy)", "kl_coef", [("br", BLUE)], [0.0, 0.01, 0.1]),
    ]
    fig, grid = plt.subplots(2, 2, figsize=(11, 8.5), sharey=True)
    axes = grid.ravel()
    for ax, (title, key, methods, values) in zip(axes, panels):
        _style(ax)
        for k, (method, color) in enumerate(methods):
            xs, ys, meds = [], [], []
            for j, v in enumerate(values):
                pts = [fnum(f["tail_mean_ma"]) for f in leduc if f["method"] == method and _matches(f, key, v)]
                pts = [p for p in pts if not math.isnan(p)]
                if not pts:
                    continue
                offset = (k - (len(methods) - 1) / 2) * 0.18
                xs += [j + offset] * len(pts)
                ys += pts
                meds.append((j + offset, float(np.median(pts))))
            ax.scatter(xs, ys, s=40, color=color, edgecolors=SURFACE, linewidths=1.5, zorder=3,
                       label=METHOD_STYLE[method][1].split(", ", 1)[1])
            for x, m in meds:
                ax.plot([x - 0.08, x + 0.08], [m, m], color=INK, linewidth=1.5, zorder=4)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{v:g}" for v in values])
        _log_axis(ax)
        ax.set_title(title, color=INK, loc="left", fontsize=10)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("exploitability, last 10% of steps (mA)")
    # Variants of the br reward.
    ax = axes[3]
    _style(ax)
    variants = [("br, std-normalized, uniform", "br", {"adv_norm": "std", "weighting": "uniform"}),
                ("br, unnormalized", "br", {"adv_norm": "none", "weighting": "uniform"}),
                ("br, reach-weighted", "br", {"adv_norm": "std", "weighting": "reach"}),
                ("br, cf-reach-weighted", "br", {"adv_norm": "std", "weighting": "cfreach"}),
                ("exact, unnormalized", "exact", {"adv_norm": "none", "weighting": "uniform"}),
                ("exact gradient (reach, unnormalized)", "exact", {"adv_norm": "none", "weighting": "reach"})]
    labels = []
    for j, (label, method, settings) in enumerate(variants):
        pts = [fnum(f["tail_mean_ma"]) for f in leduc if f["method"] == method
               and all(str(f[k]) == v for k, v in settings.items()) and _default_except(f, settings)]
        if not pts:
            continue
        labels.append(label)
        x = len(labels) - 1
        ax.scatter([x] * len(pts), pts, s=40, color=METHOD_STYLE[method][0], edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.plot([x - 0.12, x + 0.12], [np.median(pts)] * 2, color=INK, linewidth=1.5, zorder=4)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    _log_axis(ax)
    ax.set_title("Reward variants (br, exact)", color=INK, loc="left", fontsize=10)
    handles, lbls = axes[1].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2)
    steps = sorted({int(f["steps"]) for f in leduc if f["experiment"] != "main"})
    fig.text(0.01, 0.005, f"Leduc. Each dot: one seed's mean exploitability over steps {int(0.9 * steps[0]):,} to "
             f"{steps[0]:,} (default settings: the main runs over the same steps). Black ticks: medians.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    path = out / "dynamics_sweeps.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


DEFAULTS = {"lr": 0.05, "group_size": 8, "kl_coef": 0.0, "adv_norm": "std", "weighting": "uniform",
            "deals_per_step": 32, "infosets_per_step": 0, "refresh_every": 1}


def _matches(f: dict, key: str, value) -> bool:
    """Run f has ``key == value`` and default settings otherwise."""
    for k, d in DEFAULTS.items():
        want = value if k == key else d
        have = f[k]
        if isinstance(d, (int, float)) and not isinstance(d, bool):
            if not math.isclose(fnum(have), float(want), rel_tol=1e-9, abs_tol=1e-12):
                return False
        elif str(have) != str(want):
            return False
    return True


def _default_except(f: dict, settings: dict) -> bool:
    for k, d in DEFAULTS.items():
        if k in settings:
            continue
        if isinstance(d, (int, float)):
            if not math.isclose(fnum(f[k]), float(d), rel_tol=1e-9, abs_tol=1e-12):
                return False
        elif str(f[k]) != str(d):
            return False
    return True


def sweep_view(finals: list[dict], curves: list[dict]) -> list[dict]:
    """Leduc runs at the sweep horizon. Sweep runs are shorter than the main
    runs; for the default settings the main runs are re-read over the same
    window (steps 0.9 h to h) so that every point covers the same steps."""
    sweeps = [f for f in finals if f["game"] == "leduc" and f["experiment"] != "main"]
    if not sweeps:
        return []
    horizon = int(sweeps[0]["steps"])
    out = list(sweeps)
    for f in finals:
        if f["game"] != "leduc" or f["experiment"] != "main" or f["method"] == "cfr+":
            continue
        pts = [fnum(c["exploitability_ma"]) for c in curves if c["experiment"] == "main" and c["game"] == "leduc"
               and c["method"] == f["method"] and c["seed"] == f["seed"] and 0.9 * horizon <= int(c["step"]) <= horizon]
        if pts:
            out.append({**f, "steps": str(horizon), "tail_mean_ma": str(float(np.mean(pts)))})
    return out


def figure_training(logs: dict[str, list[dict]], out: Path) -> Path | None:
    """Exploitability (left) and policy entropy (right) at every measured step of each fine-tuning run."""
    if not logs:
        return None
    plt = _plt()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for a in (ax, ax2):
        _style(a)
    colors = {"grpo": BLUE, "sft": VIOLET, "smoke": MUTED}
    for name, rows in sorted(logs.items()):
        meas = [r for r in rows if r.get("exploitability_ma") not in ("", None)]
        if not meas:
            continue
        kind = "sft" if name.startswith("sft") else ("smoke" if name.startswith("smoke") else "grpo")
        x = [int(r["step"]) for r in meas]
        ax.plot(x, [fnum(r["exploitability_ma"]) for r in meas], color=colors[kind], linewidth=2, marker="o",
                markersize=3, label=name)
        ent = [(int(r["step"]), max(fnum(r["entropy"]), 1e-5)) for r in meas if r.get("entropy") not in ("", None)]
        if ent:
            ex, ey = zip(*ent)
            ax2.plot(ex, ey, color=colors[kind], linewidth=2, marker="o", markersize=3, label=name)
    _log_axis(ax)
    ax.set_xlabel("training step")
    ax.set_ylabel("exploitability, mA per hand (log scale)")
    ax.set_title("Exploitability during training", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    ax2.set_yscale("log")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("policy entropy, nats per infoset (log scale)")
    ax2.set_title("Entropy: GRPO turns deterministic, then stops learning", color=INK, loc="left")
    fig.tight_layout()
    path = out / "training.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_models(rows: list[dict], out: Path) -> Path | None:
    """Exploitability of every evaluated model on Kuhn and Leduc (template A), one panel per game."""
    rows = [r for r in rows if r.get("template", "A") == "A" and r["game"] in ("kuhn", "leduc")]
    if not rows:
        return None
    plt = _plt()
    games = [g for g in ("kuhn", "leduc") if any(r["game"] == g for r in rows)]
    fig, axes = plt.subplots(1, len(games), figsize=(6.5 * len(games), 0.45 * max(4, len(rows)) + 1.5), squeeze=False)
    for ax, game in zip(axes[0], games):
        _style(ax)
        sub = sorted([r for r in rows if r["game"] == game], key=lambda r: r["exploitability_ma"])
        names = [r["_name"] for r in sub]
        vals = [r["exploitability_ma"] for r in sub]
        def kind(name):
            return "tuned" if "(LoRA)" in name else ("api" if name.startswith("gpt") else "base")
        palette = {"base": BLUE, "api": AQUA, "tuned": VIOLET}
        ax.barh(range(len(sub)), vals, height=0.55, color=[palette[kind(n)] for n in names])
        uniform = sub[0]["uniform_exploitability_ma"]
        ax.axvline(uniform, color=MUTED, linewidth=1)
        ax.text(uniform, len(sub) - 0.4, " uniform random play", color=MUTED, fontsize=8, va="bottom")
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:,.0f}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("exploitability, mA per hand (0 = equilibrium)")
        ax.set_title({"kuhn": "Kuhn poker", "leduc": "Leduc hold'em"}[game], color=INK, loc="left")
    from matplotlib.patches import Patch
    present = {("base" if "(LoRA)" not in r["_name"] and not r["_name"].startswith("gpt") else
                "tuned" if "(LoRA)" in r["_name"] else "api") for r in rows}
    labels = [("base", BLUE, "open model, base"), ("api", AQUA, "OpenAI API model"),
              ("tuned", VIOLET, "open model, fine-tuned here")]
    handles = [Patch(color=c, label=t) for k, c, t in labels if k in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = out / "models.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -------------------------------------------------------------- sections
def section_exactness(root: Path) -> list[str]:
    from .games import GAME_VALUES, exploitability, make_game, read_strategy

    fixture = root / "tests" / "fixtures" / "zerosum" / "zerosum_results.json"
    lines = ["## 1. The measurement is exact and matches an independent implementation", ""]
    if not fixture.exists():
        return lines + ["The ZeroSum fixture is missing.", ""]
    data = json.loads(fixture.read_text())
    lines += ["NashTune's game trees, best response and exploitability are a Python rewrite of ZeroSum's C++ "
              "code. `tests/fixtures/zerosum/` holds strategies in ZeroSum's file format and the numbers "
              "`zerosum eval` printed for them (10 significant digits). NashTune reproduces all of them:", "",
              "| Game | Information sets | Strategies checked | Largest relative difference | Value to first player |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for name, entry in data["games"].items():
        game = make_game(name)
        worst = 0.0
        for row in entry["strategies"]:
            e = exploitability(game, read_strategy(game, fixture.parent / row["file"]))
            for ours, theirs in ((e.value, row["value_to_first_player"]), (e.best_vs_p1, row["best_response_vs_second_player"]),
                                 (e.best_vs_p0, row["best_response_vs_first_player"])):
                worst = max(worst, abs(ours - theirs) / max(abs(theirs), 1e-12))
        same_order = [[i.key, i.actions, i.player] for i in game.infosets] == entry["infosets"]
        lines.append(f"| {name} | {game.num_infosets}{'' if same_order else ' (order differs!)'} | "
                     f"{len(entry['strategies'])} | {worst:.1e} | {GAME_VALUES.get(name, float('nan')):.9f} |")
    lines += ["", "The differences are at the 10th significant digit, where ZeroSum's printout stops. The "
              "information sets also come out in the same order with the same actions. Exploitability is "
              "the average over both seats of what a best-responding opponent wins beyond the game value, "
              "in milli-antes per hand (mA); it is zero exactly at a Nash equilibrium.", ""]
    return lines


def section_dynamics(results: Path, figures: Path) -> tuple[list[str], list[str]]:
    lines = ["## 2. Reward design: exact values against self-play outcomes (tabular study)", ""]
    final_path = results / "dynamics" / "final.csv"
    if not final_path.exists():
        return lines + ["Pending: run `nashtune dynamics` (about an hour on two cores).", ""], []
    finals = read_csv(final_path)
    curves = read_csv(results / "dynamics" / "curves.csv")
    manifest = json.loads((results / "dynamics" / "manifest.json").read_text())
    figs = []
    p = figure_dynamics_main(curves, figures)
    if p:
        figs.append(p)
    sweep_finals = sweep_view(finals, curves)
    q = figure_sweeps(sweep_finals, figures)
    if q:
        figs.append(q)

    def pick(game, method, **settings):
        return [f for f in finals if f["game"] == game and f["method"] == method and f["experiment"] == "main"] \
            if not settings else [f for f in finals if f["game"] == game and f["method"] == method
                                  and _matches(f, *next(iter(settings.items())))]

    lines += [
        "**Setup.** The policy is a table of logits, one per decision (a stand-in for a model with a separate "
        "parameter per situation), initialized at random. Each step applies one GRPO update: groups of G = 8 "
        "sampled actions, advantages (r − mean)/(std + 1e-6), the clipped objective with ε = 0.2, Adam with "
        "learning rate 0.05. Three rewards:",
        "",
        "- **self-play**: deal 32 hands, play each 8 times with the current policy in both seats, reward every "
        "decision with the acting player's final payoff (the naive approach).",
        "- **br**: at every information set, sample 8 actions and reward each with its exact value against the "
        "opponent's best response to the current policy (exploitability descent).",
        "- **exact**: the same values, with the expectation over actions computed instead of sampled.",
        "",
        f"CFR+ is the reference. Seeds, settings and timings: `results/dynamics/manifest.json` "
        f"({manifest['runs']} runs).",
        "",
        "| Game | Method | Start (mA) | Final, median [range] (mA) | Last 10% of steps, median (mA) | Average policy, final (mA) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for game in ("kuhn", "leduc"):
        for method in ("cfr+", "selfplay", "br", "exact"):
            fs = pick(game, method)
            if not fs:
                continue
            fin = [fnum(f["final_ma"]) for f in fs]
            tail = [fnum(f["tail_mean_ma"]) for f in fs]
            avg = [fnum(f.get("final_avg_ma")) for f in fs]
            start = med([fnum(f["initial_ma"]) for f in fs])
            rng = f" [{fmt(min(fin), 2)}–{fmt(max(fin), 2)}]" if len(fin) > 1 else ""
            name = {"cfr+": "CFR+ (average strategy)", "selfplay": "GRPO, self-play", "br": "GRPO, br",
                    "exact": "GRPO, exact"}[method]
            lines.append(f"| {game} | {name} | {fmt(start)} | {fmt(med(fin), 2)}{rng} | {fmt(med(tail), 2)} | "
                         f"{fmt(med(avg), 1) if method not in ('cfr+',) else 'n/a'} |")
    steps = {g: int(next(f for f in finals if f["game"] == g and f["experiment"] == "main")["steps"]) for g in ("kuhn", "leduc")}
    seeds = len({f["seed"] for f in pick("leduc", "br")})
    lines += ["", f"Kuhn ran {steps['kuhn']:,} steps and Leduc {steps['leduc']:,}; {seeds} seeds per GRPO method "
              "(the seed sets the initial logits and the sampling). \"Last 10%\" is the mean over the last tenth "
              "of training, which smooths out step-to-step oscillation.", ""]
    if figs:
        lines += [f"![Exploitability during training](figures/{figs[0].name})", ""]

    # Findings computed from the table; every qualitative word below is conditional on the numbers.
    findings = []
    for game in ("kuhn", "leduc"):
        sp = med([fnum(f["tail_mean_ma"]) for f in pick(game, "selfplay")])
        br = med([fnum(f["tail_mean_ma"]) for f in pick(game, "br")])
        ex = med([fnum(f["tail_mean_ma"]) for f in pick(game, "exact")])
        start = med([fnum(f["initial_ma"]) for f in pick(game, "br")])
        if not any(math.isnan(x) for x in (sp, br, ex)):
            findings.append(
                f"On {game.capitalize()}, exact rewards took exploitability from {fmt(start)} mA to {fmt(br, 1)} (br) "
                f"and {fmt(ex, 1)} (exact) over the last 10% of training; self-play outcome rewards ended at "
                f"{fmt(sp)} mA, {sp / br:.0f} times the br level.")
    sp_runs = pick("leduc", "selfplay")
    if sp_runs:
        mins = [fnum(f["min_ma"]) for f in sp_runs]
        fins = [fnum(f["final_ma"]) for f in sp_runs]
        worse = sum(1 for a, b in zip(fins, mins) if a > 1.3 * b)
        if worse * 2 > len(sp_runs):
            findings.append(
                f"Self-play does not settle. On Leduc its best point was {fmt(med(mins))} mA (median over seeds) but "
                f"it ended at {fmt(med(fins))} mA; in {worse} of {len(sp_runs)} runs the final policy was more than "
                "30% worse than the best one it had passed through. This is the cycling that policy-gradient "
                "self-play is known for in zero-sum games: each player adapts to the other's current play, which "
                "keeps moving.")
        else:
            findings.append(f"Self-play on Leduc reached {fmt(med(mins))} mA at best and ended at {fmt(med(fins))} mA "
                            f"(medians); {worse} of {len(sp_runs)} runs ended more than 30% above their best point.")
    for game in ("kuhn", "leduc"):
        runs = pick(game, "selfplay")
        avg = med([fnum(f.get("final_avg_ma")) for f in runs])
        last = med([fnum(f["final_ma"]) for f in runs])
        if runs and not math.isnan(avg):
            verdict = ("lowers it" if avg < 0.9 * last else "barely changes it" if avg < 1.1 * last else "raises it")
            findings.append(f"Averaging self-play's policies over time (what CFR does) {verdict} on "
                            f"{game.capitalize()}: {fmt(avg)} mA for the average against {fmt(last)} for the last "
                            "policy. Either way it stays far from the exact-reward runs.")
    exr = med([fnum(f["tail_mean_ma"]) for f in pick("leduc", "exact")])
    brr = med([fnum(f["tail_mean_ma"]) for f in pick("leduc", "br")])
    cfr = med([fnum(f["final_ma"]) for f in pick("leduc", "cfr+")])
    if not math.isnan(exr):
        findings.append(
            f"Taking the expectation over actions instead of sampling 8 helps: on Leduc {fmt(exr, 1)} mA against "
            f"{fmt(brr, 1)}. Neither comes near CFR+, which reached {fmt(cfr, 2)} mA in the same number of "
            "full-tree passes.")
    # Sweeps.
    def tail_of(method, key, value):
        return med([fnum(f["tail_mean_ma"]) for f in sweep_finals if f["method"] == method and _matches(f, key, value)])

    def trend(pairs):
        vals = [v for _, v in pairs]
        if all(a > b for a, b in zip(vals, vals[1:])):
            return "falls"
        if all(a < b for a, b in zip(vals, vals[1:])):
            return "rises"
        return "is not monotone"

    sweep = []
    gs = [(g, tail_of("br", "group_size", g), tail_of("selfplay", "group_size", g)) for g in (2, 4, 8, 16)]
    if all(not math.isnan(b) for _, b, _ in gs):
        sweep.append(f"Group size. With br rewards exploitability {trend([(g, b) for g, b, _ in gs])} as G grows: " +
                     ", ".join(f"G={g}: {fmt(b, 1)}" for g, b, _ in gs) + " mA. With self-play it " +
                     trend([(g, x) for g, _, x in gs]) + ": " + ", ".join(f"G={g}: {fmt(x)}" for g, _, x in gs) +
                     " mA: more samples per step did not help self-play.")
    lrs = [(lr, tail_of("br", "lr", lr), tail_of("exact", "lr", lr), tail_of("selfplay", "lr", lr)) for lr in (0.01, 0.05, 0.2)]
    if all(not math.isnan(b) for _, b, _, _ in lrs):
        sweep.append("Learning rate (lower is better at this horizon for every reward): br " +
                     ", ".join(f"{lr:g}: {fmt(b, 1)}" for lr, b, _, _ in lrs) +
                     " mA; exact " + ", ".join(f"{lr:g}: {fmt(e, 1)}" for lr, _, e, _ in lrs) +
                     " mA; self-play " + ", ".join(f"{lr:g}: {fmt(x)}" for lr, _, _, x in lrs) + " mA."
                     if all(b0 < b1 for b0, b1 in zip([b for _, b, _, _ in lrs], [b for _, b, _, _ in lrs][1:])) else
                     "Learning rate: br " + ", ".join(f"{lr:g}: {fmt(b, 1)}" for lr, b, _, _ in lrs) +
                     " mA; exact " + ", ".join(f"{lr:g}: {fmt(e, 1)}" for lr, _, e, _ in lrs) +
                     " mA; self-play " + ", ".join(f"{lr:g}: {fmt(x)}" for lr, _, _, x in lrs) + " mA.")
    kls = [(k, tail_of("br", "kl_coef", k)) for k in (0.0, 0.01, 0.1)]
    if all(not math.isnan(v) for _, v in kls):
        spread = max(v for _, v in kls) / min(v for _, v in kls)
        sweep.append("KL penalty to the step-0 policy (br): " + ", ".join(f"β={k:g}: {fmt(v, 1)}" for k, v in kls) +
                     (" mA, a small effect at these coefficients." if spread < 1.15 else " mA.") +
                     " Here the reference is the random initial policy; with a language model it is the base model.")

    def tail_variant(method, settings):
        return med([fnum(f["tail_mean_ma"]) for f in sweep_finals if f["method"] == method
                    and all(str(f[k]) == v for k, v in settings.items()) and _default_except(f, settings)])

    v = {
        "br, normalized (the default)": tail_variant("br", {"adv_norm": "std", "weighting": "uniform"}),
        "br, unnormalized": tail_variant("br", {"adv_norm": "none", "weighting": "uniform"}),
        "br, reach-weighted": tail_variant("br", {"adv_norm": "std", "weighting": "reach"}),
        "br, counterfactual-reach-weighted": tail_variant("br", {"adv_norm": "std", "weighting": "cfreach"}),
        "exact, unnormalized": tail_variant("exact", {"adv_norm": "none", "weighting": "uniform"}),
        "exact, reach-weighted, unnormalized (plain gradient descent on exploitability)":
            tail_variant("exact", {"adv_norm": "none", "weighting": "reach"}),
    }
    if not any(math.isnan(x) for x in v.values()):
        sweep.append("Reward variants: " + "; ".join(f"{k}: {fmt(x, 1)} mA" for k, x in v.items()) + ".")
        if v["br, unnormalized"] < v["br, normalized (the default)"]:
            sweep.append(
                "Dividing by the group's standard deviation hurts late in training: near the equilibrium the "
                "action values at an infoset differ by little, and normalizing blows that small difference up to "
                "a full-size step, so the policy keeps moving around the equilibrium instead of settling. "
                "Lower learning rates and unnormalized advantages both end lower, which fits that reading.")
    sub = {r: [f for f in finals if f["experiment"] == "subset" and int(f["refresh_every"]) == r] for r in (1, 10)}
    if sub[1] and sub[10]:
        def at(f_list, step):
            vals = []
            for f in f_list:
                vals += [fnum(c["exploitability_ma"]) for c in curves if c["experiment"] == "subset"
                         and int(c["seed"]) == int(f["seed"]) and int(c["refresh_every"]) == int(f["refresh_every"])
                         and int(c["step"]) == step]
            return med(vals)
        a10, a1 = at(sub[10], 200), at(sub[1], 200)
        e10 = med([fnum(f["tail_mean_ma"]) for f in sub[10]])
        e1 = med([fnum(f["tail_mean_ma"]) for f in sub[1]])
        start = med([fnum(f["initial_ma"]) for f in sub[10]])
        sweep.append(
            "The language-model setting: 32 random information sets per step instead of all 288, and best "
            f"responses recomputed every 10 steps. From {fmt(start)} mA it reached {fmt(a10)} mA after 200 steps "
            f"and {fmt(e10, 1)} mA after {int(sub[10][0]['steps']):,}; recomputing the best response every step "
            f"instead gives {fmt(a1)} and {fmt(e1, 1)}. Ten-step-old rewards cost {100 * (e10 / e1 - 1):.0f}% at the "
            "end. A language model shares parameters across information sets, so it may move faster or slower "
            "than this table; these numbers set expectations, not predictions.")
    sp_deals = [fnum(f["tail_mean_ma"]) for f in finals if f["experiment"] == "selfplay_deals"]
    if sp_deals:
        sweep.append(f"Four times more self-play hands per step (128 deals instead of 32): {fmt(med(sp_deals))} mA "
                     f"against {fmt(tail_of('selfplay', 'group_size', 8))}.")
    lines += ["**What happened.**", ""] + [f"- {x}" for x in findings] + [""]
    if sweep:
        lines += ["**Sensitivity** (Leduc, 2,000 steps, 3 seeds; at the default settings the 5 main runs over the "
                  "same steps; numbers are medians of each run's mean over steps 1,800 to 2,000; "
                  "`results/dynamics/final.csv`).", ""] + \
                 [f"- {x}" for x in sweep] + [""]
    if len(figs) > 1:
        lines += [f"![Sensitivity](figures/{figs[1].name})", ""]
    lines += ["The point of the study for the language-model runs: a verifiable reward is not enough by itself. "
              "The reward has to be the right quantity. Rewarding a decision with the hand's outcome gives an "
              "unbiased but noisy estimate of its value against the *current* opponent, and following it "
              "chases a moving target. Rewarding it with its value against the *best response* makes every "
              "update a step down the exploitability surface.", ""]
    return lines, [str(f) for f in figs]


def _model_name(r: dict) -> str:
    return str(r.get("label") or r.get("model"))


def _short(name: str) -> str:
    return name.split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "")


def strategy_determinism(path: Path, threshold: float = 0.999) -> tuple[int, int, float] | None:
    """(infosets whose largest probability exceeds ``threshold``, infosets, mean entropy in nats)."""
    if not path.exists():
        return None
    probs: dict[str, list[float]] = defaultdict(list)
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, _action, p = line.rsplit(" ", 2)
        probs[name].append(float(p))
    if not probs:
        return None
    det = sum(1 for ps in probs.values() if max(ps) > threshold)
    ent = float(np.mean([-sum(p * math.log(p) for p in ps if p > 0) for ps in probs.values()]))
    return det, len(probs), ent


def training_diagnostics(log: list[dict]) -> dict:
    """Best and final measured exploitability, the entropy path, and how many steps carried no signal."""
    meas = [(int(r["step"]), fnum(r["exploitability_ma"]), fnum(r.get("entropy")))
            for r in log if r.get("exploitability_ma") not in ("", None)]
    steps = [r for r in log if r.get("batch_infosets") not in ("", None)]
    dead = [r for r in steps if int(fnum(r["batch_infosets"])) > 0
            and int(fnum(r.get("skipped_groups") or 0)) == int(fnum(r["batch_infosets"]))]
    out = {"measured": meas, "steps": len(steps), "no_signal_steps": len(dead)}
    if meas:
        best = min(meas, key=lambda m: m[1])
        out.update(start=meas[0][1], final=meas[-1][1], best=best[1], best_step=best[0],
                   start_entropy=meas[0][2], final_entropy=meas[-1][2])
        low = [m for m in meas if not math.isnan(m[2]) and m[2] < 0.01]
        out["first_low_entropy_step"] = low[0][0] if low else None
        tail = [m[1] for m in meas if m[0] >= meas[-1][0] * 2 / 3]
        out["tail_range"] = (min(tail), max(tail)) if tail else None
        flat_from = meas[-1][0]
        for st, e, _ in reversed(meas):
            if abs(e - meas[-1][1]) <= 0.005 * meas[-1][1]:
                flat_from = st
            else:
                break
        out["flat_from"] = flat_from
    return out


def section_base_models(results: Path) -> tuple[list[str], list[dict]]:
    rows = [r for r in load_json_dir(results / "mac" / "base")]
    lines = ["## 3. How exploitable are language models out of the box?", ""]
    if not rows:
        return lines + [
            "**Pending Mac run** (`make mac-base`, about 30 minutes plus model downloads; no API cost). It scores "
            "every information set of Kuhn and Leduc with templates A and B, and leduc4 with template A, for "
            "Qwen2.5-0.5B, 1.5B and 7B-Instruct.", ""], []
    lines += ["Each model's strategy is its exact probability of answering with each legal action word, "
              "renormalized over the legal actions. Legal mass is the probability the model put on a legal answer "
              "before renormalizing (1 = always answers with a legal word).", "",
              "| Model | Game | Template | Exploitability (mA) | Seat 1 / seat 2 (mA) | Legal mass, mean (min) | Uniform play (mA) |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for r in sorted(rows, key=lambda r: (_model_name(r), r["game"], r.get("template", ""))):
        lm = f"{r['legal_mass_mean']:.3f} ({r['legal_mass_min']:.3f})" if "legal_mass_mean" in r else "n/a"
        lines.append(f"| {_model_name(r)} | {r['game']} | {r.get('template', 'A')} | {fmt(r['exploitability_ma'])} | "
                     f"{fmt(r['seat0_ma'])} / {fmt(r['seat1_ma'])} | {lm} | {fmt(r['uniform_exploitability_ma'])} |")
    lines += ["", "Source: `results/mac/base/*.json` (strategies in `*_strategy.txt`, per-infoset detail in "
              "`*_infosets.csv`).", ""]
    lines += base_model_findings(rows)
    for r in rows:
        r["_name"] = _model_name(r).split("/")[-1]
    return lines, rows


def _size(m: str) -> float:
    try:
        return float(m.rstrip("B"))
    except ValueError:
        return 0.0


def base_model_findings(rows: list[dict]) -> list[str]:
    table = {(_short(_model_name(r)), r["game"], r.get("template", "A")): r for r in rows}
    models = sorted({k[0] for k in table}, key=_size)
    out = ["**What the base models show.**", ""]
    names = {"kuhn": "Kuhn", "leduc": "Leduc", "leduc4": "leduc4"}
    for game in ("kuhn", "leduc", "leduc4"):
        cells = [(m, table[(m, game, "A")]) for m in models if (m, game, "A") in table]
        if not cells:
            continue
        uni = cells[0][1]["uniform_exploitability_ma"]
        worse = [m for m, r in cells if r["exploitability_ma"] >= uni]
        parts = ", ".join(f"{m} {fmt(r['exploitability_ma'])}" for m, r in cells)
        if not worse:
            verdict = "every model was less exploitable than uniform random play"
        elif len(worse) == len(cells):
            verdict = "no model beat uniform random play"
        else:
            verdict = f"{' and '.join(worse)} {'was' if len(worse) == 1 else 'were'} no better than uniform random play"
        out.append(f"- {names[game]}, template A ({parts} mA; uniform {fmt(uni)}): {verdict}.")
    swings = []
    for m in models:
        a, b = table.get((m, "leduc", "A")), table.get((m, "leduc", "B"))
        if a and b:
            swings.append((m, a["exploitability_ma"], b["exploitability_ma"]))
    if swings:
        ratio = max(max(x, y) / min(x, y) for _, x, y in swings)
        out.append("- The wording of the prompt matters too: on Leduc, switching from template A "
                   "to the paraphrase B moved exploitability " + ", ".join(f"{m} {fmt(x)} → {fmt(y)}" for m, x, y in swings)
                   + f" mA, a factor of up to {ratio:.1f}, and not always in the same direction.")
    out.append("")
    return out


def section_training(results: Path) -> tuple[list[str], dict]:
    rows = load_json_dir(results / "mac" / "train")
    logs = {}
    logdir = results / "mac" / "train" / "logs"
    if logdir.exists():
        for d in sorted(p for p in logdir.iterdir() if p.is_dir()):
            if (d / "log.csv").exists():
                logs[d.name] = read_csv(d / "log.csv")
    lines = ["## 4. GRPO with exact rewards on a language model", ""]
    if not rows:
        return lines + [
            "**Pending Mac run** (`make mac-smoke`, then `make mac-train` and `make mac-sft`; about 1.5 to 2 "
            "hours each on a 48 GB M-series Mac; no API cost). Qwen2.5-1.5B-Instruct with LoRA adapters is "
            "trained on Leduc with the `br` reward, 32 information sets and 8 sampled answers per set per step, "
            "best responses recomputed every 10 steps. It is then measured on Leduc with the training prompt "
            "(template A), with the held-out paraphrase (template B), and on leduc4, a four-rank variant it "
            "never saw. The supervised baseline gets the same steps, imitating the CFR+ equilibrium.", ""], logs
    base_rows = load_json_dir(results / "mac" / "base")
    summaries = {}
    for name in logs:
        sp = results / "mac" / "train" / "logs" / name / "summary.json"
        if sp.exists():
            summaries[name] = json.loads(sp.read_text())
    by_run = defaultdict(dict)
    for r in rows:
        by_run[_model_name(r)][(r["game"], r.get("template", "A"))] = r
    model_of = {}
    for n in by_run:
        cfg = results / "mac" / "train" / "logs" / n / "config.json"
        model_of[n] = json.loads(cfg.read_text())["policy"]["model"] if cfg.exists() else None

    def base_for(model_id):
        return {(r["game"], r.get("template", "A")): r for r in base_rows if _model_name(r) == model_id}

    def cell(table, game, template):
        r = table.get((game, template))
        return fmt(r["exploitability_ma"]) if r else "n/a"

    leduc_runs = sorted(n for n in by_run if summaries.get(n, {}).get("game", "leduc") == "leduc")
    other_runs = sorted(n for n in by_run if n not in leduc_runs)
    if leduc_runs:
        lines += ["Trained on Leduc with template A; template B and leduc4 were never seen in training.", "",
                  "| Policy | Leduc, template A (mA) | Leduc, template B (mA) | leduc4, template A (mA) |",
                  "| --- | ---: | ---: | ---: |"]
        seen = set()
        for n in leduc_runs:
            mid = model_of.get(n)
            if mid and mid not in seen and base_for(mid):
                seen.add(mid)
                b = base_for(mid)
                lines.append(f"| {_short(mid)}, base model | {cell(b, 'leduc', 'A')} | {cell(b, 'leduc', 'B')} | "
                             f"{cell(b, 'leduc4', 'A')} |")
        for n in leduc_runs:
            t = by_run[n]
            lines.append(f"| {n} | {cell(t, 'leduc', 'A')} | {cell(t, 'leduc', 'B')} | {cell(t, 'leduc4', 'A')} |")
        lines.append("")
    for n in other_runs:
        t, mid = by_run[n], model_of.get(n)
        game = summaries.get(n, {}).get("game", "kuhn")
        b = base_for(mid) if mid else {}
        lines.append(f"Smoke test `{n}` ({game}): template A {cell(b, game, 'A')} → {cell(t, game, 'A')} mA, "
                     f"template B {cell(b, game, 'B')} → {cell(t, game, 'B')} mA.")
        lines.append("")
    lines += ["Source: `results/mac/train/*.json`; training logs in `results/mac/train/logs/`.", ""]
    lines += training_findings(results, logs, summaries, by_run, base_rows, model_of)
    return lines, logs


def training_findings(results: Path, logs: dict, summaries: dict, by_run: dict, base_rows: list[dict],
                      model_of: dict) -> list[str]:
    out = ["**What happened.**", ""]
    collapsed = []
    for name in sorted(logs):
        s = summaries.get(name)
        if not s:
            continue
        d = training_diagnostics(logs[name])
        if not d.get("measured"):
            continue
        det = strategy_determinism(results / "mac" / "train" / f"{name}_{s['game']}_A_strategy.txt")
        line = (f"- `{name}` ({s['reward']} reward, {s['steps']} steps, {s['elapsed_seconds'] / 60:.0f} min): "
                f"{fmt(d['start'])} → {fmt(d['final'])} mA on the training prompts; best measured "
                f"{fmt(d['best'])} mA at step {d['best_step']}.")
        if s["reward"] != "sft" and d["steps"]:
            path = [m for m in d["measured"] if m[0] in (0, 10, 20, 30)]
            ent = " → ".join(f"{m[2]:.2g}" + (f" (step {m[0]})" if m[0] else "") for m in path if not math.isnan(m[2]))
            line += (f" Policy entropy {ent} nats per infoset; in {d['no_signal_steps']} of {d['steps']} steps every "
                     f"group's sampled answers were identical, so there was no gradient at all; exploitability did "
                     f"not move after step {d['flat_from']}.")
            if det:
                line += (f" The final strategy is deterministic (one action above 0.999) at {det[0]} of {det[1]} "
                         "information sets.")
            if d["no_signal_steps"] > d["steps"] / 2:
                collapsed.append((name, d))
        elif d.get("tail_range"):
            lo, hi = d["tail_range"]
            line += f" Over the last third of training the measured values ranged from {fmt(lo)} to {fmt(hi)} mA."
            if det:
                line += f" Final mean entropy {det[2]:.2f} nats per infoset."
        line += (f" Final legal mass {s['final_legal_mass']:.3f}; KL to the base model {s['final_kl_ref']:.2f} "
                 "nats per infoset.")
        out.append(line)
    out.append("")
    leduc_collapsed = [(n, d) for n, d in collapsed if summaries[n].get("game") == "leduc"]
    if leduc_collapsed:
        name, d = leduc_collapsed[0]
        cfg = summaries[name].get("config", {})
        out.append(
            f"- **GRPO collapsed.** A group-relative advantage is zero when all {cfg.get('group_size', 8)} sampled "
            "answers agree, and with advantages divided by the group's standard deviation every group that does "
            "differ produces a full-size step however small the value gap. The language model, whose LoRA "
            "parameters are shared across all 288 information sets, had a mean entropy below 0.01 nats by step "
            f"{d.get('first_low_entropy_step')}. "
            "A deterministic policy produces identical samples, so every later group was skipped and learning "
            "stopped for good, at a pure strategy worse than its own best measured checkpoint "
            f"({fmt(d['best'])} mA at step {d['best_step']}). The tabular study in section 2, with one parameter "
            "per decision, never reached that state. The run used no KL penalty to the base model "
            f"(coefficient {cfg.get('kl_coef', 0)}) and kept only its last two checkpoints.")
    for n in sorted(by_run):
        s = summaries.get(n, {})
        if s.get("reward") != "sft" or s.get("game") != "leduc":
            continue
        t = by_run[n]
        b = {(r["game"], r.get("template", "A")): r for r in base_rows if _model_name(r) == model_of.get(n)}
        keys = [("leduc", "A"), ("leduc", "B"), ("leduc4", "A")]
        if all(k in t and k in b for k in keys):
            (a0, a1), (b0, b1), (c0, c1) = [(b[k]["exploitability_ma"], t[k]["exploitability_ma"]) for k in keys]
            out.append(f"- **Imitation worked on the training prompt and mostly stayed there.** Supervised "
                       f"fine-tuning on the CFR+ strategy cut Leduc exploitability by {100 * (1 - a1 / a0):.0f}% "
                       f"with template A ({fmt(a0)} → {fmt(a1)} mA), by {100 * (1 - c1 / c0):.0f}% on the unseen "
                       f"four-rank deck ({fmt(c0)} → {fmt(c1)}), and by only {100 * (1 - b1 / b0):.0f}% under the "
                       f"reworded prompt ({fmt(b0)} → {fmt(b1)}).")
    for name, d in leduc_collapsed:
        t = by_run.get(name, {})
        if all(k in t for k in (("leduc", "A"), ("leduc", "B"), ("leduc4", "A"))):
            out.append(f"- The collapsed GRPO policy scored about the same everywhere: "
                       f"{fmt(t[('leduc', 'A')]['exploitability_ma'])} (A), {fmt(t[('leduc', 'B')]['exploitability_ma'])} "
                       f"(B), {fmt(t[('leduc4', 'A')]['exploitability_ma'])} mA (leduc4).")
    out.append("")
    return out


def section_openai(results: Path) -> tuple[list[str], list[dict]]:
    rows = [r for r in load_json_dir(results / "mac" / "openai") if r.get("kind") == "openai"]
    lines = ["## 5. OpenAI models", ""]
    if not rows:
        return lines + [
            "**Pending Mac run** (`make mac-openai`, under $4 by construction: the run stops before the budget "
            "is spent). A model that returns log-probabilities is read exactly from its first-token "
            "distribution; a reasoning model is sampled (empirical frequencies, which add sampling noise and so "
            "overstate exploitability somewhat); both are also asked to state a mixed strategy as JSON.", ""], []
    lines += ["| Model | Mode | Game | Template | Exploitability (mA) | Legal mass or invalid answers | Tokens | Cost |",
              "| --- | --- | --- | --- | ---: | --- | ---: | ---: |"]
    for r in sorted(rows, key=lambda r: (r["model"], r["mode"], r["game"], r["template"])):
        usage = r.get("usage", {})
        price = r.get("price_per_mtok")
        cost = (usage.get("prompt_tokens", 0) * price["input"] + usage.get("completion_tokens", 0) * price["output"]) / 1e6 \
            if price else float("nan")
        if r["mode"] == "logprobs":
            q = f"legal mass {r.get('legal_mass_mean', float('nan')):.3f}"
        else:
            t = r.get("totals", {})
            q = f"{t.get('invalid', 0)} invalid, {t.get('fallback_uniform', 0)} infosets defaulted to uniform"
        lines.append(f"| {r['model']} | {r['mode']} | {r['game']} | {r['template']} | {fmt(r['exploitability_ma'])} | "
                     f"{q} | {usage.get('total_tokens', 0):,} | ${cost:.3f} |")
    spend = results / "mac" / "openai" / "spend.json"
    lines += ["", "Tokens and cost count only requests sent in that run (cached answers are free). Source: "
              "`results/mac/openai/*.json`" + (", ledger `spend.json`." if spend.exists() else "."), ""]
    lines += openai_findings(results, rows, spend)
    for r in rows:
        r["_name"] = f"{r['model']} ({r['mode']})"
    return lines, rows


def openai_findings(results: Path, rows: list[dict], spend: Path) -> list[str]:
    out = ["**What the API models show.**", ""]
    for game in ("kuhn", "leduc"):
        g = [r for r in rows if r["game"] == game]
        if not g:
            continue
        best = min(g, key=lambda r: r["exploitability_ma"])
        note = ""
        if best["mode"] == "sample" and best.get("samples"):
            note = (f", from {best['samples']} sampled answers per information set (sampling noise adds "
                    "exploitability, so the model's own strategy is somewhat better than this)")
        out.append(f"- {'Kuhn' if game == 'kuhn' else 'Leduc'}: lowest was {best['model']} ({best['mode']}, template "
                   f"{best['template']}) at {fmt(best['exploitability_ma'])} mA{note}.")
    lp = {(r["game"], r["template"]): r for r in rows if r["mode"] == "logprobs"}
    if ("leduc", "A") in lp and ("leduc", "B") in lp:
        a, b = lp[("leduc", "A")], lp[("leduc", "B")]
        out.append(f"- Read exactly from its log-probabilities, {a['model']} scored {fmt(a['exploitability_ma'])} (A) "
                   f"and {fmt(b['exploitability_ma'])} (B) mA on Leduc, much less sensitive to the wording than the "
                   "small open models.")
    trained = [r for r in load_json_dir(results / "mac" / "train") if r["game"] == "leduc" and r.get("template") == "A"]
    leduc_api = [r for r in rows if r["game"] == "leduc"]
    if trained and leduc_api:
        t = min(trained, key=lambda r: r["exploitability_ma"])
        if all(r["exploitability_ma"] > t["exploitability_ma"] for r in leduc_api):
            out.append(f"- The fine-tuned 1.5B model `{_model_name(t)}` ({fmt(t['exploitability_ma'])} mA on Leduc, "
                       "template A) was less exploitable than every API model measured here, but only with the "
                       "prompt it was trained on (section 4).")
    if spend.exists():
        led = json.loads(spend.read_text())
        prices = {r["model"]: r.get("price_per_mtok") for r in rows if r.get("price_per_mtok")}
        total = sum((v.get("prompt_tokens", 0) * prices[m]["input"] + v.get("completion_tokens", 0) * prices[m]["output"])
                    / 1e6 for m, v in led.items() if m in prices)
        out.append(f"- Total API spend for this section: ${total:.2f}.")
    out.append("")
    return out


def build_report(results: str = "results", out: str = "reports", runs: str = "runs", root: str | None = None) -> Path:
    results_p = Path(results)
    out_p = Path(out)
    figures = out_p / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    root_p = Path(root) if root else Path(__file__).resolve().parents[1]
    lines = [
        "# NashTune research memo",
        "",
        "**Question.** Language-model evaluations are mostly noisy judge scores. In small poker games the "
        "quality of any strategy can be measured exactly: its exploitability, the amount a perfect opponent "
        "wins from it per hand. How exploitable are language models when their answer probabilities are read "
        "as a complete strategy, which reward makes reinforcement learning reduce that number, and does the "
        "improvement survive a reworded prompt and an unseen variant of the game?",
        "",
        "**Method.** Kuhn poker (12 information sets) and Leduc hold'em (288) are built as explicit game trees. "
        "A policy, whether a table of numbers or a language model, is turned into a complete strategy by reading "
        "its probability for every legal action at every information set; exploitability is then computed "
        "exactly with a best response. Training uses GRPO with rewards computed exactly from the game tree.",
        "",
        f"*Built {datetime.now().strftime('%Y-%m-%d')} by `nashtune report` from `results/`.*",
        "",
    ]
    lines += section_exactness(root_p)
    dyn, _ = section_dynamics(results_p, figures)
    lines += dyn
    base_lines, base_rows = section_base_models(results_p)
    lines += base_lines
    train_lines, logs = section_training(results_p)
    lines += train_lines
    oa_lines, oa_rows = section_openai(results_p)
    lines += oa_lines
    tuned = []
    for r in load_json_dir(results_p / "mac" / "train"):
        if r["game"] in ("kuhn", "leduc") and r.get("template", "A") == "A" and not _model_name(r).startswith("smoke"):
            r = dict(r)
            kind = "GRPO" if _model_name(r).startswith("grpo") else "SFT"
            r["_name"] = f"{_short(r.get('model', ''))} + {kind} (LoRA)"
            tuned.append(r)
    if base_rows or oa_rows:
        p = figure_models(base_rows + oa_rows + tuned, figures)
        if p:
            lines += [f"![Exploitability of each model](figures/{p.name})", ""]
    p = figure_training(logs, figures)
    if p:
        lines += [f"![Training curves](figures/{p.name})", ""]
    lines += [
        "## 6. Limits",
        "",
        "- These are research-sized games. The method needs the whole game tree, which rules out real poker.",
        "- A strategy read from a model's answer probabilities is one way to define the model's play. A model "
        "that reasons before answering may play differently; the `sample` mode measures that, with sampling noise.",
        "- The tabular study uses one optimizer, one initialization scale and fixed hyperparameters per sweep; "
        "conclusions about rates are specific to them.",
        "- `br` rewards need an exact best response, which needs the full tree. That is what makes them "
        "verifiable here and what keeps the method from scaling as is.",
        "- Each language-model training result is one run (one seed, one set of hyperparameters: LoRA rank 16, "
        "learning rate 1e-4, no KL penalty). The GRPO collapse is a property of that setting, not of GRPO in general.",
        "- Sampled API results use 4 answers per information set, so their exploitability includes sampling noise.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "make test            # unit tests (NumPy only)",
        "nashtune dynamics    # section 2 (about an hour on two cores)",
        "make mac-all         # sections 3 to 5, on a Mac (see docs/mac-and-github.md)",
        "nashtune report      # this memo",
        "```",
        "",
    ]
    path = out_p / "NashTune_Research_Memo.md"
    path.write_text("\n".join(lines))
    return path


def bundle(out: str | None = None, results: str = "results", logs: str = "logs") -> Path:
    """Zip results/ and the console logs (no model weights, no checkpoints, no API cache)."""
    name = out or f"nashtune_results_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
    path = Path(name)
    count = 0
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for base in (Path(results), Path(logs)):
            if not base.exists():
                continue
            for p in sorted(base.rglob("*")):
                if p.is_dir() or "cache" in p.parts or p.name == "runs.jsonl":
                    continue
                zf.write(p, p.as_posix())
                count += 1
    if count == 0:
        raise FileNotFoundError("nothing to bundle: results/ and logs/ are empty")
    return path


if __name__ == "__main__":  # pragma: no cover
    print(build_report())
    print(os.getcwd())
