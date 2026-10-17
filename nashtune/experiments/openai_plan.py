"""Experiment D: OpenAI models on Kuhn and Leduc within a dollar budget.

The plan (configs/openai_plan.json) names models by placeholder
(``$CHAT_MODEL`` for a model that returns log-probabilities, ``$REASONING_MODEL``
for one that does not); you supply real names from ``nashtune openai-models``.
Prices per million tokens come from a JSON file you fill in from OpenAI's
pricing page. Before anything is sent, the dry run prints every run's request
count, estimated input tokens, an upper bound on output tokens, and the cost
bound. Spending is recorded in a ledger, and each run gets a token cap equal to
the remaining budget priced at the model's higher (output) rate, so the total
cannot exceed the budget.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..evaluate import evaluate, slug
from ..games import make_game

PLACEHOLDERS = ("$CHAT_MODEL", "$REASONING_MODEL")


def load_plan(path: str | os.PathLike, chat_model: str | None, reasoning_model: str | None) -> dict:
    plan = json.loads(Path(path).read_text())
    names = {"$CHAT_MODEL": chat_model, "$REASONING_MODEL": reasoning_model}
    runs = []
    for run in plan["runs"]:
        run = dict(run)
        if run["model"] in names:
            if not names[run["model"]]:
                continue  # that model was not given: skip its runs
            run["model"] = names[run["model"]]
        runs.append(run)
    plan["runs"] = runs
    return plan


def run_prefix(out_dir: Path, run: dict) -> Path:
    return out_dir / f"{slug(run['model'])}_{run['mode']}_{run['game']}_{run['template']}"


def _policy(run: dict, client, echo):
    from ..policy.openai_backend import OpenAIPolicy

    return OpenAIPolicy(make_game(run["game"]), client, run["model"], mode=run["mode"], template=run["template"],
                        samples=int(run.get("samples", 8)), top_logprobs=int(run.get("top_logprobs", 20)),
                        max_completion_tokens=run.get("max_completion_tokens"),
                        reasoning_effort=run.get("reasoning_effort"), echo=echo)


def price_of(prices: dict, model: str) -> tuple[float, float] | None:
    p = prices.get(model)
    if not p:
        return None
    return float(p["input"]), float(p["output"])


def ledger_path(out_dir: Path) -> Path:
    return out_dir / "spend.json"


def spent_usd(out_dir: Path, prices: dict) -> float:
    path = ledger_path(out_dir)
    if not path.exists():
        return 0.0
    total = 0.0
    for model, u in json.loads(path.read_text()).items():
        pr = price_of(prices, model)
        if pr:
            total += (u["prompt_tokens"] * pr[0] + u["completion_tokens"] * pr[1]) / 1e6
    return total


def record_spend(out_dir: Path, model: str, usage: dict) -> None:
    path = ledger_path(out_dir)
    data = json.loads(path.read_text()) if path.exists() else {}
    entry = data.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "requests": 0})
    for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "requests"):
        entry[k] += int(usage.get(k, 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


def dry_run(plan: dict, prices: dict, out_dir: Path, echo=print) -> dict:
    from ..policy.openai_backend import OpenAIClient

    client = OpenAIClient(api_key="dry-run", cache_dir=out_dir / "cache")
    total_bound = total_likely = 0.0
    missing = []
    echo(f"{'model':28s} {'mode':8s} {'game':6s} T  {'requests':>8s} {'in tok':>9s} {'out max':>9s} "
         f"{'$ max':>7s} {'$ likely':>8s}  done")
    for run in plan["runs"]:
        policy = _policy(run, client, echo)
        est = policy.plan()
        likely_out = est["requests"] * est["samples_per_infoset"] * int(run.get("expected_output_tokens", 1))
        done = run_prefix(out_dir, run).with_suffix(".json").exists()
        pr = price_of(prices, run["model"])
        if pr is None:
            missing.append(run["model"])
            bound = likely = float("nan")
        else:
            bound = (est["estimated_prompt_tokens"] * pr[0] + est["max_output_tokens"] * pr[1]) / 1e6
            likely = (est["estimated_prompt_tokens"] * pr[0] + likely_out * pr[1]) / 1e6
            if not done:
                total_bound += bound
                total_likely += likely
        echo(f"{run['model'][:28]:28s} {run['mode']:8s} {run['game']:6s} {run['template']}  {est['requests']:8d} "
             f"{est['estimated_prompt_tokens']:9d} {est['max_output_tokens']:9d} {bound:7.3f} {likely:8.3f}  "
             f"{'yes' if done else ''}")
    spent = spent_usd(out_dir, prices)
    budget = float(plan.get("budget_usd", 4.0))
    echo(f"\nRemaining runs: at most ${total_bound:.2f}, likely ${total_likely:.2f}. "
         f"Already spent ${spent:.2f} of the ${budget:.2f} budget.")
    echo("Input tokens are estimated at four characters per token. 'Max' assumes every answer uses its "
         "max_completion_tokens; 'likely' uses the plan's expected_output_tokens.")
    if missing:
        echo(f"No price for: {', '.join(sorted(set(missing)))}. Add them to the prices file before running.")
    return {"bound_usd": total_bound, "likely_usd": total_likely, "spent_usd": spent, "missing_prices": missing}


def run_plan(plan: dict, prices: dict, out_dir: Path, budget_usd: float | None = None, echo=print,
             client_kwargs: dict | None = None) -> list[dict]:
    from ..policy.openai_backend import BudgetExceeded, OpenAIClient, UnsupportedParameter

    out_dir.mkdir(parents=True, exist_ok=True)
    budget = float(budget_usd if budget_usd is not None else plan.get("budget_usd", 4.0))
    missing = sorted({r["model"] for r in plan["runs"] if price_of(prices, r["model"]) is None})
    if missing:
        raise SystemExit(f"No price for {', '.join(missing)}: add input and output prices per million tokens "
                         "to the prices file (see docs/mac-and-github.md).")
    results = []
    for run in plan["runs"]:
        prefix = run_prefix(out_dir, run)
        if prefix.with_suffix(".json").exists():
            echo(f"[openai-plan] done already: {prefix.name}")
            continue
        remaining = budget - spent_usd(out_dir, prices)
        p_in, p_out = price_of(prices, run["model"])
        cap = int(remaining / max(p_in, p_out) * 1e6)
        if cap <= 1000:
            echo(f"[openai-plan] budget exhausted (${remaining:.2f} left); stopping before {prefix.name}")
            break
        client = OpenAIClient(cache_dir=out_dir / "cache", max_total_tokens=cap, echo=echo, **(client_kwargs or {}))
        policy = _policy(run, client, echo)
        echo(f"[openai-plan] {prefix.name}: up to {cap} new tokens (${remaining:.2f} left)")
        stop = False
        try:
            summary = evaluate(policy, prefix, {"label": prefix.name, "kind": "openai", **run,
                                                 "price_per_mtok": {"input": p_in, "output": p_out}}, echo=echo)
            results.append(summary)
        except UnsupportedParameter as exc:
            echo(f"[openai-plan] skipped {prefix.name}: {exc}. Use a different mode for this model.")
            (out_dir / f"{prefix.name}.skipped.txt").write_text(str(exc) + "\n")
        except BudgetExceeded as exc:
            echo(f"[openai-plan] {exc}")
            stop = True
        finally:
            record_spend(out_dir, run["model"], client.usage.as_dict())
        if stop:
            break
    echo(f"[openai-plan] spent so far: ${spent_usd(out_dir, prices):.2f} of ${budget:.2f}")
    return results
