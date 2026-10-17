"""The ``nashtune`` command line. Run ``nashtune --help`` or ``nashtune COMMAND --help``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _game_arg(p: argparse.ArgumentParser, default: str | None = None) -> None:
    if default is None:
        p.add_argument("game", help="kuhn, leduc, leduc4, leduc5, ...")
    else:
        p.add_argument("--game", default=default, help=f"kuhn, leduc, leduc4, leduc5 (default {default})")


def _hf_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto", help="auto (cuda, then mps, then cpu), mps, cuda or cpu")
    p.add_argument("--dtype", default="auto", help="auto (float32 on mps/cpu), float32, float16, bfloat16")
    p.add_argument("--batch-rows", type=int, default=16, help="sequences per forward pass (lower it if memory runs out)")
    p.add_argument("--surface", default="both", choices=["both", "lower"],
                   help="count 'call' and 'Call' (both) or only 'call' (lower) as the answer")


# ------------------------------------------------------------------ commands
def cmd_info(a) -> int:
    from .games import exploitability, game_value, make_game

    g = make_game(a.game)
    c = g.counts()
    print(f"game {g.name}  ranks {' '.join(g.rank_names)}")
    print(f"nodes {c['nodes']} (chance {c['chance']}, decision {c['decision']}, terminal {c['terminal']})")
    print(f"information_sets {c['infosets']} (player 0: {c['infosets_player0']}, player 1: {c['infosets_player1']})")
    print(f"actions {c['actions']}")
    print(f"value_to_first_player {game_value(g.name):.12f}")
    print(f"uniform_strategy_exploitability_ma {exploitability(g, g.uniform_strategy()).milli:.4f}")
    return 0


def cmd_solve(a) -> int:
    from .games import CFRPlus, exploitability, make_game, write_strategy

    g = make_game(a.game)
    solver = CFRPlus(g)
    every = a.every or a.iterations
    while solver.iteration < a.iterations:
        solver.run(min(every, a.iterations - solver.iteration))
        e = exploitability(g, solver.average_strategy())
        print(f"iteration {solver.iteration:7d}  exploitability {e.milli:10.4f} mA/hand  "
              f"value in [{e.value_lower:.6f}, {e.value_upper:.6f}]")
    if a.save:
        write_strategy(g, solver.average_strategy(), a.save)
        print(f"saved {a.save}")
    return 0


def cmd_eval(a) -> int:
    from .games import exploitability, game_value, make_game, read_strategy

    g = make_game(a.game)
    e = exploitability(g, read_strategy(g, a.strategy))
    seat0, seat1 = e.seat_milli(game_value(g.name))
    print(f"value_to_first_player {e.value:.10g}")
    print(f"best_response_vs_second_player {e.best_vs_p1:.10g}")
    print(f"best_response_vs_first_player {e.best_vs_p0:.10g}")
    print(f"exploitability_ma_per_hand {e.milli:.10g}")
    print(f"seat0_exploitability_ma {seat0:.10g}")
    print(f"seat1_exploitability_ma {seat1:.10g}")
    print(f"game_value_between {e.value_lower:.10g} {e.value_upper:.10g}")
    return 0


def cmd_prompts(a) -> int:
    from .games import make_game
    from .prompts import render

    g = make_game(a.game)
    infos = [g.infoset_by_key(a.infoset)] if a.infoset else g.infosets[: a.limit]
    for i, info in enumerate(infos):
        p = render(g, info, a.template, answer=a.answer)
        if i == 0:
            print(f"[system]\n{p.system}\n")
        print(f"[user: {info.key}]\n{p.user}\n")
    return 0


def cmd_eval_hf(a) -> int:
    from .evaluate import evaluate, slug
    from .experiments.mac import default_dtype
    from .games import make_game
    from .policy.hf import HFPolicy

    label = a.label or (a.model + (f"+{Path(a.adapter).parent.name}" if a.adapter else ""))
    prefix = Path(a.out) / f"{slug(label)}_{a.game}_{a.template}"
    if prefix.with_suffix(".json").exists() and not a.force:
        print(f"{prefix}.json exists (use --force to redo)")
        return 0
    policy = HFPolicy(make_game(a.game), a.model, template=a.template, adapter=a.adapter, device=a.device,
                      dtype=default_dtype(a.model, a.dtype), batch_rows=a.batch_rows, surface=a.surface)
    evaluate(policy, prefix, {"label": label, "kind": "hf-adapter" if a.adapter else "hf-base",
                              "template": a.template})
    print(f"wrote {prefix}.json")
    return 0


def cmd_bench_hf(a) -> int:
    from .games import make_game
    from .policy.hf import HFPolicy, bench

    policy = HFPolicy(make_game(a.game), a.model, device=a.device, dtype=a.dtype, batch_rows=a.batch_rows,
                      surface=a.surface, train=a.train)
    bench(policy, n_infosets=a.infosets)
    return 0


def _openai_client(a, max_tokens=None):
    from .policy.openai_backend import OpenAIClient

    return OpenAIClient(cache_dir=a.cache, max_total_tokens=max_tokens)


def cmd_openai_models(a) -> int:
    client = _openai_client(a)
    for name in client.models():
        print(name)
    return 0


def cmd_eval_openai(a) -> int:
    from .evaluate import evaluate, slug
    from .games import make_game
    from .policy.openai_backend import OpenAIPolicy

    client = _openai_client(a, a.max_total_tokens)
    policy = OpenAIPolicy(make_game(a.game), client, a.model, mode=a.mode, template=a.template, samples=a.samples,
                          top_logprobs=a.top_logprobs, max_completion_tokens=a.max_completion_tokens,
                          reasoning_effort=a.reasoning_effort)
    est = policy.plan()
    print(json.dumps(est, indent=1))
    if a.price_in is not None and a.price_out is not None:
        bound = (est["estimated_prompt_tokens"] * a.price_in + est["max_output_tokens"] * a.price_out) / 1e6
        print(f"cost upper bound ${bound:.3f} (input at ${a.price_in}/M, output at ${a.price_out}/M tokens)")
    if a.dry_run:
        return 0
    prefix = Path(a.out) / f"{slug(a.model)}_{a.mode}_{a.game}_{a.template}"
    evaluate(policy, prefix, {"label": prefix.name, "kind": "openai", "model": a.model, "mode": a.mode,
                              "template": a.template})
    print(json.dumps(client.usage.as_dict(), indent=1))
    return 0


def cmd_openai_plan(a) -> int:
    from .experiments.openai_plan import dry_run, load_plan, run_plan

    if not a.chat_model and not a.reasoning_model:
        raise SystemExit("give --chat-model and/or --reasoning-model (names from `nashtune openai-models`)")
    plan = load_plan(a.plan, a.chat_model, a.reasoning_model)
    prices = json.loads(Path(a.prices).read_text()) if a.prices and Path(a.prices).exists() else {}
    prices = {k: v for k, v in prices.items() if not k.startswith("_")}
    if a.budget_usd is not None:
        plan["budget_usd"] = a.budget_usd
    out = Path(a.out)
    estimate = dry_run(plan, prices, out)
    if a.dry_run:
        return 0
    budget = float(plan.get("budget_usd", 4.0))
    if estimate["missing_prices"]:
        raise SystemExit("fill in the prices file first")
    if estimate["bound_usd"] + estimate["spent_usd"] > budget and not a.yes:
        print(f"\nThe worst case (${estimate['bound_usd']:.2f}) is above the remaining budget. The runs stop "
              "before the budget is spent regardless; pass --yes to start.")
        return 1
    run_plan(plan, prices, out, budget)
    return 0


def _train_config(a):
    from .train import TrainConfig

    return TrainConfig(game=a.game, reward=a.reward, steps=a.steps, group_size=a.group_size,
                       infosets_per_step=a.infosets_per_step, deals_per_step=a.deals_per_step,
                       refresh_every=a.refresh_every, updates_per_batch=a.updates_per_batch, clip_eps=a.clip_eps,
                       kl_coef=a.kl_coef, weighting=a.weighting, adv_norm=a.adv_norm, seed=a.seed,
                       checkpoint_every=a.checkpoint_every, sft_target=a.sft_target, print_every=a.print_every)


def cmd_train(a) -> int:
    from .games import make_game
    from .train import Trainer

    hf = a.backend == "hf"
    defaults = {"lr": 1e-4 if hf else 0.05, "infosets_per_step": 32 if hf else 0, "refresh_every": 10 if hf else 1,
                "checkpoint_every": 10 if hf else 0, "steps": 200}
    for k, v in defaults.items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    cfg = _train_config(a)
    game = make_game(a.game)
    if hf:
        from .policy.hf import HFPolicy

        if not a.model:
            raise SystemExit("--model is required with --backend hf")
        policy = HFPolicy(game, a.model, template=a.template, device=a.device, dtype=a.dtype,
                          batch_rows=a.batch_rows, surface=a.surface, train=True, lr=a.lr, lora_r=a.lora_r,
                          lora_alpha=a.lora_alpha, seed=a.seed)
    else:
        from .policy.tabular import TabularPolicy

        policy = TabularPolicy(game, init=a.init, init_scale=a.init_scale, seed=a.seed, lr=a.lr)
    out = a.out or f"runs/{a.reward}_{a.game}_{a.backend}_seed{a.seed}"
    trainer = Trainer(policy, cfg, out_dir=out, resume=not a.no_resume)
    trainer.run()
    print(f"wrote {out}/log.csv, {out}/summary.json, {out}/final_strategy.txt")
    return 0


def cmd_dynamics(a) -> int:
    from .experiments.dynamics import run_all

    run_all(a.out, quick=a.quick, jobs=a.jobs)
    print(f"wrote {a.out}/curves.csv, final.csv, manifest.json")
    return 0


def cmd_mac(a) -> int:
    from .experiments import mac

    common = {"device": a.device, "dtype": a.dtype}
    if a.step in ("base", "all"):
        mac.step_base(models=a.models or mac.BASE_MODELS, batch_rows=a.batch_rows, **common)
    if a.step in ("smoke", "all"):
        mac.step_smoke(model_id=a.smoke_model, **common)
    if a.step in ("train", "all"):
        mac.step_train("grpo", model_id=a.train_model, minutes=a.minutes, steps=a.steps, lr=a.lr,
                       kl_coef=a.kl_coef, batch_rows=a.batch_rows, **common)
    if a.step in ("sft", "all"):
        mac.step_train("sft", model_id=a.train_model, minutes=a.minutes, steps=a.steps, lr=a.lr,
                       batch_rows=a.batch_rows, **common)
    return 0


def cmd_report(a) -> int:
    from .report import build_report

    path = build_report(results=a.results, out=a.out, runs=a.runs)
    print(f"wrote {path}")
    return 0


def cmd_bundle(a) -> int:
    from .report import bundle

    path = bundle(a.out)
    print(f"wrote {path}  (send this file back)")
    return 0


# -------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nashtune", description="Exact exploitability of language-model poker "
                                 "strategies, and GRPO fine-tuning with exact rewards.")
    ap.add_argument("--version", action="version", version=f"nashtune {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="tree size, information sets, game value")
    _game_arg(p)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("solve", help="CFR+ equilibrium")
    _game_arg(p)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--every", type=int, default=0, help="report every N iterations")
    p.add_argument("--save", help="write the average strategy (ZeroSum format)")
    p.set_defaults(func=cmd_solve)

    p = sub.add_parser("eval", help="exploitability and value of a strategy file (same output as `zerosum eval`)")
    _game_arg(p)
    p.add_argument("--strategy", required=True)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("prompts", help="show the prompts a model sees")
    _game_arg(p)
    p.add_argument("--template", default="A", choices=["A", "B"])
    p.add_argument("--infoset", help="one infoset key, e.g. 0:K:J:kbc/kb")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--answer", default="word", choices=["word", "json"])
    p.set_defaults(func=cmd_prompts)

    p = sub.add_parser("eval-hf", help="exploitability of a Hugging Face model (needs torch)")
    p.add_argument("--model", required=True, help="Hugging Face model id, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--adapter", help="a LoRA adapter directory written by `nashtune train`")
    _game_arg(p, "leduc")
    p.add_argument("--template", default="A", choices=["A", "B"])
    p.add_argument("--out", default="results/mac/base")
    p.add_argument("--label", help="name used in the output file names")
    p.add_argument("--force", action="store_true")
    _hf_args(p)
    p.set_defaults(func=cmd_eval_hf)

    p = sub.add_parser("bench-hf", help="measure scoring and training speed on this machine")
    p.add_argument("--model", required=True)
    _game_arg(p, "leduc")
    p.add_argument("--infosets", type=int, default=32)
    p.add_argument("--train", action="store_true", help="also time one LoRA training step")
    _hf_args(p)
    p.set_defaults(func=cmd_bench_hf)

    p = sub.add_parser("openai-models", help="list the models your OPENAI_API_KEY can use")
    p.add_argument("--cache", default="results/mac/openai/cache")
    p.set_defaults(func=cmd_openai_models)

    p = sub.add_parser("eval-openai", help="exploitability of an OpenAI model")
    p.add_argument("--model", required=True, help="a name from `nashtune openai-models`")
    p.add_argument("--mode", required=True, choices=["logprobs", "sample", "stated"])
    _game_arg(p, "kuhn")
    p.add_argument("--template", default="A", choices=["A", "B"])
    p.add_argument("--samples", type=int, default=8, help="answers per infoset in sample mode")
    p.add_argument("--top-logprobs", type=int, default=20)
    p.add_argument("--max-completion-tokens", type=int)
    p.add_argument("--reasoning-effort", help="passed through for reasoning models (dropped if rejected)")
    p.add_argument("--max-total-tokens", type=int, default=200_000, help="stop before using more tokens than this")
    p.add_argument("--price-in", type=float, help="USD per million input tokens (for the cost estimate)")
    p.add_argument("--price-out", type=float, help="USD per million output tokens")
    p.add_argument("--dry-run", action="store_true", help="print the request count and token estimate only")
    p.add_argument("--cache", default="results/mac/openai/cache")
    p.add_argument("--out", default="results/mac/openai")
    p.set_defaults(func=cmd_eval_openai)

    p = sub.add_parser("openai-plan", help="experiment D: every OpenAI run within a dollar budget")
    p.add_argument("--plan", default="configs/openai_plan.json")
    p.add_argument("--prices", default="configs/openai_prices.json")
    p.add_argument("--chat-model", help="a model that returns log-probabilities")
    p.add_argument("--reasoning-model", help="a reasoning model (sampled answers)")
    p.add_argument("--budget-usd", type=float)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--yes", action="store_true", help="start even if the worst case exceeds the budget")
    p.add_argument("--out", default="results/mac/openai")
    p.set_defaults(func=cmd_openai_plan)

    p = sub.add_parser("train", help="GRPO (br, exact, selfplay) or SFT training")
    p.add_argument("--backend", default="tabular", choices=["tabular", "hf"])
    _game_arg(p, "kuhn")
    p.add_argument("--reward", default="br", choices=["br", "exact", "selfplay", "sft"])
    p.add_argument("--steps", type=int)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--infosets-per-step", type=int, help="0 = all (tabular default); hf default 32")
    p.add_argument("--deals-per-step", type=int, default=32)
    p.add_argument("--refresh-every", type=int, help="recompute policy and best response every k steps")
    p.add_argument("--updates-per-batch", type=int, default=1)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--kl-coef", type=float, default=0.0)
    p.add_argument("--weighting", default="uniform", choices=["uniform", "reach", "cfreach"])
    p.add_argument("--adv-norm", default="std", choices=["std", "none"])
    p.add_argument("--sft-target", default="cfr+", help="cfr+ or a strategy file")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint-every", type=int)
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--lr", type=float, help="tabular default 0.05 (Adam on logits); hf default 1e-4 (AdamW on LoRA)")
    p.add_argument("--init", default="random", choices=["random", "uniform"], help="tabular only")
    p.add_argument("--init-scale", type=float, default=1.0, help="tabular only")
    p.add_argument("--model", help="hf only: Hugging Face model id")
    p.add_argument("--template", default="A", choices=["A", "B"], help="hf only")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--out", help="run directory (default runs/<reward>_<game>_<backend>_seed<seed>)")
    p.add_argument("--no-resume", action="store_true")
    _hf_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("dynamics", help="experiment A: reward-design study with the tabular policy")
    p.add_argument("--out", default="results/dynamics")
    p.add_argument("--quick", action="store_true", help="a seconds-long version (for testing)")
    p.add_argument("--jobs", type=int, default=1)
    p.set_defaults(func=cmd_dynamics)

    p = sub.add_parser("mac", help="experiments B and C on a Mac (resumable steps)")
    p.add_argument("step", choices=["base", "smoke", "train", "sft", "all"])
    p.add_argument("--models", nargs="*", help="base step: Hugging Face model ids")
    p.add_argument("--train-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--smoke-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--minutes", type=float, default=100.0, help="time budget for the main training run")
    p.add_argument("--steps", type=int, default=0, help="training steps (0 = fit the time budget)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl-coef", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--batch-rows", type=int, default=16)
    p.set_defaults(func=cmd_mac)

    p = sub.add_parser("report", help="build reports/NashTune_Research_Memo.md and figures from results/")
    p.add_argument("--results", default="results")
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="reports")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("bundle", help="zip results/ and training logs to send back")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_bundle)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted; rerun the same command to resume", file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"nashtune: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
