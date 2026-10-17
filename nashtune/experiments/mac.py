"""Experiments B and C: the runs that need a language model on Tom's Mac.

Each step skips work whose result file already exists, and training resumes
from its last checkpoint, so any step can be interrupted and rerun.

* ``base``: exploitability of base models (hf backend), Kuhn and Leduc with
  templates A and B, and leduc4 with template A (for the transfer comparison).
* ``smoke``: a short GRPO run on Kuhn with the smallest model, to check the
  whole training path end to end in a few minutes.
* ``train``: the main GRPO (``br`` rewards) LoRA run on Leduc, then the
  trained adapter on Leduc (templates A and B) and on leduc4 (unseen game).
* ``sft``: the supervised baseline: the same model, steps and infosets per step,
  trained to imitate the CFR+ equilibrium; evaluated the same way.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path

from ..evaluate import evaluate, slug
from ..games import make_game
from ..train import TrainConfig, Trainer

BASE_MODELS = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct")
TRAIN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_EVALS = (("kuhn", "A"), ("kuhn", "B"), ("leduc", "A"), ("leduc", "B"), ("leduc4", "A"))
TRAINED_EVALS = (("leduc", "A"), ("leduc", "B"), ("leduc4", "A"))


def default_dtype(model_id: str, dtype: str) -> str:
    """bfloat16 for models of 3B parameters or more (by name), to halve memory
    (bfloat16 rather than float16, which can overflow in large models);
    float32 otherwise. Override with --dtype."""
    if dtype != "auto":
        return dtype
    m = re.search(r"(\d+(?:\.\d+)?)[Bb]\b", model_id)
    return "bfloat16" if m and float(m.group(1)) >= 3 else "auto"


def eval_prefix(out: Path, label: str, game: str, template: str) -> Path:
    return out / f"{slug(label)}_{game}_{template}"


def run_evals(model_id: str, label: str, evals, out: Path, adapter: str | None = None, device: str = "auto",
              dtype: str = "auto", batch_rows: int = 16, kind: str = "hf-base", echo=print) -> list[dict]:
    from ..policy.hf import HFPolicy, free_memory, load_model

    todo = [(g, t) for g, t in evals if not eval_prefix(out, label, g, t).with_suffix(".json").exists()]
    if not todo:
        echo(f"[mac] {label}: evaluations already done")
        return []
    dtype = default_dtype(model_id, dtype)
    model, tokenizer = load_model(model_id, device, dtype, adapter=adapter, echo=echo)
    results = []
    for game, template in todo:
        policy = HFPolicy(make_game(game), model_id, template=template, device=device, dtype=dtype,
                          batch_rows=batch_rows, model=model, tokenizer=tokenizer, echo=echo)
        policy.adapter = adapter
        results.append(evaluate(policy, eval_prefix(out, label, game, template),
                                {"label": label, "kind": kind, "template": template}, echo=echo))
    del model
    free_memory()
    return results


def step_base(models=BASE_MODELS, out: str = "results/mac/base", device: str = "auto", dtype: str = "auto",
              batch_rows: int = 16, echo=print) -> None:
    failed = []
    for model_id in models:
        try:
            run_evals(model_id, model_id, BASE_EVALS, Path(out), device=device, dtype=dtype, batch_rows=batch_rows,
                      echo=echo)
        except (RuntimeError, OSError, ValueError) as exc:  # keep going with the other models
            echo(f"[mac] {model_id} failed: {exc}")
            failed.append(model_id)
    if failed:
        raise SystemExit(f"base evaluations failed for {', '.join(failed)}; see the messages above")


def choose_steps(policy, cfg: TrainConfig, minutes: float, echo=print) -> int:
    from ..policy.hf import bench

    k = cfg.infosets_per_step or policy.game.num_infosets
    b = bench(policy, n_infosets=k, group_size=cfg.group_size, echo=echo)
    per_step = b["seconds_per_train_step"] + b["seconds_per_infoset_scored"] * k
    per_refresh = b["seconds_full_extraction_estimate"]
    per = per_step + per_refresh / cfg.refresh_every
    steps = int(minutes * 60 / per)
    steps = max(cfg.refresh_every, min(1000, steps - steps % cfg.refresh_every))
    echo(f"[mac] measured {per_step:.1f} s per step and {per_refresh:.0f} s per full extraction; "
         f"{steps} steps fit in {minutes:.0f} minutes")
    return steps


def train_run(name: str, model_id: str, cfg: TrainConfig, minutes: float | None, runs: str = "runs",
              device: str = "auto", dtype: str = "auto", batch_rows: int = 16, lr: float = 1e-4, lora_r: int = 16,
              lora_alpha: int = 32, steps_like: str | None = None, echo=print) -> Path:
    """Train (or resume, or skip if finished). Returns the run directory."""
    from ..policy.hf import HFPolicy, free_memory

    out = Path(runs) / name
    if (out / "summary.json").exists():
        echo(f"[mac] {name}: training already finished")
        return out
    saved = out / "config.json"
    policy = HFPolicy(make_game(cfg.game), model_id, template="A", device=device, dtype=dtype,
                      batch_rows=batch_rows, train=True, lr=lr, lora_r=lora_r, lora_alpha=lora_alpha,
                      seed=cfg.seed, echo=echo)
    if saved.exists():
        cfg.steps = int(json.loads(saved.read_text())["train"]["steps"])
        echo(f"[mac] {name}: resuming a run of {cfg.steps} steps")
    elif steps_like and (Path(runs) / steps_like / "config.json").exists():
        cfg.steps = int(json.loads((Path(runs) / steps_like / "config.json").read_text())["train"]["steps"])
        echo(f"[mac] {name}: using the same number of steps as {steps_like}: {cfg.steps}")
    elif cfg.steps <= 0:
        cfg.steps = choose_steps(policy, cfg, minutes or 100.0, echo=echo)
    trainer = Trainer(policy, cfg, out_dir=out, resume=True, echo=echo)
    trainer.run()
    del trainer, policy
    free_memory()
    return out


def copy_logs(run: Path, results: str = "results/mac/train") -> None:
    """Keep the training log, config and summary with the results (they are
    small; the run directory also holds checkpoints and is not bundled)."""
    import shutil

    dest = Path(results) / "logs" / run.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("log.csv", "summary.json", "config.json"):
        if (run / name).exists():
            shutil.copy2(run / name, dest / name)


def step_smoke(model_id: str = SMOKE_MODEL, runs: str = "runs", out: str = "results/mac/train", device: str = "auto",
               dtype: str = "auto", echo=print) -> None:
    name = f"smoke_grpo_kuhn_{slug(model_id.split('/')[-1])}"
    cfg = TrainConfig(game="kuhn", reward="br", steps=30, group_size=8, infosets_per_step=0, refresh_every=5,
                      checkpoint_every=10, print_every=5, seed=0)
    run = train_run(name, model_id, cfg, None, runs=runs, device=device, dtype=dtype, echo=echo)
    copy_logs(run, out)
    run_evals(model_id, name, (("kuhn", "A"), ("kuhn", "B")), Path(out), adapter=str(run / "policy" / "adapter"),
              device=device, dtype=dtype, kind="hf-grpo-smoke", echo=echo)


def step_train(kind: str = "grpo", model_id: str = TRAIN_MODEL, minutes: float = 100.0, steps: int = 0,
               infosets_per_step: int = 32, group_size: int = 8, refresh_every: int = 10, lr: float = 1e-4,
               kl_coef: float = 0.0, runs: str = "runs", out: str = "results/mac/train", device: str = "auto",
               dtype: str = "auto", batch_rows: int = 16, seed: int = 0, echo=print) -> None:
    short = slug(model_id.split("/")[-1])
    name = f"{kind}_leduc_{short}"
    cfg = TrainConfig(game="leduc", reward="br" if kind == "grpo" else "sft", steps=steps, group_size=group_size,
                      infosets_per_step=infosets_per_step, refresh_every=refresh_every, kl_coef=kl_coef,
                      checkpoint_every=refresh_every, print_every=refresh_every, seed=seed)
    start = time.time()
    run = train_run(name, model_id, cfg, minutes, runs=runs, device=device, dtype=dtype, batch_rows=batch_rows,
                    lr=lr, steps_like=f"grpo_leduc_{short}" if kind == "sft" and steps <= 0 else None, echo=echo)
    copy_logs(run, out)
    run_evals(model_id, name, TRAINED_EVALS, Path(out), adapter=str(run / "policy" / "adapter"), device=device,
              dtype=dtype, batch_rows=batch_rows, kind=f"hf-{kind}", echo=echo)
    echo(f"[mac] {name}: done in {(time.time() - start) / 60:.1f} min; settings {json.dumps(asdict(cfg))}")
