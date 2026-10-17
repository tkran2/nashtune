# NashTune

Exact evaluation and exact-reward fine-tuning of language models in small
poker games.

Most evaluations of language models are noisy judge scores. In small poker
games the quality of any strategy can be measured exactly: its
exploitability, the amount a perfect opponent wins from it per hand. NashTune
turns a language model into a complete poker strategy by reading its
probability for every legal action at every decision point, measures that
strategy's exploitability exactly, and fine-tunes an open model with GRPO
using rewards computed exactly from the game tree. It then checks whether the
improvement survives a reworded prompt and an unseen variant of the game.

**Findings** (full write-up: [`reports/NashTune_Research_Memo.md`](reports/NashTune_Research_Memo.md)).
The language-model results were measured on an Apple M4 Pro (Qwen2.5 with
LoRA in PyTorch on MPS, and the OpenAI API for $1.50); the tabular study runs
anywhere in NumPy.

- **Imitating the exact equilibrium beat every API model, but only on the
  prompt it was trained with.** LoRA fine-tuning of Qwen2.5-1.5B-Instruct on
  the CFR+ strategy took Leduc exploitability from 2,289 to **339 mA** (−85%),
  below gpt-5-mini (710 mA, sampled) and gpt-4.1-mini (1,576 mA read from
  log-probabilities, 1,222 when asked to state a mixed strategy). Under a
  reworded prompt the same model scored 1,507 mA (only −17% from its base),
  and on the unseen four-rank deck 917 mA (−64%).
- **GRPO with exact rewards collapsed on the language model.** It cut
  exploitability from 2,289 to 1,196 mA in 10 steps, then turned
  deterministic: mean policy entropy fell from 0.60 to 0.0003 nats per
  decision by step 30. From then on all 8 sampled answers to every question
  were identical, so every group had zero reward variance: 296 of 310 steps
  produced no gradient and the policy stayed at 1,467 mA. The tabular study
  below, with one parameter per decision, never collapsed.
- **Base models are poor and prompt-sensitive.** On Leduc, Qwen2.5-0.5B was no
  better than uniform random play (2,389 against 2,374 mA), 1.5B was at 2,289
  and 7B at 1,245; rewording the prompt moved these by up to 1.5x, in both
  directions. On Kuhn every model beat uniform play (458 mA): 7B 183,
  gpt-5-mini 102.

![Exploitability of each model](reports/figures/models.png)

The tabular study (one parameter per decision, many seeds) is where the
reward design was worked out:

- **The reward decides whether GRPO learns to play well.** Rewarding each
  decision with its exact value against the opponent's best response took a
  randomly initialized Leduc policy from 2,671 to **48 mA** per hand (median
  of 5 seeds, mean over the last 10% of 3,000 steps). Rewarding it with the
  outcome of self-play hands, the obvious verifiable reward, ended at
  **1,387 mA**, 29 times higher. On Kuhn poker: 1.9 against 176 mA.
  (`results/dynamics/final.csv`)
- **Self-play does not settle.** On Leduc its best point was 923 mA but it
  ended at 1,329 mA (medians); 3 of 5 runs ended more than 30% above their best
  point. More samples per step did not help it: group sizes 2, 4, 8 and 16 gave
  1,628, 1,935, 1,970 and 2,258 mA.
- **Standard GRPO normalization costs accuracy near the equilibrium.**
  Dividing advantages by the group's standard deviation left the exact-reward
  policy at 49.7 mA on Leduc after 2,000 steps; without it, 27.4. Plain gradient
  descent on exploitability (exact expectation, reach weighting, no
  normalization) reached 15.4. CFR+, a dedicated solver, was at 0.08 mA after
  the same 2,000 full-tree passes.
- **The measurement is exact.** NashTune's game trees, best responses and
  exploitability match ZeroSum, an independent C++ implementation, on 14
  strategies across Kuhn, Leduc, leduc4 and leduc5 to the 10 digits ZeroSum
  prints (largest relative difference 3.9e-10). (`tests/fixtures/zerosum/`)
- **The study is reproducible.** It is seeded throughout; after an interruption
  it was resumed, and five runs recomputed with the final code matched the stored
  curves exactly (`results/dynamics/reproducibility.json`).

![Exploitability during training](reports/figures/dynamics_main.png)

![Language-model training: exploitability and entropy](reports/figures/training.png)

## Reproducing the language-model results

These need a language model. Exact commands, setup and troubleshooting are in
[`docs/mac-and-github.md`](docs/mac-and-github.md). Measured on an Apple M4 Pro
with 48 GB: GRPO training 34 minutes (310 steps), the supervised baseline 84
minutes, the OpenAI runs about 2.5 hours (most of it gpt-5-mini reasoning on
Leduc) for $1.50. The times in the table were estimates made before the run.

| Step | Command | What it measures | Time | Cost |
| --- | --- | --- | --- | --- |
| B | `make mac-base` | Exploitability of Qwen2.5-0.5B, 1.5B and 7B-Instruct on Kuhn and Leduc, prompt templates A and B, plus leduc4 | about 30 min plus 19 GB of downloads | none |
| C | `make mac-smoke` | A 30-step GRPO run on Kuhn with the 0.5B model, end to end | about 10 min | none |
| C | `make mac-train` | GRPO (exact best-response rewards) with LoRA on Qwen2.5-1.5B-Instruct on Leduc, then Leduc with templates A and B and the unseen leduc4 | about 100 min training plus 10 min evaluation | none |
| C | `make mac-sft` | The supervised baseline: the same model and steps, imitating the CFR+ equilibrium | about the same | none |
| D | `make mac-openai OPENAI_CHAT_MODEL=… OPENAI_REASONING_MODEL=…` | OpenAI models, read through log-probabilities, sampled answers, or a stated mixed strategy | 20 to 60 min | under $4 by construction |

`make mac-all` runs them in order; every step resumes where it stopped. Then
`make report` rebuilds the memo and `make bundle` zips the results to send back.

## Run it

Python 3.10 or later. The core needs NumPy and Matplotlib.

```bash
python3 -m pip install -e .
make test     # 93 tests, under a minute; 10 need torch and 1 the ZeroSum binary, and skip without them
nashtune info leduc
nashtune prompts leduc --infoset 0:K:J:kbc/kb --template A
nashtune train --game leduc --reward br --steps 300       # tabular GRPO, 10 seconds
nashtune train --game leduc --reward selfplay --steps 300
nashtune dynamics --jobs 2                  # the whole study, about an hour on two cores
nashtune report                             # rebuild the memo and figures
```

`nashtune train` prints exploitability as it goes and writes `log.csv`,
`summary.json` and the final strategy to its run directory.

What a model is asked, for one Leduc information set (template A):

```text
You are the first player. Your card: King. Public card: Jack.
Round 1: You checked. Opponent bet 2. You called.
Round 2: You checked. Opponent bet 4.
Pot: 10 chips. To call: 4 chips.
Legal actions: fold, call, raise.
Your action?
```

The model's strategy at this point is its probability of answering `fold`,
`call` or `raise` (any capitalization), renormalized over the three. Doing this
for all 288 information sets gives a complete strategy, and its exploitability
follows exactly.

## What is implemented

| Piece | File | Notes |
| --- | --- | --- |
| Games | `nashtune/games/poker.py`, `game.py` | Kuhn, Leduc and leducN built exactly as ZeroSum builds them (same node numbering, information-set keys and order) |
| Exact analysis | `nashtune/games/analysis.py` | Expected value, reach, best response, exploitability per seat, counterfactual action values and reach; vectorized by tree depth |
| Solver | `nashtune/games/cfr.py` | CFR+ (reference equilibrium, SFT target, optimal baseline) |
| Strategy files | `nashtune/games/strategy_io.py` | ZeroSum's format, `infoset action probability`; `zerosum eval` reads them |
| Prompts | `nashtune/prompts.py` | Template A (training) and template B (held out: different wording, card letters, different order), plus a JSON-answer variant |
| Rewards | `nashtune/rewards.py` | `br`: Q(I, a) against the best response (exploitability descent); `selfplay`: sampled hands, payoff to the acting player |
| GRPO math | `nashtune/grpo.py` | Group advantages, clipped surrogate (ε = 0.2), exact KL to the step-0 policy, cross-entropy for SFT; analytic gradients |
| Trainer | `nashtune/train.py` | Backend-agnostic; reward modes `br`, `exact`, `selfplay`, `sft`; infoset subsets, stale-reward refresh, checkpoints and exact resume |
| Tabular policy | `nashtune/policy/tabular.py` | Softmax table with Adam, gradient-checked |
| Hugging Face policy | `nashtune/policy/hf.py`, `scoring.py`, `torch_loss.py` | Exact answer probabilities with left-padded batching; LoRA training with the same loss in torch |
| OpenAI policy | `nashtune/policy/openai_backend.py` | `logprobs`, `sample` and `stated` modes; disk cache, token accounting, budget guard, retry on rejected parameters |
| Experiments | `nashtune/experiments/` | `dynamics.py` (A), `mac.py` (B, C), `openai_plan.py` (D) |
| Memo | `nashtune/report.py` | Builds the memo and figures from `results/`; missing experiments are marked pending |

## Why the answers can be trusted

- **An independent implementation.** ZeroSum (C++, a separate project) computed
  the value and both best-response payoffs of 14 fixed strategies; NashTune
  reproduces all of them, and the information-set order, to ZeroSum's printed
  precision. With `ZEROSUM_BIN` set, a test also runs the binary live.
- **Known answers.** Every member of Kuhn's hand-derived family of equilibria has
  exploitability zero (below 1e-12) and value −1/18.
- **Brute force.** On Kuhn the best response is checked against all 64 pure
  strategies of the responder, for several random opponents and the tie-heavy
  uniform one.
- **The rewards are checked as derivatives.** Perturbing the policy at one
  information set changes the player's value by exactly the amount the
  counterfactual values predict, and against the best response the same values
  give the derivative of the worst-case value (Danskin's theorem), checked by
  finite differences.
- **The loss is checked three ways.** The grouped clipped surrogate equals the
  per-sample textbook form; its gradient matches finite differences; the torch
  version matches the NumPy version (in CI).
- **The language-model plumbing is tested without a language model.** Left
  padding, multi-token answers and the "last k positions" bookkeeping are checked
  against a NumPy stand-in model; the torch path is checked in CI against
  unpadded one-sequence-at-a-time scoring with a tiny random Qwen2 model.
- **The OpenAI client is tested against a local fake server** for all three
  modes, caching, the token budget, rate-limit retries, and models that reject
  `max_tokens`, `temperature`, `logprobs` or `n`.
- **Resume is exact.** An interrupted and resumed tabular training run is
  bit-identical to an uninterrupted one. The torch backend has the same test
  (tolerance 1e-6), which runs in CI.

## Limits

- Research-sized games. Exact best responses need the whole game tree, which is
  what makes the rewards verifiable and what keeps the method from scaling as is.
- A strategy read from answer probabilities is one definition of a model's
  play. A model that reasons before answering may play differently; the
  `sample` mode measures that, at the price of sampling noise, which itself
  raises measured exploitability.
- The tabular study uses one optimizer and fixed hyperparameters per sweep.
  A language model shares parameters across information sets, so it can learn
  faster or slower than a table; the tabular numbers set expectations only.
- Each language-model training result is a single run with one setting
  (LoRA rank 16, learning rate 1e-4, no KL penalty, 8 samples per question).
  The GRPO collapse is a property of that setting; it has not been rerun with
  the obvious remedies (keeping the best checkpoint, a lower learning rate,
  unnormalized advantages, a KL penalty to the base model).
- Sampled OpenAI results use 4 answers per decision, so they include sampling
  noise, which raises measured exploitability.

## Layout

```text
nashtune/          games/ (trees, analysis, CFR+, strategy files), prompts, rewards, grpo, train,
                   policy/ (tabular, hf, openai), experiments/, report, cli
tests/             unit and end-to-end tests; fixtures/zerosum/ (the ZeroSum cross-check)
results/dynamics/  experiment A: curves.csv, final.csv, manifest.json
results/mac/       experiments B, C, D (Apple M4 Pro)
reports/           the memo and its figures
configs/           the OpenAI plan and prices file
docs/              interview guide, Mac and GitHub instructions
scripts/           make_zerosum_fixture.py
```

MIT license.
