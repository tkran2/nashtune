# Mac runs and GitHub publishing

Everything in sections 1 and 2 of the memo was produced in the cloud sandbox.
Sections 3 to 5 need a language model and are run on your Mac. Every step
below skips work that is already finished and resumes interrupted training, so
if anything stops (sleep, Ctrl-C, a crash), run the same command again.

Times are estimates for a 48 GB Apple-silicon Mac and have not been measured;
step 2 measures your machine, and the training step sizes itself to a time
budget from that measurement.

## 1. Set up (10 minutes)

Download `nashtune.zip` into Downloads. In Terminal:

```bash
mkdir -p ~/Projects
unzip ~/Downloads/nashtune.zip -d ~/Projects
cd ~/Projects/nashtune
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[hf]"
make test          # about 1 minute, NumPy only
make test-torch    # about 1 minute: the model code on a tiny random model, no downloads
```

If `python3.12` is missing: `brew install python@3.12`. Each new Terminal
window needs `cd ~/Projects/nashtune && source .venv/bin/activate` first.

Keep the Mac awake and plugged in for the long steps:

```bash
caffeinate -dims &
```

## 2. Check speed (2 minutes plus a 1 GB download)

```bash
make mac-check
```

This loads Qwen2.5-0.5B-Instruct, scores 12 Kuhn prompts and times one
training step. It prints `seconds_per_infoset_scored` and
`seconds_per_train_step`. If it fails with an MPS error, rerun with
`PYTORCH_ENABLE_MPS_FALLBACK=1 make mac-check` and tell me what it printed.

## 3. Experiment B: base models (about 30 minutes plus about 19 GB of downloads)

```bash
make mac-base
```

Scores every information set of Kuhn and Leduc with prompt templates A and B,
and leduc4 with template A, for Qwen2.5-0.5B, 1.5B and 7B-Instruct. The 7B model
runs in bfloat16 to halve its memory (about 15 GB); if it reports non-finite
probabilities, rerun that model with `--dtype float32` (about 30 GB). Results go to
`results/mac/base/`. No API cost.

The same thing one model and game at a time:

```bash
nashtune eval-hf --model Qwen/Qwen2.5-1.5B-Instruct --game leduc --template A
nashtune eval-hf --model Qwen/Qwen2.5-1.5B-Instruct --game leduc --template B
nashtune eval-hf --model Qwen/Qwen2.5-7B-Instruct --game kuhn --template A --dtype bfloat16
```

Any Hugging Face model id works with `--model`.

## 4. Experiment C: GRPO and the supervised baseline (about 4 hours in total)

First a short end-to-end run on Kuhn with the 0.5B model (about 10 minutes):

```bash
make mac-smoke
```

It prints exploitability every 5 steps. It should fall; if it rises steadily,
stop and send me `logs/mac-smoke.log`.

Then the main run (about 100 minutes of training plus 10 minutes of
evaluation):

```bash
make mac-train
```

It trains LoRA adapters on Qwen2.5-1.5B-Instruct on Leduc with exact
best-response rewards: 32 information sets per step, 8 sampled answers per
set, best responses recomputed every 10 steps, a checkpoint every 10 steps.
Before training it measures one step and picks the number of steps that fits
`TRAIN_MINUTES` (100 by default); it prints the choice and an ETA every 10 steps.
For a shorter run: `make mac-train TRAIN_MINUTES=60`. After training it measures
the adapter on Leduc with templates A and B and on leduc4.

Then the supervised baseline, with the same number of steps (about the same time):

```bash
make mac-sft
```

The underlying commands, if you want to run one piece by hand:

```bash
nashtune mac train --minutes 100          # or --steps 150 for a fixed count
nashtune mac sft
nashtune eval-hf --model Qwen/Qwen2.5-1.5B-Instruct --adapter runs/grpo_leduc_Qwen2.5-1.5B-Instruct/policy/adapter --game leduc4 --template B --out results/mac/train --label grpo_leduc_Qwen2.5-1.5B-Instruct
nashtune train --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --game kuhn --reward br --steps 30 --out runs/my_run
```

If memory runs short (the Mac becomes very slow or the run dies), add
`--batch-rows 8` to `nashtune mac train` / `nashtune mac sft` and rerun; the
run resumes from its last checkpoint.

## 5. Experiment D: OpenAI models (under $4)

```bash
export OPENAI_API_KEY=sk-...        # your key; it is never written to disk
nashtune openai-models              # the models your key can use
```

Pick two names from that list:

- a chat model that returns log-probabilities (non-reasoning; the `gpt-4.1` and
  `gpt-4o` families did when this was written), for `OPENAI_CHAT_MODEL`;
- a small reasoning model, for `OPENAI_REASONING_MODEL`.

`configs/openai_prices.json` holds input and output prices in dollars per
million tokens for gpt-4.1-mini and gpt-5-mini, the two models used for the
published results (as of 2026-10-16). For other models, add their names and
prices from https://openai.com/api/pricing/, and check that the existing prices
are still current. Then look at the plan and its cost before anything is sent:

```bash
make mac-openai-dry OPENAI_CHAT_MODEL=name-one OPENAI_REASONING_MODEL=name-two
```

The `$ likely` column is the estimate to trust; `$ max` assumes every reasoning
answer uses its full 2,000-token allowance. Then run it:

```bash
make mac-openai OPENAI_CHAT_MODEL=name-one OPENAI_REASONING_MODEL=name-two
```

Spending is recorded in `results/mac/openai/spend.json`, and each run is capped
so that the total cannot pass $4 (every token is counted at the model's output
price). If the budget runs out, the plan stops; runs it did not finish show as
pending in the memo. If the chat model turns out not to return log-probabilities, its
`logprobs` runs are skipped with a message and the other modes still run.
Answers are cached in `results/mac/openai/cache/`, so rerunning costs nothing.

One run by hand, with a cost estimate first:

```bash
nashtune eval-openai --model name-one --mode logprobs --game leduc --template A --price-in 0.40 --price-out 1.60 --dry-run
nashtune eval-openai --model name-one --mode logprobs --game leduc --template A --max-total-tokens 150000
```

## 6. Build the memo and send the results back

```bash
make report        # rewrites reports/NashTune_Research_Memo.md and figures from results/
make bundle        # writes nashtune_results_<date>.zip (results/ and logs/, no model weights)
```

Send me the zip. Or run everything after setup in one go (about 5 hours,
resumable):

```bash
make mac-all OPENAI_CHAT_MODEL=name-one OPENAI_REASONING_MODEL=name-two
```

## Publish to GitHub

Install and sign in to the GitHub CLI once:

```bash
brew install gh
gh auth login --scopes workflow
gh auth setup-git
```

Choose GitHub.com and HTTPS. The `workflow` scope lets you push the CI
workflow in `.github/workflows/`.

```bash
cd ~/Projects/nashtune
git init -b main
git add .
git -c user.name='Tom Krane' -c user.email='tkran2@users.noreply.github.com' commit -m 'Measure LLM poker strategies exactly and train them with exact rewards'
gh repo create nashtune --public --source=. --remote=origin --push
gh repo view --json url --jq .url
```

`.gitignore` keeps `runs/` (checkpoints, hundreds of MB), `logs/`, the API cache
and result bundles out of the repository; `results/` and `reports/` are committed.
CI runs the tests on Ubuntu and macOS with Python 3.12, including the torch tests
on a tiny model.

After the Mac results come back and the memo is rebuilt:

```bash
make test
git add results reports README.md docs
git commit -m 'Add language-model results'
git push
```
