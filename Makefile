# NashTune. `make test` needs only NumPy; the mac-* targets need the hf extras
# (see docs/mac-and-github.md). Every mac-* step skips finished work and
# resumes interrupted training, so rerunning a target is always safe.

SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
PY ?= python3
NASHTUNE ?= $(PY) -m nashtune
# Experiment D: real model names from `nashtune openai-models`.
OPENAI_CHAT_MODEL ?=
OPENAI_REASONING_MODEL ?=
TRAIN_MINUTES ?= 100

.PHONY: test test-torch dynamics report bundle mac-check mac-base mac-smoke mac-train mac-sft mac-openai-dry mac-openai mac-all clean

test:
	$(PY) -m unittest discover -s tests -v

# Needs torch, transformers and peft installed.
test-torch:
	NASHTUNE_REQUIRE_TORCH=1 $(PY) -m unittest discover -s tests -p "test_torch_backend.py" -v

# Experiment A (tabular dynamics study), about an hour on two cores. Resumable.
dynamics:
	$(NASHTUNE) dynamics --jobs 2

report:
	$(NASHTUNE) report

bundle:
	$(NASHTUNE) bundle

logs:
	mkdir -p logs

# A one-minute check that the model backend works on this machine.
mac-check: | logs
	$(NASHTUNE) bench-hf --model Qwen/Qwen2.5-0.5B-Instruct --game kuhn --infosets 12 --train 2>&1 | tee logs/mac-check.log

# Experiment B: base models, Kuhn and Leduc, templates A and B, plus leduc4.
mac-base: | logs
	$(NASHTUNE) mac base 2>&1 | tee -a logs/mac-base.log

# Experiment C: a short end-to-end GRPO run on Kuhn first, then the main run and the SFT baseline.
mac-smoke: | logs
	$(NASHTUNE) mac smoke 2>&1 | tee -a logs/mac-smoke.log

mac-train: | logs
	$(NASHTUNE) mac train --minutes $(TRAIN_MINUTES) 2>&1 | tee -a logs/mac-train.log

mac-sft: | logs
	$(NASHTUNE) mac sft --minutes $(TRAIN_MINUTES) 2>&1 | tee -a logs/mac-sft.log

# Experiment D: OpenAI models, under the budget in configs/openai_plan.json.
mac-openai-dry:
	$(NASHTUNE) openai-plan --chat-model "$(OPENAI_CHAT_MODEL)" --reasoning-model "$(OPENAI_REASONING_MODEL)" --dry-run

mac-openai: | logs
	$(NASHTUNE) openai-plan --chat-model "$(OPENAI_CHAT_MODEL)" --reasoning-model "$(OPENAI_REASONING_MODEL)" --yes 2>&1 | tee -a logs/mac-openai.log

# Each step runs even if an earlier one failed (its errors are in logs/); rerun to retry.
mac-all:
	-$(MAKE) mac-base
	-$(MAKE) mac-smoke
	-$(MAKE) mac-train
	-$(MAKE) mac-sft
	-$(MAKE) mac-openai
	$(MAKE) report
	$(MAKE) bundle

clean:
	rm -rf build dist *.egg-info
