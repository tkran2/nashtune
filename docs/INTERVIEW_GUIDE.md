# Own the project in an interview

Before you present this: run `make test`, run
`nashtune train --game leduc --reward selfplay --steps 300` and the same with
`--reward br`, and watch the two exploitability columns diverge. Read
`nashtune/rewards.py` and `nashtune/grpo.py` once end to end; they are short
and they are what interviewers will ask about. Know the two language-model
results cold: the supervised run's 339 mA and why it does not transfer, and
why GRPO collapsed (question 18).

## A 45-second explanation

"Most LLM evaluations are judge scores, which are noisy. I wanted a setting
where the quality of a model's behavior can be measured exactly. In small poker
games it can: I read the model's probability of each legal action at every
decision point, which gives a complete strategy, and then compute exactly how
much a perfect opponent would win from it per hand. That number, the
exploitability, is the evaluation, and the same game tree gives exact rewards
for reinforcement learning. Before spending GPU time I studied reward design in
a tabular setting: with the obvious verifiable reward, the outcome of self-play
hands, GRPO cycled and ended 29 times more exploitable than with my reward,
which is each action's exact value against the opponent's best response. That
is gradient descent on exploitability. On a real model, Qwen2.5-1.5B with
LoRA, imitating the exact equilibrium cut Leduc exploitability 85%, below
gpt-5-mini, but the gain mostly vanished under a reworded prompt. GRPO with
the same exact rewards collapsed: within 20 steps the model answered
deterministically, every group of samples agreed, and the gradient was zero
from then on. Finding that took reading the entropy and the skipped-group
counts, not just the score."

## Questions you should be able to answer

**1. What is exploitability, exactly?**

Fix my strategy for both seats. Let the opponent choose the best possible
reply to my seat-1 play, and separately to my seat-2 play. Exploitability is
the average over the two seats of what those replies win beyond the game's
value, in milli-antes per hand (mA, thousandths of the forced bet). It is zero
exactly at a Nash equilibrium. Because the game value appears once with each
sign, the average does not need the value at all:
`(best reply payoff vs my seat-0 play + best reply payoff vs my seat-1 play) / 2`.
The per-seat numbers do need it, and I take it from an exact linear program
(Kuhn −1/18, Leduc −0.085606).

**2. How do you turn a language model into a strategy?**

For each of Leduc's 288 information sets I render a prompt (rules, seat, cards,
betting so far, pot, legal actions) through the model's chat template, ending
where the answer starts. For each legal action word I compute the exact
probability the model's answer starts with that word: the product of
next-token probabilities over its tokens, summed over "call" and "Call". I
renormalize over the legal actions. The unnormalized total is the "legal mass":
how much of its probability the model put on legal answers at all. Nothing is
sampled, so the measured exploitability has no noise.

**3. Why not just sample answers and count?**

Two reasons. Noise: with n samples per decision the frequencies are off by
about 1/sqrt(n), and noise makes a strategy more exploitable, so sampling
biases the measurement upward. Cost: exact scoring is one forward pass per
prompt. Sampling is still needed for API reasoning models that do not expose
log-probabilities, and I report it as an estimate with that bias.

**4. What reward do you train with, and why is it "verifiable"?**

For an information set I of player i and action a: Q(I, a), the expected
payoff of taking a at I and following the current policy afterwards, when the
opponent plays a best response to the current policy, conditioned on reaching
I. It is computed from the game tree: the counterfactual action value
q(I, a) = sum over histories h in I of (chance × opponent reach of h) × (payoff
after h·a), divided by the counterfactual reach of I. No judge, no sampling of
opponents, no noise except which actions GRPO samples.

**5. Why does raising those values reduce exploitability?**

Player i's worst-case value is u_i(σ_i, BR(σ_i)), a maximum over opponent
strategies (taken with a minus sign). Danskin's theorem: where the best
response is unique, the gradient of that maximum is the gradient with the best
response held fixed. With the opponent fixed, the value is linear in the policy
at any one information set, and its derivative is
`d u_i / d σ_i(I, a) = π_i(I) × R(I) × Q(I, a)`, where π_i(I) is my own
probability of steering to I and R(I) the counterfactual reach. So ascending Q
weighted by the reach probability π_i(I)R(I) is gradient ascent on the
worst-case value, which is gradient descent on exploitability. This is
exploitability descent (Lockhart et al., 2019). The finite-difference test in
`tests/test_analysis.py` checks exactly this formula.

**6. Why did self-play outcome rewards fail?**

The payoff of a self-play hand is an unbiased estimate of an action's value
against the *current* opponent. Following that is simultaneous gradient play in
a zero-sum game, and gradient play does not converge there in general; it
orbits the equilibrium (rock-paper-scissors is the textbook case). Measured: on
Leduc the self-play policy's best point was 923 mA but it ended at 1,329, and
3 of 5 runs ended more than 30% above their best point. The best-response
reward does not have this problem because it asks "how good is this action
against the worst case", which does not move when the opponent's current
mistakes move.

**7. Describe GRPO and what you kept and changed.**

For each prompt, sample G answers from the current policy, score them, and use
(r − mean)/(std + 1e-6) within the group as the advantage, so no value network
is needed (Shao et al., 2024, DeepSeekMath). The loss is PPO's clipped
surrogate with ε = 0.2 plus a KL penalty to the reference model. I kept all of
that. One change: the policy at a decision is a categorical over two or three
actions, so I compute the KL exactly instead of with the usual one-sample
estimator. I also store a batch as, per information set and action, the summed
positive and negative advantages; that is exact because min(ρA, clip(ρ)A)
equals A × min(ρ, clip ρ) for A > 0 and A × max(ρ, clip ρ) for A < 0.

**8. The standard normalization hurt. Why?**

Near the equilibrium, the actions an equilibrium mixes between have nearly
equal values (that is what makes mixing optimal). Dividing by the group's
standard deviation turns a tiny difference into an advantage of size one, so
the step never shrinks and the policy keeps circling the equilibrium. On Leduc
after 2,000 steps: 49.7 mA normalized, 27.4 unnormalized, and lower learning
rates also ended lower (35.2 at 0.01 against 49.7 at 0.05). Both fit that
explanation; I have not proved it.

**9. Why not just run CFR+?**

For the poker games themselves, CFR+ is far better: 0.04 mA on Leduc against
48 for GRPO. The point is not to solve poker. It is a testbed where an
LLM-training method can be scored exactly. A language model cannot carry
CFR's per-decision regret tables, but it can be trained with gradients, which
is why the question "which reward makes gradient training work" matters. CFR+
also gives the supervised baseline: fine-tune the model to imitate the CFR+
equilibrium and compare.

**10. What happens at information sets the best response never lets you reach?**

Their counterfactual reach is zero, so Q is 0/0. The trainer skips them; the
exact gradient there is zero anyway (it is multiplied by the reach). They are
not frozen forever: when the policy changes, the best response changes, and
they can become reachable.

**11. How did you make this affordable on a laptop?**

A full Leduc extraction is 288 prompts. Levers: train on 32 random information
sets per step; recompute the full policy and the best response only every 10
steps and use those slightly stale values in between; score all answer words
from a single forward pass when each is one token; left-pad batches so the
model computes logits only for the last position; LoRA instead of full
fine-tuning. The tabular study measured what the stale values cost: on Leduc
with 32 information sets per step, 190 mA at the end with best responses every
10 steps against 115 with fresh ones every step.

**12. The torch code never ran in your sandbox. Why trust it?**

Everything that can be NumPy is NumPy and tested: the loss, the batch
bookkeeping, the trainer. The padding and "last k positions" logic is tested
against a NumPy stand-in language model. The torch-specific parts have their
own tests that CI runs with a tiny randomly initialized Qwen2 model and a
tokenizer built in the test: batched scoring equals unpadded
one-sequence-at-a-time scoring, the torch loss equals the NumPy loss and its
gradient, a resumed run equals an uninterrupted one, and a saved adapter
reloads to the same policy. On the Mac the smoke run and the full runs
(0.5B, 1.5B and 7B) then worked first time on MPS.

**13. What do the transfer tests show, and what don't they?**

Template B rewords the prompt, writes cards as letters and reorders the parts;
if the gain disappears there, the model learned the wording, not the game.
leduc4 adds a fourth rank, so it has information sets the model never trained
on; a gain there means something general about betting with strong and weak
hands carried over. Neither is evidence about real poker. Measured: the
supervised model went 2,289 → 339 mA with the training prompt but only
1,820 → 1,507 under the reworded one, and 2,560 → 917 on leduc4. So it
learned something about the game that carries to a bigger deck, but most of
the gain was tied to the wording. The collapsed GRPO policy scored about the
same everywhere (1,467 / 1,466 / 1,524), which says little, since it barely
learned.

**14. What can go wrong reading strategies from OpenAI models?**

Log-probabilities come back only for the top 20 first tokens, so a legal answer
outside the top 20 counts as zero; the legal mass column shows how much was
captured. Reasoning models do not return log-probabilities and reject
parameters like `temperature`, `max_tokens` and sometimes `n`; the client drops
or renames what a model rejects, remembers it, and retries. Sampled strategies
carry the noise bias from question 3. Cost is controlled by a dry run, a
ledger and a per-run token cap priced at the output rate, so the total cannot
pass the budget.

**15. How do you know the exploitability code is right?**

Four checks. It reproduces ZeroSum, a separate C++ implementation, on 14
strategies to the 10 digits ZeroSum prints. Every member of Kuhn's hand-derived
equilibrium family has exploitability zero. The best response is never beaten
by any of the 64 pure strategies on Kuhn. And the value computed by a backward
pass equals the value computed as a sum over terminal histories.

**16. Why 288 information sets for Leduc and not 936?**

936 keeps the two suits apart. Suits never matter in Leduc, so dealing ranks
with the right probabilities merges them and gives 288, with the same game
value. ZeroSum does the same, which is why the counts and order match.

**17. What would you do next?**

First, fix the GRPO collapse (question 18) and rerun: keep the best measured
checkpoint, lower the learning rate, drop the standard-deviation
normalization, and compare a KL penalty to the base model. Then anneal the
learning rate or drop the standard-deviation normalization late in
training and see whether GRPO approaches CFR+; train with the reach weighting
that makes the update the exact gradient; replace the exact best response with
a learned one, which is the step needed for games too large to solve; and try
the method on a game with bet sizing, where the action space is larger.

**18. Why did GRPO collapse on the language model but not in the tabular study?**

A group-relative advantage is zero when all G sampled answers agree. A softmax
policy that puts nearly all its mass on one answer produces identical samples,
so its groups are skipped and it gets no gradient, and nothing brings it back:
the gradients of a KL penalty or an entropy bonus also vanish at a
deterministic softmax. So the question is how it got there so fast. Two
things: dividing advantages by the group's standard deviation makes every
non-zero group a full-size step, however small the value difference; and in a
language model the LoRA parameters are shared across all 288 decisions, so a
step at one decision moves the others. At learning rate 1e-4 entropy went 0.60
→ 0.27 → 0.008 → 0.0003 nats at steps 0, 10, 20, 30. In the table, each
decision has its own parameters and moved at its own pace. The logs showed
it: `skipped_groups` equal to the batch size in 296 of 310 steps and a loss of
exactly zero. Remedies: keep the best checkpoint (it was 1,196 mA at step
10), a lower learning rate, unnormalized advantages, a KL penalty to the base
model applied before the policy collapses, or resampling at a higher
temperature when a group comes back uniform.

## Numbers to know

| Fact | Value |
| --- | --- |
| Kuhn / Leduc / leduc4 / leduc5 information sets | 12 / 288 / 504 / 780 |
| Game value to the first player | Kuhn −1/18; Leduc −0.085606424 |
| Agreement with ZeroSum (14 strategies) | largest relative difference 3.9e-10 |
| Leduc, uniform random play | 2,374 mA |
| Leduc, GRPO with best-response rewards (last 10%, median of 5) | 48.0 mA (exact expectation: 35.2) |
| Leduc, GRPO with self-play outcome rewards | 1,387 mA (29 times the br level) |
| Kuhn, br against self-play | 1.9 against 176 mA |
| Leduc, CFR+ after 3,000 iterations | 0.04 mA |
| Leduc after 2,000 steps: normalized / unnormalized / exact gradient | 49.7 / 27.4 / 15.4 mA |
| Laptop setting (32 infosets per step, refresh every 10), tabular, Leduc | 438 mA after 200 steps, 190 after 2,000 |
| Tests | 93: 82 run anywhere, 10 need torch (they run in CI on macOS and Ubuntu), 1 needs the ZeroSum binary |
| Base models, Leduc template A: Qwen2.5 0.5B / 1.5B / 7B | 2,389 / 2,289 / 1,245 mA (uniform 2,374) |
| Base models, Kuhn template A: 0.5B / 1.5B / 7B | 351 / 333 / 183 mA (uniform 458) |
| SFT on Qwen2.5-1.5B: Leduc A / Leduc B / leduc4 | 339 / 1,507 / 917 mA (base 2,289 / 1,820 / 2,560) |
| GRPO on Qwen2.5-1.5B: best (step 10) / final A / B / leduc4 | 1,196 / 1,467 / 1,466 / 1,524 mA |
| GRPO collapse | entropy 0.60 → 0.0003 nats by step 30; 296 of 310 steps had no gradient |
| OpenAI, Leduc: gpt-4.1-mini logprobs A / B / stated; gpt-5-mini sampled | 1,576 / 1,515 / 1,222; 710 mA |
| OpenAI, Kuhn: gpt-5-mini stated | 102 mA |
| OpenAI spend | $1.50 |

## What not to claim

- Not a poker bot and not evidence about real poker. These are research-sized
  games, chosen because they can be solved exactly.
- Not a new algorithm. GRPO, exploitability descent, CFR+ and LoRA are all
  published. The contribution is the exact measurement of language-model
  strategies, the reward-design comparison, and the tested pipeline.
- Do not say GRPO "solved" Leduc: 48 mA is far from CFR+'s 0.04.
- The tabular study is one optimizer, one initialization scale and a few
  hyperparameter sweeps. Say "in this setting".
- Do not say GRPO improved the language model: it helped for 10 steps and then
  collapsed; the final number is worse than its best checkpoint.
- Do not say the fine-tuned model "beats GPT-5": it beat gpt-5-mini on one
  prompt wording it was trained on, and gpt-5-mini's number is from 4 samples
  per decision. Under the reworded prompt the fine-tuned model was far worse.
- Every language-model training number is one run with one setting.
