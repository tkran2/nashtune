"""Backend-agnostic GRPO arithmetic in NumPy.

The policy at an information set is a small categorical distribution over the
legal actions, so everything here works on one log-probability vector per
infoset. The loss for one update batch is

    L = sum_I [ - sum_a ( P[I,a] * min(rho, clip(rho)) + N[I,a] * max(rho, clip(rho)) )
                + kl_coef[I] * KL(pi(I) || pi_ref(I))
                + ce_weight[I] * CE(target(I), pi(I)) ]

with rho = pi(a|I) / pi_old(a|I) and clip(rho) = clip(rho, 1 - eps, 1 + eps).
P and N are the sums of (sample weight x advantage) over the samples that took
action a at I with positive and negative advantage respectively. This is exactly
the PPO/GRPO clipped surrogate summed over samples, because
min(rho A, clip(rho) A) = A min(rho, clip(rho)) when A > 0 and
A max(rho, clip(rho)) when A < 0. Grouping by sign lets a batch of any number
of samples be stored as two small arrays per infoset.

The KL term is computed exactly over the legal actions (no estimator is
needed for a categorical this small). The cross-entropy term is used by the
supervised (SFT) baseline.

``nashtune.policy.torch_loss`` mirrors ``loss_and_grad`` line for line in torch;
a CI test checks that the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ADV_EPS = 1e-6  # added to the group standard deviation
ZERO_VAR = 1e-12  # groups with a smaller reward spread carry no signal and are skipped


# --------------------------------------------------------------- advantages
def group_advantages(rewards: np.ndarray, norm: str = "std") -> tuple[np.ndarray, bool]:
    """GRPO advantages for one group: (r - mean) / (std + 1e-6), population std.

    Returns ``(advantages, used)``; ``used`` is False (and the advantages zero)
    when every reward in the group is the same.
    """
    r = np.asarray(rewards, dtype=float)
    spread = float(r.std())
    if spread < ZERO_VAR:
        return np.zeros_like(r), False
    centered = r - r.mean()
    if norm == "std":
        return centered / (spread + ADV_EPS), True
    if norm == "none":
        return centered, True
    raise ValueError(f"unknown advantage normalization {norm!r}")


def expected_advantages(q: np.ndarray, pi: np.ndarray, norm: str = "std") -> tuple[np.ndarray, bool]:
    """The infinite-group limit of ``group_advantages``: rewards q[a] drawn with
    probability pi[a]. Mean and standard deviation are taken under pi."""
    q = np.asarray(q, dtype=float)
    pi = np.asarray(pi, dtype=float)
    mean = float(pi @ q)
    spread = float(np.sqrt(max(pi @ (q - mean) ** 2, 0.0)))
    if spread < ZERO_VAR:
        return np.zeros_like(q), False
    if norm == "std":
        return (q - mean) / (spread + ADV_EPS), True
    if norm == "none":
        return q - mean, True
    raise ValueError(f"unknown advantage normalization {norm!r}")


# -------------------------------------------------------------- the batch
@dataclass
class UpdateBatch:
    """Everything one optimizer update needs, per infoset (see module doc)."""

    ids: np.ndarray  # infoset indices, unique
    old_logp: list[np.ndarray]
    adv_pos: list[np.ndarray]
    adv_neg: list[np.ndarray]
    ref_logp: list[np.ndarray]
    kl_coef: np.ndarray
    ce_target: list[np.ndarray]
    ce_weight: np.ndarray
    clip_eps: float = 0.2
    stats: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(len(self.ids))

    def subset(self, positions) -> "UpdateBatch":
        pos = list(positions)
        return UpdateBatch(
            ids=self.ids[pos],
            old_logp=[self.old_logp[i] for i in pos],
            adv_pos=[self.adv_pos[i] for i in pos],
            adv_neg=[self.adv_neg[i] for i in pos],
            ref_logp=[self.ref_logp[i] for i in pos],
            kl_coef=self.kl_coef[pos],
            ce_target=[self.ce_target[i] for i in pos],
            ce_weight=self.ce_weight[pos],
            clip_eps=self.clip_eps,
            stats=self.stats,
        )


class BatchBuilder:
    """Accumulates weighted samples into an ``UpdateBatch``."""

    def __init__(self, action_counts: list[int] | np.ndarray, clip_eps: float = 0.2) -> None:
        self.action_counts = action_counts
        self.clip_eps = clip_eps
        self._order: list[int] = []
        self._pos: dict[int, np.ndarray] = {}
        self._neg: dict[int, np.ndarray] = {}
        self._old: dict[int, np.ndarray] = {}
        self._ref: dict[int, np.ndarray] = {}
        self._kl: dict[int, float] = {}
        self._ce_t: dict[int, np.ndarray] = {}
        self._ce_w: dict[int, float] = {}

    def _touch(self, infoset: int) -> None:
        if infoset not in self._pos:
            n = int(self.action_counts[infoset])
            self._order.append(infoset)
            self._pos[infoset] = np.zeros(n)
            self._neg[infoset] = np.zeros(n)
            self._kl[infoset] = 0.0
            self._ce_t[infoset] = np.zeros(n)
            self._ce_w[infoset] = 0.0

    def set_old(self, infoset: int, old_logp: np.ndarray) -> None:
        self._touch(infoset)
        self._old[infoset] = np.asarray(old_logp, dtype=float)

    def set_ref(self, infoset: int, ref_logp: np.ndarray) -> None:
        self._touch(infoset)
        self._ref[infoset] = np.asarray(ref_logp, dtype=float)

    def add_sample(self, infoset: int, action: int, advantage: float, weight: float) -> None:
        self._touch(infoset)
        value = weight * advantage
        if value > 0:
            self._pos[infoset][action] += value
        elif value < 0:
            self._neg[infoset][action] += value

    def add_kl(self, infoset: int, coef: float) -> None:
        self._touch(infoset)
        self._kl[infoset] += coef

    def add_ce(self, infoset: int, target: np.ndarray, weight: float) -> None:
        self._touch(infoset)
        self._ce_t[infoset] = np.asarray(target, dtype=float)
        self._ce_w[infoset] += weight

    def build(self) -> UpdateBatch:
        """The batch, without infosets that contribute nothing to the loss (for
        example a skipped zero-variance group with no KL term): a language
        model would otherwise spend a forward pass on them."""
        order = [i for i in self._order
                 if np.any(self._pos[i]) or np.any(self._neg[i]) or self._kl[i] or self._ce_w[i]]
        for i in order:
            if i not in self._old:
                raise ValueError(f"infoset {i} has samples but no old log-probabilities")
            if i not in self._ref:
                self._ref[i] = self._old[i].copy()
        return UpdateBatch(
            ids=np.array(order, dtype=np.int64),
            old_logp=[self._old[i] for i in order],
            adv_pos=[self._pos[i] for i in order],
            adv_neg=[self._neg[i] for i in order],
            ref_logp=[self._ref[i] for i in order],
            kl_coef=np.array([self._kl[i] for i in order], dtype=float),
            ce_target=[self._ce_t[i] for i in order],
            ce_weight=np.array([self._ce_w[i] for i in order], dtype=float),
            clip_eps=self.clip_eps,
        )


# ----------------------------------------------------------------- the loss
def kl_categorical(logp: np.ndarray, ref_logp: np.ndarray) -> float:
    p = np.exp(logp)
    return float(np.sum(p * (logp - ref_logp)))


def loss_and_grad(logp: list[np.ndarray], batch: UpdateBatch) -> tuple[float, list[np.ndarray], dict]:
    """Loss and its gradient with respect to each infoset's log-probability
    vector. ``logp[j]`` belongs to ``batch.ids[j]``."""
    eps = batch.clip_eps
    total = 0.0
    grads: list[np.ndarray] = []
    clipped_mass = 0.0
    adv_mass = 0.0
    kl_sum = 0.0
    for j, lp in enumerate(logp):
        lp = np.asarray(lp, dtype=float)
        ratio = np.exp(lp - batch.old_logp[j])
        clipped = np.clip(ratio, 1.0 - eps, 1.0 + eps)
        pos, neg = batch.adv_pos[j], batch.adv_neg[j]
        surrogate = np.sum(pos * np.minimum(ratio, clipped) + neg * np.maximum(ratio, clipped))
        # d min(r, clip r)/dr is 1 while r <= 1 + eps; d max(r, clip r)/dr is 1 while r >= 1 - eps.
        live_pos = ratio <= 1.0 + eps
        live_neg = ratio >= 1.0 - eps
        g = -(pos * live_pos + neg * live_neg) * ratio
        loss = -surrogate
        clipped_mass += float(np.sum(pos * ~live_pos) - np.sum(neg * ~live_neg))
        adv_mass += float(np.sum(pos) - np.sum(neg))
        if batch.kl_coef[j] != 0.0:
            p = np.exp(lp)
            kl = float(np.sum(p * (lp - batch.ref_logp[j])))
            kl_sum += kl
            loss += batch.kl_coef[j] * kl
            g = g + batch.kl_coef[j] * p * (lp - batch.ref_logp[j] + 1.0)
        if batch.ce_weight[j] != 0.0:
            loss += -batch.ce_weight[j] * float(np.sum(batch.ce_target[j] * lp))
            g = g - batch.ce_weight[j] * batch.ce_target[j]
        total += float(loss)
        grads.append(g)
    stats = {"clip_frac": clipped_mass / adv_mass if adv_mass > 0 else 0.0}
    return total, grads, stats


# --------------------------------------------------- softmax helpers (tabular)
def log_softmax(x: np.ndarray) -> np.ndarray:
    m = np.max(x)
    z = x - m
    return z - np.log(np.sum(np.exp(z)))


def logits_grad(g_logp: np.ndarray, logp: np.ndarray) -> np.ndarray:
    """Chain rule through log-softmax: dL/dtheta_b = g_b - pi_b * sum_a g_a."""
    return g_logp - np.exp(logp) * np.sum(g_logp)


class Adam:
    """Adam (Kingma and Ba, 2015) on a flat NumPy parameter vector."""

    def __init__(self, size: int, lr: float, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8) -> None:
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.m = np.zeros(size)
        self.v = np.zeros(size)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> None:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad * grad
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        params -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

    def state_dict(self) -> dict:
        return {"lr": self.lr, "m": self.m.copy(), "v": self.v.copy(), "t": self.t}

    def load_state_dict(self, state: dict) -> None:
        self.lr = float(state["lr"])
        self.m = np.array(state["m"], dtype=float)
        self.v = np.array(state["v"], dtype=float)
        self.t = int(state["t"])


def entropy(logp: np.ndarray) -> float:
    p = np.exp(logp)
    return float(-np.sum(np.where(p > 0, p * logp, 0.0)))
