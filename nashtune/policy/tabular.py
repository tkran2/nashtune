"""A softmax table: one logit per (infoset, action). Used for the tests and
for the reward-design study, where it stands in for a model with a separate
parameter for every decision. Gradients are analytic and gradient-checked."""

from __future__ import annotations

import numpy as np

from ..games.game import Game
from ..grpo import Adam, UpdateBatch, log_softmax, logits_grad, loss_and_grad
from .base import Policy


class TabularPolicy(Policy):
    trainable = True

    def __init__(self, game: Game, init: str = "uniform", init_scale: float = 1.0, seed: int = 0,
                 lr: float = 0.05, logits: np.ndarray | None = None) -> None:
        self.game = game
        if logits is not None:
            self.logits = np.array(logits, dtype=float)
        elif init == "uniform":
            self.logits = np.zeros(game.action_slots)
        elif init == "random":
            self.logits = np.random.default_rng(seed).normal(0.0, init_scale, game.action_slots)
        else:
            raise ValueError(f"unknown init {init!r} (uniform or random)")
        self.init = init
        self.init_scale = init_scale
        self.optimizer = Adam(game.action_slots, lr)

    @classmethod
    def from_strategy(cls, game: Game, strategy: np.ndarray, floor: float = 1e-12, **kwargs) -> "TabularPolicy":
        return cls(game, logits=np.log(np.maximum(np.asarray(strategy, dtype=float), floor)), **kwargs)

    def _slice(self, i: int) -> slice:
        info = self.game.infosets[int(i)]
        return slice(info.offset, info.offset + info.action_count)

    def logpi(self, ids) -> list[np.ndarray]:
        return [log_softmax(self.logits[self._slice(i)]) for i in ids]

    def strategy(self) -> np.ndarray:
        return self.extract().strategy

    def loss_and_logit_grad(self, batch: UpdateBatch) -> tuple[float, np.ndarray, dict]:
        logs = self.logpi(batch.ids)
        loss, grads, stats = loss_and_grad(logs, batch)
        full = np.zeros_like(self.logits)
        for i, lp, g in zip(batch.ids, logs, grads):
            full[self._slice(i)] += logits_grad(g, lp)
        return loss, full, stats

    def update(self, batch: UpdateBatch, n_updates: int = 1) -> dict:
        stats: dict = {}
        losses = []
        for _ in range(n_updates):
            loss, grad, stats = self.loss_and_logit_grad(batch)
            losses.append(loss)
            self.optimizer.step(self.logits, grad)
        return {"loss": float(losses[0]) if losses else 0.0, **stats}

    def state_dict(self) -> dict:
        return {"logits": self.logits.copy(), "optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.logits = np.array(state["logits"], dtype=float)
        self.optimizer.load_state_dict(state["optimizer"])

    def describe(self) -> dict:
        return {"backend": "tabular", "init": self.init, "init_scale": self.init_scale, "lr": self.optimizer.lr}
