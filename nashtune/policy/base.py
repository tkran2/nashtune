"""The policy interface every backend implements.

A policy maps an information set to a probability for each legal action.
``logpi`` returns those log-probabilities (renormalized over the legal
actions) for a list of infosets; ``extract`` returns the complete strategy
for the whole game, which is what exploitability is measured on.
Trainable backends also implement ``update`` (one or more optimizer steps on
the loss defined by a ``nashtune.grpo.UpdateBatch``) and checkpoint state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..games.game import Game


@dataclass
class Extraction:
    strategy: np.ndarray  # one probability per action slot, each infoset sums to 1
    legal_mass: np.ndarray | None = None  # per infoset: unnormalized probability on legal answers
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)


class Policy:
    game: Game
    trainable: bool = False

    def logpi(self, ids) -> list[np.ndarray]:
        raise NotImplementedError

    def extract(self) -> Extraction:
        start = time.time()
        ids = np.arange(self.game.num_infosets)
        logs = self.logpi(ids)
        strategy = np.empty(self.game.action_slots)
        for info, lp in zip(self.game.infosets, logs):
            p = np.exp(lp)
            strategy[info.offset : info.offset + info.action_count] = p / p.sum()
        return Extraction(strategy=strategy, seconds=time.time() - start)

    def update(self, batch, n_updates: int = 1) -> dict:
        raise NotImplementedError(f"{type(self).__name__} cannot be trained")

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> None:
        pass

    def save(self, directory) -> None:
        """Write whatever is needed to reuse the trained policy (optional)."""

    def describe(self) -> dict:
        return {"backend": type(self).__name__}


def strategy_from_logs(game: Game, ids, logs: list[np.ndarray], base: np.ndarray | None = None) -> np.ndarray:
    out = np.array(base, dtype=float, copy=True) if base is not None else game.uniform_strategy()
    for i, lp in zip(ids, logs):
        info = game.infosets[int(i)]
        p = np.exp(lp)
        out[info.offset : info.offset + info.action_count] = p / p.sum()
    return out
