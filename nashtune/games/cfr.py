"""CFR+ (Tammelin, 2014): the reference equilibrium solver and the optimal
baseline in the training-dynamics study.

Each iteration updates the players in turn. At every infoset the solver adds
each action's counterfactual regret (how much better the action would have
done than the current strategy, weighted by how likely chance and the opponent
are to bring the game there), floors the running total at zero, and plays
actions in proportion to it. The average strategy, with iteration t weighted
by t, converges to a Nash equilibrium.
"""

from __future__ import annotations

import numpy as np

from .analysis import counterfactual_values, exploitability, infoset_values, own_reach
from .game import Game


class CFRPlus:
    def __init__(self, game: Game) -> None:
        self.game = game
        self.regret = np.zeros(game.action_slots)
        self.strategy_sum = np.zeros(game.action_slots)
        self.iteration = 0
        self._counts = np.array([info.action_count for info in game.infosets], dtype=float)

    def current_strategy(self) -> np.ndarray:
        g = self.game
        pos = np.maximum(self.regret, 0.0)
        total = np.bincount(g.slot_infoset, weights=pos, minlength=g.num_infosets)[g.slot_infoset]
        uniform = 1.0 / self._counts[g.slot_infoset]
        return np.where(total > 0, pos / np.where(total > 0, total, 1.0), uniform)

    def average_strategy(self) -> np.ndarray:
        g = self.game
        total = np.bincount(g.slot_infoset, weights=self.strategy_sum, minlength=g.num_infosets)[g.slot_infoset]
        uniform = 1.0 / self._counts[g.slot_infoset]
        return np.where(total > 0, self.strategy_sum / np.where(total > 0, total, 1.0), uniform)

    def iterate(self) -> None:
        g = self.game
        self.iteration += 1
        t = float(self.iteration)
        for player in (0, 1):
            sigma = self.current_strategy()
            cf = counterfactual_values(g, sigma, player)
            mask = g.slot_player == player
            value = infoset_values(g, np.where(mask, sigma, 0.0), cf.q)[g.slot_infoset]
            self.regret[mask] += (cf.q - value)[mask]
            np.maximum(self.regret, 0.0, out=self.regret, where=mask)
            reach_me = own_reach(g, sigma, player)[g.slot_infoset]
            self.strategy_sum[mask] += (t * reach_me * sigma)[mask]

    def run(self, iterations: int, every: int = 0, callback=None) -> np.ndarray:
        for _ in range(iterations):
            self.iterate()
            if callback is not None and every and self.iteration % every == 0:
                callback(self)
        return self.average_strategy()


def solve(game: Game, iterations: int = 2000) -> np.ndarray:
    """CFR+ average strategy after ``iterations`` iterations."""
    return CFRPlus(game).run(iterations)


def solve_to(game: Game, target_ma: float, max_iterations: int = 100_000, check_every: int = 50) -> tuple[np.ndarray, int, float]:
    """Run CFR+ until exploitability is at most ``target_ma`` milli-antes."""
    solver = CFRPlus(game)
    gap = float("inf")
    while solver.iteration < max_iterations:
        solver.run(check_every)
        gap = exploitability(game, solver.average_strategy()).milli
        if gap <= target_ma:
            break
    return solver.average_strategy(), solver.iteration, gap
