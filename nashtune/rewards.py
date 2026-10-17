"""Exact, verifiable rewards.

``br`` (exploitability descent). For a policy sigma, let b be the opponent's
best response to it. For an infoset I of player i, the reward for action a is
Q(I, a): the expected payoff of taking a at I, conditioned on reaching I, when
player i follows sigma afterwards and the opponent plays b. It equals the
counterfactual action value divided by the counterfactual reach of I.
Raising these values is gradient ascent on player i's worst-case payoff
(Danskin's theorem: the gradient of a max is the gradient at the maximizer),
which is gradient descent on exploitability (Lockhart et al., 2019).

``selfplay`` (the naive baseline). Sample complete hands with the current
policy in both seats; each decision is rewarded with the acting player's final
payoff for that hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .games import game_value
from .games.analysis import Exploitability, best_response, counterfactual_values, expected_value
from .games.game import CHANCE, DECISION, TERMINAL, Game


@dataclass
class RefreshInfo:
    """What the trainer recomputes from the full current policy."""

    strategy: np.ndarray
    expl: Exploitability
    q: np.ndarray  # conditional action values Q(I, a), per slot
    defined: np.ndarray  # per infoset: reachable by chance and the best-responding opponent
    cf_reach: np.ndarray  # per infoset
    own_reach: np.ndarray  # per infoset, under sigma
    br_profile: list[np.ndarray]  # br_profile[i]: player i's sigma with the opponent's best response

    @property
    def reach_weight(self) -> np.ndarray:
        """Probability of reaching each infoset when its owner plays sigma and the
        opponent plays the best response: own reach x counterfactual reach."""
        return self.own_reach * self.cf_reach

    def seat_milli(self, name: str) -> tuple[float, float]:
        return self.expl.seat_milli(game_value(name))


def refresh(game: Game, strategy: np.ndarray, need_values: bool = True) -> RefreshInfo:
    strategy = np.asarray(strategy, dtype=float)
    br_values = [0.0, 0.0]
    profiles: list[np.ndarray] = [strategy, strategy]
    q = np.zeros(game.action_slots)
    cf_reach = np.zeros(game.num_infosets)
    own_reach = np.zeros(game.num_infosets)
    defined = np.zeros(game.num_infosets, dtype=bool)
    for player in (0, 1):
        # The opponent best-responds to player's part of the strategy.
        value, profile = best_response(game, strategy, 1 - player)
        br_values[1 - player] = value
        profiles[player] = profile
        if need_values:
            cf = counterfactual_values(game, profile, player)
            qp, dp = cf.conditional(game)
            mask = game.slot_player == player
            q[mask] = qp[mask]
            pmask = game.infoset_player == player
            cf_reach[pmask] = cf.cf_reach[pmask]
            own_reach[pmask] = cf.own_reach[pmask]
            defined[pmask] = dp[pmask]
    expl = Exploitability(value=expected_value(game, strategy), best_vs_p1=br_values[0], best_vs_p0=br_values[1])
    return RefreshInfo(strategy=strategy, expl=expl, q=q, defined=defined, cf_reach=cf_reach,
                       own_reach=own_reach, br_profile=profiles)


# ------------------------------------------------------------------ self-play
@dataclass
class Rollouts:
    deal: np.ndarray  # group (deal) index of each rollout
    payoff: np.ndarray  # payoff to player 0 of each rollout
    decisions: list[tuple[int, int, int, int]]  # (rollout, player, infoset, action)
    probs: dict[int, np.ndarray]  # the policy used, per infoset visited


def sample_deals(game: Game, count: int, rng: np.random.Generator) -> list[tuple[int, float]]:
    """``count`` deals: a child of the root chance node (the private cards) and
    a uniform number that fixes the public card for every rollout of the deal."""
    root = 0
    start, n = int(game.first_child[root]), int(game.child_count[root])
    probs = game.chance_prob[start : start + n]
    picks = rng.choice(n, size=count, p=probs / probs.sum())
    publics = rng.random(count)
    return [(int(game.children[start + int(k)]), float(u)) for k, u in zip(picks, publics)]


def _pick(probs: np.ndarray, u: float) -> int:
    c = np.cumsum(probs)
    k = int(np.searchsorted(c, u * c[-1], side="right"))
    return min(k, len(probs) - 1)


def simulate(game: Game, deals: list[tuple[int, float]], group_size: int, probs_fn,
             rng: np.random.Generator) -> Rollouts:
    """Play ``group_size`` hands from each deal with the policy in both seats.

    Rollouts advance in lockstep so the policy is queried once per batch of
    newly visited infosets (``probs_fn(list_of_ids) -> list of prob arrays``)."""
    n = len(deals) * group_size
    node = np.array([deals[r // group_size][0] for r in range(n)], dtype=np.int64)
    public_u = np.array([deals[r // group_size][1] for r in range(n)])
    cache: dict[int, np.ndarray] = {}
    decisions: list[tuple[int, int, int, int]] = []
    active = list(range(n))
    while active:
        waiting: list[int] = []
        for r in active:
            h = int(node[r])
            while game.node_type[h] == CHANCE:
                start, k = int(game.first_child[h]), int(game.child_count[h])
                h = int(game.children[start + _pick(game.chance_prob[start : start + k], public_u[r])])
            node[r] = h
            if game.node_type[h] == DECISION:
                waiting.append(r)
        if not waiting:
            break
        need = sorted({int(game.node_infoset[node[r]]) for r in waiting} - cache.keys())
        if need:
            for i, p in zip(need, probs_fn(need)):
                p = np.asarray(p, dtype=float)
                cache[i] = p / p.sum()
        draws = rng.random(len(waiting))
        for r, u in zip(waiting, draws):
            h = int(node[r])
            i = int(game.node_infoset[h])
            a = _pick(cache[i], float(u))
            decisions.append((r, int(game.node_player[h]), i, a))
            node[r] = game.children[int(game.first_child[h]) + a]
        active = [r for r in waiting if game.node_type[node[r]] != TERMINAL]
        # Rollouts sitting on a chance node are advanced at the top of the loop.
    payoff = game.utility[node]
    return Rollouts(deal=np.arange(n) // group_size, payoff=payoff, decisions=decisions, probs=cache)
