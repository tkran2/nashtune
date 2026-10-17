"""Exact game-tree arithmetic: expected value, reach probabilities, best
response, exploitability, and counterfactual action values.

Everything is vectorized over the nodes at one depth of the tree, so a full
Leduc best response takes a few milliseconds in NumPy. Definitions match
ZeroSum's include/zerosum/analysis.hpp.

Notation: ``strategy`` is a behavior profile for both players, one probability
per action slot. Values are in antes; exploitability is reported in
milli-antes per hand (mA) by ``Exploitability.milli``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .game import Game


def _with_one(strategy: np.ndarray) -> np.ndarray:
    # Index -1 (edges out of chance nodes) reads 1.0.
    return np.append(np.asarray(strategy, dtype=float), 1.0)


def reach(game: Game, strategy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chance reach and each player's own reach of every node.

    Returns ``(chance, own)`` with ``own[p][h]`` the product of player p's
    action probabilities on the path to h.
    """
    sig = _with_one(strategy)
    n = game.num_nodes
    chance = np.ones(n)
    own = np.ones((2, n))
    for idx in game.by_depth[1:]:
        par = game.parent[idx]
        chance[idx] = chance[par] * game.edge_prob[idx]
        f = sig[game.edge_slot[idx]]
        pp = game.parent_player[idx]
        own[0, idx] = own[0, par] * np.where(pp == 0, f, 1.0)
        own[1, idx] = own[1, par] * np.where(pp == 1, f, 1.0)
    return chance, own


def node_values(game: Game, strategy: np.ndarray) -> np.ndarray:
    """Expected payoff to player 0 from every node when both follow ``strategy``."""
    sig = _with_one(strategy)
    weight = np.where(game.edge_slot >= 0, sig[game.edge_slot], game.edge_prob)
    v = game.utility.copy()
    n = game.num_nodes
    for idx in reversed(game.by_depth[1:]):
        v += np.bincount(game.parent[idx], weights=weight[idx] * v[idx], minlength=n)
    return v


def expected_value(game: Game, strategy: np.ndarray) -> float:
    """Payoff to player 0 when both players follow ``strategy``."""
    return float(node_values(game, strategy)[0])


def expected_value_terminal_sum(game: Game, strategy: np.ndarray) -> float:
    """The same number computed a second way: sum over terminal histories of
    reach probability times payoff. Used as a self-check in the tests."""
    chance, own = reach(game, strategy)
    z = game.terminals
    return float(np.sum(chance[z] * own[0, z] * own[1, z] * game.utility[z]))


def best_response(game: Game, strategy: np.ndarray, me: int) -> tuple[float, np.ndarray]:
    """The most ``me`` can win, on average, against the other player's part of
    ``strategy``, and a pure strategy that achieves it.

    The returned profile copies the opponent's part of ``strategy`` and puts the
    best response in ``me``'s slots. At each of my information sets I must pick
    one action for every history in it, so the comparison is between sums over
    the set, each history weighted by the probability that chance and the
    opponent bring the game there. Ties go to the first action, as in ZeroSum.
    """
    chance, own = reach(game, strategy)
    cr = chance * own[1 - me]
    sig = _with_one(strategy)
    other_weight = np.where(game.edge_slot >= 0, sig[game.edge_slot], game.edge_prob)
    sign = 1.0 if me == 0 else -1.0
    v = sign * game.utility
    br = np.array(strategy, dtype=float, copy=True)
    n = game.num_nodes
    for d in range(len(game.by_depth) - 1, 0, -1):
        idx = game.by_depth[d]
        # Decide my infosets one level up, using the finished values at depth d.
        infos = game.infosets_at.get((d - 1, me))
        if infos is not None and infos.size:
            mine = idx[game.parent_player[idx] == me]
            score = np.bincount(game.edge_slot[mine], weights=cr[game.parent[mine]] * v[mine],
                                minlength=game.action_slots)
            slots = game.action_matrix[infos]
            vals = np.where(slots >= 0, score[slots], -np.inf)
            best = np.argmax(vals, axis=1)
            valid = slots >= 0
            br[slots[valid]] = 0.0
            br[slots[np.arange(len(infos)), best]] = 1.0
        brw = np.append(br, 1.0)
        weight = np.where(game.parent_player[idx] == me, brw[game.edge_slot[idx]], other_weight[idx])
        v += np.bincount(game.parent[idx], weights=weight * v[idx], minlength=n)
    return float(v[0]), br


@dataclass(frozen=True)
class Exploitability:
    value: float  # payoff to player 0 when both follow the strategy
    best_vs_p1: float  # player 0's best-response payoff against player 1's part
    best_vs_p0: float  # player 1's best-response payoff against player 0's part

    @property
    def exploitability(self) -> float:
        """Average amount per hand (antes) a perfect opponent wins from this
        strategy, over both seats. Zero exactly at a Nash equilibrium."""
        return 0.5 * (self.best_vs_p1 + self.best_vs_p0)

    @property
    def milli(self) -> float:
        return 1000.0 * self.exploitability

    @property
    def value_lower(self) -> float:
        return -self.best_vs_p0

    @property
    def value_upper(self) -> float:
        return self.best_vs_p1

    def seat_milli(self, game_value: float) -> tuple[float, float]:
        """How much a perfect opponent wins beyond the game value against the
        strategy's seat-0 and seat-1 play, in mA. They average to ``milli``."""
        return (1000.0 * (self.best_vs_p0 + game_value), 1000.0 * (self.best_vs_p1 - game_value))

    def as_dict(self) -> dict[str, float]:
        return {"value": self.value, "best_vs_p1": self.best_vs_p1, "best_vs_p0": self.best_vs_p0,
                "exploitability": self.exploitability, "exploitability_ma": self.milli}


def exploitability(game: Game, strategy: np.ndarray) -> Exploitability:
    return Exploitability(
        value=expected_value(game, strategy),
        best_vs_p1=best_response(game, strategy, 0)[0],
        best_vs_p0=best_response(game, strategy, 1)[0],
    )


@dataclass(frozen=True)
class CounterfactualValues:
    """Counterfactual quantities for one player against a fixed profile.

    ``q[slot]``: sum over histories h in the slot's infoset of (chance x
    opponent reach of h) x (player's expected payoff after taking the action at
    h and following the profile). Only the player's slots are filled.
    ``cf_reach[I]``: sum over h in I of chance x opponent reach.
    ``own_reach[I]``: the player's own probability of steering to I.
    """

    player: int
    q: np.ndarray
    cf_reach: np.ndarray
    own_reach: np.ndarray
    value: float  # player's expected payoff under the profile

    def conditional(self, game: Game, min_reach: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
        """Expected payoff of each action conditioned on reaching its infoset:
        q / cf_reach. Returns ``(Q, defined)``; Q is 0 where the infoset cannot
        be reached by the opponent and chance (``defined`` False there)."""
        r = self.cf_reach[game.slot_infoset]
        defined = (game.slot_player == self.player) & (r > min_reach)
        out = np.zeros_like(self.q)
        out[defined] = self.q[defined] / r[defined]
        return out, defined[game.action_matrix[:, 0]]


def counterfactual_values(game: Game, profile: np.ndarray, player: int) -> CounterfactualValues:
    chance, own = reach(game, profile)
    v = node_values(game, profile)
    if player == 1:
        v = -v
    cr = chance * own[1 - player]
    edges = game.player_edges[player]
    q = np.bincount(game.edge_slot[edges], weights=cr[game.parent[edges]] * v[edges],
                    minlength=game.action_slots)
    nodes = game.player_nodes[player]
    infoset_of = game.node_infoset[nodes]
    cf_reach = np.bincount(infoset_of, weights=cr[nodes], minlength=game.num_infosets)
    own_reach = np.zeros(game.num_infosets)
    own_reach[infoset_of] = own[player, nodes]
    return CounterfactualValues(player=player, q=q, cf_reach=cf_reach, own_reach=own_reach, value=float(v[0]))


def own_reach(game: Game, strategy: np.ndarray, who: int) -> np.ndarray:
    """Probability that player ``who``'s own choices lead to each of its infosets."""
    _, own = reach(game, strategy)
    nodes = game.player_nodes[who]
    out = np.zeros(game.num_infosets)
    out[game.node_infoset[nodes]] = own[who, nodes]
    return out


def infoset_values(game: Game, strategy: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-infoset sum over actions of probability x q."""
    return np.bincount(game.slot_infoset, weights=np.asarray(strategy) * q, minlength=game.num_infosets)
