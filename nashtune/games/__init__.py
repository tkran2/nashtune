"""Small poker games and exact game-tree analysis."""

from __future__ import annotations

from .analysis import (
    CounterfactualValues,
    Exploitability,
    best_response,
    counterfactual_values,
    expected_value,
    exploitability,
    node_values,
    own_reach,
    reach,
)
from .cfr import CFRPlus, solve, solve_to
from .game import CHANCE, DECISION, TERMINAL, Game, Infoset
from .poker import PokerRules, build_poker, kuhn_rules, leduc_rules, make_game, rules_for
from .strategy_io import format_strategy, parse_strategy, read_strategy, write_strategy

# Values to the first player, in antes per hand. Kuhn's is exact (-1/18); the
# others are from ZeroSum's sequence-form linear program (scripts/lp_oracle.py).
GAME_VALUES = {
    "kuhn": -1.0 / 18.0,
    "leduc": -0.085606424078,
    "leduc4": -0.102242921155,
    "leduc5": -0.112768934481,
}


def game_value(name: str) -> float:
    """The game's value to the first player. Falls back to a long CFR+ run for
    games not in the table (accurate to about 1e-6 antes)."""
    if name not in GAME_VALUES:
        game = make_game(name)
        strategy, _, _ = solve_to(game, target_ma=1e-3, max_iterations=20_000)
        GAME_VALUES[name] = exploitability(game, strategy).value
    return GAME_VALUES[name]

__all__ = [
    "CHANCE", "DECISION", "TERMINAL", "GAME_VALUES", "CFRPlus", "CounterfactualValues", "Exploitability",
    "Game", "Infoset", "PokerRules", "best_response", "build_poker", "counterfactual_values",
    "expected_value", "exploitability", "format_strategy", "game_value", "kuhn_rules", "leduc_rules", "make_game",
    "node_values", "own_reach", "parse_strategy", "reach", "read_strategy", "rules_for", "solve",
    "solve_to", "write_strategy",
]
