"""Strategy files in ZeroSum's format: one line per action,
``infoset action probability``, for example ``0:K:J:kb/b c 0.5``. Lines starting
with ``#`` are comments. ``zerosum eval GAME --strategy FILE`` reads the same
files."""

from __future__ import annotations

import os

import numpy as np

from .game import Game


def format_strategy(game: Game, strategy: np.ndarray) -> str:
    strategy = np.asarray(strategy, dtype=float)
    lines = [f"# zerosum strategy for {game.name}", "# infoset action probability"]
    for info in game.infosets:
        for a, action in enumerate(info.actions):
            lines.append(f"{info.key} {action} {float(strategy[info.offset + a]):.17g}")
    return "\n".join(lines) + "\n"


def write_strategy(game: Game, strategy: np.ndarray, path: str | os.PathLike) -> None:
    game.check_strategy(strategy)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(format_strategy(game, strategy))
    os.replace(tmp, path)


def parse_strategy(game: Game, text: str, source: str = "strategy") -> np.ndarray:
    strategy = np.full(game.action_slots, -1.0)
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"{source}: bad strategy line: {line!r}")
        key, action, prob_text = parts
        try:
            info = game.infoset_by_key(key)
        except KeyError:
            raise ValueError(f"{source}: strategy is for a different game: {key}") from None
        position = info.actions.find(action) if len(action) == 1 else -1
        try:
            prob = float(prob_text)
        except ValueError:
            raise ValueError(f"{source}: bad strategy line: {line!r}") from None
        if position < 0 or not (-1e-9 <= prob <= 1 + 1e-9):
            raise ValueError(f"{source}: bad strategy line: {line!r}")
        strategy[info.offset + position] = prob
    for info in game.infosets:
        chunk = strategy[info.offset : info.offset + info.action_count]
        if np.any(chunk < 0):
            raise ValueError(f"{source}: strategy is missing {info.key}")
        if abs(float(chunk.sum()) - 1.0) > 1e-6:
            raise ValueError(f"{source}: probabilities at {info.key} do not sum to 1")
    return strategy


def read_strategy(game: Game, path: str | os.PathLike) -> np.ndarray:
    with open(path, encoding="utf-8") as fh:
        return parse_strategy(game, fh.read(), str(path))
