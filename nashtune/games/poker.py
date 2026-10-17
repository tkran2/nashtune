"""The Kuhn/Leduc family of small poker games, built exactly as ZeroSum builds
them (include/zerosum/game.hpp): same rules, same node numbering, same
information-set keys and order.

Each player antes, is dealt one private card, and bets over one or two rounds;
a public card is revealed before the second round. A pair with the public card
beats any high card. Cards are dealt as ranks with probabilities, so the two
suits of Leduc are merged (288 information sets rather than 936).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from .game import CHANCE, DECISION, TERMINAL, Game, Infoset

_THIRTEEN = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
RANK_WORDS = {
    "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
    "8": "Eight", "9": "Nine", "T": "Ten", "J": "Jack", "Q": "Queen", "K": "King", "A": "Ace",
}


@dataclass(frozen=True)
class PokerRules:
    name: str
    ranks: int = 3  # distinct card ranks
    copies: int = 1  # cards per rank in the deck
    rounds: int = 1
    bet: tuple[int, int] = (1, 1)  # bet size in each round
    max_raises: int = 1  # bets plus raises allowed per round
    ante: int = 1

    def rank_names(self) -> list[str]:
        # Three-rank games use J, Q, K as in the papers that introduced them;
        # larger decks take the top N ranks of a standard deck.
        first = 9 if self.ranks == 3 else 13 - self.ranks
        return _THIRTEEN[first : first + self.ranks]


def kuhn_rules() -> PokerRules:
    """Kuhn (1950): three cards, one round, one bet. Solvable by hand."""
    return PokerRules("kuhn", 3, 1, 1, (1, 1), 1, 1)


def leduc_rules(ranks: int = 3) -> PokerRules:
    """Leduc hold'em (Southey et al., 2005), optionally with more ranks."""
    return PokerRules("leduc" if ranks == 3 else f"leduc{ranks}", ranks, 2, 2, (2, 4), 2, 1)


def rules_for(name: str) -> PokerRules:
    if name == "kuhn":
        return kuhn_rules()
    if name == "leduc":
        return leduc_rules(3)
    if name.startswith("leduc"):
        try:
            ranks = int(name[5:])
        except ValueError:
            raise ValueError(f"unknown game '{name}' (try kuhn, leduc, leduc4, leduc5)") from None
        return leduc_rules(ranks)
    raise ValueError(f"unknown game '{name}' (try kuhn, leduc, leduc4, leduc5)")


@dataclass
class _State:
    rank: tuple[int, int]
    public_rank: int
    round: int
    paid: list[int]
    raises: int  # bets and raises so far this round
    to_act: int
    checked: bool  # the first player checked this round
    history: str

    def copy(self) -> "_State":
        return _State(self.rank, self.public_rank, self.round, list(self.paid), self.raises,
                      self.to_act, self.checked, self.history)


class _Builder:
    def __init__(self, rules: PokerRules) -> None:
        if not (2 <= rules.ranks <= 13 and 1 <= rules.copies <= 4 and 1 <= rules.rounds <= 2
                and rules.max_raises >= 1 and rules.ante >= 1):
            raise ValueError("unsupported poker rules")
        if rules.ranks * rules.copies < 2 + (rules.rounds - 1):
            raise ValueError("deck is too small")
        self.rules = rules
        self.rank_names = rules.rank_names()
        self.node_type: list[int] = []
        self.node_player: list[int] = []
        self.node_infoset: list[int] = []
        self.first_child: list[int] = []
        self.child_count: list[int] = []
        self.utility: list[float] = []
        self.children: list[int] = []
        self.chance_prob: list[float] = []
        self.infosets: list[Infoset] = []
        self.ids: dict[str, int] = {}
        self.action_slots = 0

    def add_node(self, kind: int) -> int:
        self.node_type.append(kind)
        self.node_player.append(-1)
        self.node_infoset.append(-1)
        self.first_child.append(0)
        self.child_count.append(0)
        self.utility.append(0.0)
        return len(self.node_type) - 1

    def attach(self, node: int, kids: list[int], probs: list[float] | None) -> None:
        self.first_child[node] = len(self.children)
        self.child_count[node] = len(kids)
        for i, kid in enumerate(kids):
            self.children.append(kid)
            self.chance_prob.append(probs[i] if probs else 0.0)

    def terminal(self, utility: float) -> int:
        node = self.add_node(TERMINAL)
        self.utility[node] = float(utility)
        return node

    def showdown(self, s: _State) -> int:
        def strength(player: int) -> int:
            rank = s.rank[player]
            return (100 if rank == s.public_rank else 0) + rank  # a pair beats any high card

        a, b = strength(0), strength(1)
        if a == b:
            return self.terminal(0.0)
        return self.terminal(s.paid[1] if a > b else -s.paid[0])

    def end_of_round(self, s: _State) -> int:
        if s.round + 1 >= self.rules.rounds:
            return self.showdown(s)
        deck = self.rules.ranks * self.rules.copies
        node = self.add_node(CHANCE)
        kids: list[int] = []
        probs: list[float] = []
        for r in range(self.rules.ranks):
            left = self.rules.copies - (r == s.rank[0]) - (r == s.rank[1])
            if left <= 0:
                continue
            nxt = s.copy()
            nxt.public_rank = r
            nxt.round = s.round + 1
            nxt.raises = 0
            nxt.to_act = 0
            nxt.checked = False
            nxt.history += "/"
            kids.append(self.betting(nxt))
            probs.append(left / (deck - 2))
        self.attach(node, kids, probs)
        return node

    def betting(self, s: _State) -> int:
        me = s.to_act
        other = 1 - me
        owed = s.paid[other] - s.paid[me]
        size = self.rules.bet[s.round]
        if owed > 0:
            actions = "fc" + ("r" if s.raises < self.rules.max_raises else "")
        else:
            actions = "k" + ("b" if s.raises < self.rules.max_raises else "")

        node = self.add_node(DECISION)
        public_name = "-" if s.public_rank < 0 else self.rank_names[s.public_rank]
        key = f"{me}:{self.rank_names[s.rank[me]]}:{public_name}:{s.history}"
        index = self.ids.get(key)
        if index is None:
            index = len(self.infosets)
            self.infosets.append(Infoset(index=index, player=me, key=key, actions=actions,
                                         offset=self.action_slots, private_rank=s.rank[me],
                                         public_rank=s.public_rank, history=s.history))
            self.action_slots += len(actions)
            self.ids[key] = index
        self.node_player[node] = me
        self.node_infoset[node] = index

        kids: list[int] = []
        for action in actions:
            nxt = s.copy()
            nxt.history += action
            nxt.to_act = other
            if action == "f":
                kids.append(self.terminal(-s.paid[0] if me == 0 else s.paid[1]))
            elif action == "c":
                nxt.paid[me] = s.paid[other]
                kids.append(self.end_of_round(nxt))
            elif action == "k":
                if s.checked:
                    kids.append(self.end_of_round(nxt))
                else:
                    nxt.checked = True
                    kids.append(self.betting(nxt))
            else:  # 'b' or 'r': match what is owed, then add one bet
                nxt.paid[me] = s.paid[other] + size
                nxt.raises = s.raises + 1
                kids.append(self.betting(nxt))
        self.attach(node, kids, None)
        return node

    def build(self) -> Game:
        rules = self.rules
        deck = rules.ranks * rules.copies
        root = self.add_node(CHANCE)
        kids: list[int] = []
        probs: list[float] = []
        for a in range(rules.ranks):
            for b in range(rules.ranks):
                left = rules.copies - (1 if a == b else 0)
                if left <= 0:
                    continue
                state = _State((a, b), -1, 0, [rules.ante, rules.ante], 0, 0, False, "")
                kids.append(self.betting(state))
                probs.append(rules.copies / deck * left / (deck - 1))
        self.attach(root, kids, probs)
        return Game(
            name=rules.name,
            rules=rules,
            node_type=np.array(self.node_type, dtype=np.int8),
            node_player=np.array(self.node_player, dtype=np.int8),
            node_infoset=np.array(self.node_infoset, dtype=np.int64),
            first_child=np.array(self.first_child, dtype=np.int64),
            child_count=np.array(self.child_count, dtype=np.int64),
            utility=np.array(self.utility, dtype=float),
            children=np.array(self.children, dtype=np.int64),
            chance_prob=np.array(self.chance_prob, dtype=float),
            infosets=self.infosets,
            action_slots=self.action_slots,
            rank_names=self.rank_names,
        )


def build_poker(rules: PokerRules) -> Game:
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, 10_000))
    try:
        return _Builder(rules).build()
    finally:
        sys.setrecursionlimit(limit)


_CACHE: dict[str, Game] = {}


def make_game(name: str) -> Game:
    """"kuhn", "leduc", or "leducN" for N ranks (4 to 13). Games are cached;
    treat the returned object as read-only."""
    if name not in _CACHE:
        _CACHE[name] = build_poker(rules_for(name))
    return _CACHE[name]
