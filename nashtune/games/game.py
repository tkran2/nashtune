"""A two-player zero-sum game stored as an explicit tree, plus the flat arrays
that the vectorized analysis code in ``analysis.py`` works on.

The layout follows ZeroSum (github.com/tkran2/zerosum) exactly: nodes are
numbered in the order the builder creates them, information sets in the order
they are first reached, and a strategy is one probability per (infoset,
action) pair laid out by ``Infoset.offset``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CHANCE, DECISION, TERMINAL = 0, 1, 2


@dataclass(frozen=True)
class Infoset:
    """Everything one player cannot tell apart: same private card, same public
    card, same betting. The player must act the same way at every node in it."""

    index: int
    player: int
    key: str  # ZeroSum format, e.g. "0:K:J:kb/b"
    actions: str  # one letter per action: k check, b bet, f fold, c call, r raise
    offset: int  # start of this infoset's slice in per-action arrays
    private_rank: int
    public_rank: int  # -1 before the public card is dealt
    history: str

    @property
    def action_count(self) -> int:
        return len(self.actions)

    @property
    def slots(self) -> range:
        return range(self.offset, self.offset + len(self.actions))

    @property
    def round(self) -> int:
        return self.history.count("/")


@dataclass
class Game:
    name: str
    rules: object
    node_type: np.ndarray  # int8
    node_player: np.ndarray  # int8, -1 unless decision
    node_infoset: np.ndarray  # int32, -1 unless decision
    first_child: np.ndarray  # int32
    child_count: np.ndarray  # int32
    utility: np.ndarray  # float64, payoff to player 0 at terminals
    children: np.ndarray  # int32
    chance_prob: np.ndarray  # float64, parallel to children
    infosets: list[Infoset]
    action_slots: int
    rank_names: list[str]
    # Derived arrays, filled by compile().
    parent: np.ndarray = field(init=False, repr=False)
    depth: np.ndarray = field(init=False, repr=False)
    edge_slot: np.ndarray = field(init=False, repr=False)  # slot of the action that led here, -1 if chance
    edge_prob: np.ndarray = field(init=False, repr=False)  # chance probability that led here (1 if not chance)
    parent_player: np.ndarray = field(init=False, repr=False)  # player who acted to get here, -1 if chance
    by_depth: list[np.ndarray] = field(init=False, repr=False)
    slot_infoset: np.ndarray = field(init=False, repr=False)  # infoset index of each action slot
    slot_action: np.ndarray = field(init=False, repr=False)  # action position within its infoset
    slot_player: np.ndarray = field(init=False, repr=False)
    infoset_player: np.ndarray = field(init=False, repr=False)
    infoset_depth: np.ndarray = field(init=False, repr=False)
    action_matrix: np.ndarray = field(init=False, repr=False)  # [n_infosets, max_actions] slots, -1 padded
    terminals: np.ndarray = field(init=False, repr=False)
    decision_nodes: np.ndarray = field(init=False, repr=False)
    player_edges: list[np.ndarray] = field(init=False, repr=False)
    player_nodes: list[np.ndarray] = field(init=False, repr=False)
    infosets_at: dict = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.compile()

    # ------------------------------------------------------------------ basics
    @property
    def num_nodes(self) -> int:
        return int(self.node_type.shape[0])

    @property
    def num_infosets(self) -> int:
        return len(self.infosets)

    def child(self, node: int, action: int) -> int:
        return int(self.children[self.first_child[node] + action])

    def infoset_by_key(self, key: str) -> Infoset:
        return self.infosets[self._key_index[key]]

    def player_infosets(self, player: int) -> list[Infoset]:
        return [info for info in self.infosets if info.player == player]

    def uniform_strategy(self) -> np.ndarray:
        s = np.empty(self.action_slots)
        for info in self.infosets:
            s[info.offset : info.offset + info.action_count] = 1.0 / info.action_count
        return s

    def counts(self) -> dict[str, int]:
        return {
            "nodes": self.num_nodes,
            "chance": int(np.sum(self.node_type == CHANCE)),
            "decision": int(np.sum(self.node_type == DECISION)),
            "terminal": int(np.sum(self.node_type == TERMINAL)),
            "infosets": self.num_infosets,
            "infosets_player0": sum(1 for i in self.infosets if i.player == 0),
            "infosets_player1": sum(1 for i in self.infosets if i.player == 1),
            "actions": self.action_slots,
        }

    # ----------------------------------------------------------------- compile
    def compile(self) -> None:
        n = self.num_nodes
        parent = np.full(n, -1, dtype=np.int64)
        edge_slot = np.full(n, -1, dtype=np.int64)
        edge_prob = np.ones(n)
        parent_player = np.full(n, -1, dtype=np.int64)
        for node in range(n):
            if self.node_type[node] == TERMINAL:
                continue
            start = int(self.first_child[node])
            for a in range(int(self.child_count[node])):
                child = int(self.children[start + a])
                if child <= node:
                    raise ValueError("children must be numbered after their parent")
                parent[child] = node
                if self.node_type[node] == CHANCE:
                    edge_prob[child] = self.chance_prob[start + a]
                else:
                    info = self.infosets[int(self.node_infoset[node])]
                    edge_slot[child] = info.offset + a
                    parent_player[child] = int(self.node_player[node])
        depth = np.zeros(n, dtype=np.int64)
        for node in range(1, n):  # parents are numbered before children
            depth[node] = depth[parent[node]] + 1
        self.parent = parent
        self.depth = depth
        self.edge_slot = edge_slot
        self.edge_prob = edge_prob
        self.parent_player = parent_player
        self.by_depth = [np.flatnonzero(depth == d) for d in range(int(depth.max()) + 1)]

        slot_infoset = np.empty(self.action_slots, dtype=np.int64)
        slot_action = np.empty(self.action_slots, dtype=np.int64)
        slot_player = np.empty(self.action_slots, dtype=np.int64)
        width = max(info.action_count for info in self.infosets)
        action_matrix = np.full((self.num_infosets, width), -1, dtype=np.int64)
        for info in self.infosets:
            for a in range(info.action_count):
                slot_infoset[info.offset + a] = info.index
                slot_action[info.offset + a] = a
                slot_player[info.offset + a] = info.player
                action_matrix[info.index, a] = info.offset + a
        self.slot_infoset = slot_infoset
        self.slot_action = slot_action
        self.slot_player = slot_player
        self.action_matrix = action_matrix
        self.infoset_player = np.array([info.player for info in self.infosets], dtype=np.int64)

        # Every node of an infoset sits at the same depth in these games (the
        # key contains the whole public history). The best-response code relies
        # on it, so check it once here.
        infoset_depth = np.full(self.num_infosets, -1, dtype=np.int64)
        decision_nodes = np.flatnonzero(self.node_type == DECISION)
        for node in decision_nodes:
            i = int(self.node_infoset[node])
            if infoset_depth[i] == -1:
                infoset_depth[i] = depth[node]
            elif infoset_depth[i] != depth[node]:
                raise ValueError(f"infoset {self.infosets[i].key} spans several depths")
        self.infoset_depth = infoset_depth
        self.decision_nodes = decision_nodes
        self.terminals = np.flatnonzero(self.node_type == TERMINAL)
        self._key_index = {info.key: info.index for info in self.infosets}
        # Nodes reached by each player's own actions, and each player's decision nodes.
        self.player_edges = [np.flatnonzero(parent_player == p) for p in (0, 1)]
        self.player_nodes = [decision_nodes[self.node_player[decision_nodes] == p] for p in (0, 1)]
        # Infosets of each player at each depth (for the best-response pass).
        self.infosets_at = {
            (int(d), p): np.flatnonzero((infoset_depth == d) & (self.infoset_player == p))
            for d in np.unique(infoset_depth) for p in (0, 1)
        }

    # --------------------------------------------------------------- helpers
    def check_strategy(self, strategy: np.ndarray, atol: float = 1e-6) -> None:
        strategy = np.asarray(strategy, dtype=float)
        if strategy.shape != (self.action_slots,):
            raise ValueError(f"strategy has shape {strategy.shape}, expected ({self.action_slots},)")
        if np.any(strategy < -1e-12) or not np.all(np.isfinite(strategy)):
            raise ValueError("strategy has negative or non-finite entries")
        sums = np.bincount(self.slot_infoset, weights=strategy, minlength=self.num_infosets)
        bad = np.flatnonzero(np.abs(sums - 1.0) > atol)
        if bad.size:
            raise ValueError(f"probabilities at {self.infosets[int(bad[0])].key} sum to {sums[bad[0]]}")

    def normalize(self, strategy: np.ndarray) -> np.ndarray:
        """Rescale each infoset's slice to sum to one."""
        strategy = np.asarray(strategy, dtype=float)
        sums = np.bincount(self.slot_infoset, weights=strategy, minlength=self.num_infosets)
        return strategy / sums[self.slot_infoset]

    def with_player(self, base: np.ndarray, source: np.ndarray, who: int) -> np.ndarray:
        """Take player ``who``'s decisions from ``source`` and the rest from ``base``."""
        out = np.array(base, dtype=float, copy=True)
        mask = self.slot_player == who
        out[mask] = np.asarray(source, dtype=float)[mask]
        return out
