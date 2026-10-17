from __future__ import annotations

import unittest

from nashtune.games import make_game
from nashtune.prompts import ACTION_WORDS, parse_action_word, render, render_all, replay

KUHN_A_SYSTEM = (
    "You are playing Kuhn poker against one opponent. The deck has three cards, from lowest to highest: "
    "Jack, Queen and King. Each player antes 1 chip and is dealt one private card. There is one betting "
    "round, and the first player acts first. A bet is 1 chip, and at most one bet is allowed per round. "
    "If nobody folds, the higher card wins the pot. Answer with exactly one word."
)
LEDUC_A_SYSTEM = (
    "You are playing Leduc poker against one opponent. The deck has 6 cards, two of each rank, from lowest "
    "to highest: Jack, Queen and King. Each player antes 1 chip and is dealt one private card. There are two "
    "betting rounds, and one public card is revealed after the first. The first player acts first in each "
    "round. A bet or raise is 2 chips in round 1 and 4 chips in round 2, and each round allows at most two "
    "bets (a bet and a raise). If nobody folds, a card that pairs the public card wins; otherwise the higher "
    "card wins; equal cards split the pot. Answer with exactly one word."
)
LEDUC_B_SYSTEM = (
    "Game: Leduc poker, heads-up. Cards: J, Q, K (two of each), ranked low to high. Everyone posts 1 chip "
    "before the deal and receives one hidden card. Betting happens twice, and a shared board card is turned "
    "face up between the two rounds. The first seat speaks first in both rounds. Bet size: 2 chips before "
    "the board card, 4 after; a round may contain at most two bets (one bet and one raise). At the end, "
    "pairing the board card wins; without a pair the higher card wins; identical ranks split. Reply with a "
    "single word."
)

GOLDEN_USER = {
    ("kuhn", "1:Q:-:b", "A"): "You are the second player. Your card: Queen.\nBetting: Opponent bet 1.\n"
    "Pot: 3 chips. To call: 1 chip.\nLegal actions: fold, call.\nYour action?",
    ("kuhn", "1:Q:-:b", "B"): "Choose one of: call, fold.\nSo far: your opponent bets 1.\n"
    "You hold the Q. You are in the second seat.\nThere are 3 chips in the middle, and staying in costs 1.",
    ("leduc", "0:K:J:kbc/kb", "A"): "You are the first player. Your card: King. Public card: Jack.\n"
    "Round 1: You checked. Opponent bet 2. You called.\nRound 2: You checked. Opponent bet 4.\n"
    "Pot: 10 chips. To call: 4 chips.\nLegal actions: fold, call, raise.\nYour action?",
    ("leduc", "0:K:J:kbc/kb", "B"): "Choose one of: raise, call, fold.\nSo far: before the board card, you "
    "check, your opponent bets 2, you call; after the board card, you check, your opponent bets 4.\n"
    "You hold the K; the board card is the J. You are in the first seat.\n"
    "There are 10 chips in the middle, and staying in costs 4.",
    ("leduc4", "1:A:-:k", "A"): "You are the second player. Your card: Ace. Public card: not dealt yet.\n"
    "Round 1: Opponent checked.\nPot: 2 chips. To call: 0 chips.\nLegal actions: check, bet.\nYour action?",
    ("leduc4", "1:A:-:k", "B"): "Choose one of: bet, check.\nSo far: before the board card, your opponent "
    "checks.\nYou hold the A; the board card is not out yet. You are in the second seat.\n"
    "There are 2 chips in the middle, and nothing to call.",
}


class TestPrompts(unittest.TestCase):
    def test_golden_system_messages(self):
        self.assertEqual(render(make_game("kuhn"), 0, "A").system, KUHN_A_SYSTEM)
        self.assertEqual(render(make_game("leduc"), 0, "A").system, LEDUC_A_SYSTEM)
        self.assertEqual(render(make_game("leduc"), 0, "B").system, LEDUC_B_SYSTEM)

    def test_golden_user_messages(self):
        for (game, key, template), expected in GOLDEN_USER.items():
            with self.subTest(game=game, key=key, template=template):
                self.assertEqual(render(make_game(game), key, template).user, expected)

    def test_every_infoset_renders_in_both_templates(self):
        for name in ("kuhn", "leduc", "leduc4", "leduc5"):
            g = make_game(name)
            for template in ("A", "B"):
                prompts = render_all(g, template)
                self.assertEqual(len(prompts), g.num_infosets)
                self.assertEqual(len({p.user for p in prompts}), g.num_infosets)  # all distinct
                for info, p in zip(g.infosets, prompts):
                    self.assertEqual(p.actions, tuple(ACTION_WORDS[a] for a in info.actions))
                    for word in p.actions:
                        self.assertIn(word, p.user)
                    self.assertLess(len(p.user), 400)
                    self.assertLess(len(p.system), 700)

    def test_templates_differ_and_are_deterministic(self):
        g = make_game("leduc")
        for info in g.infosets:
            a, b = render(g, info, "A"), render(g, info, "B")
            self.assertNotEqual(a.user, b.user)
            self.assertEqual(a, render(g, info, "A"))

    def test_replay_amounts(self):
        g = make_game("leduc")
        sit = replay(g.rules, "brc/br", 0)
        self.assertEqual(sit.pot, 1 + 1 + 4 + 4 + 4 + 8)
        self.assertEqual(sit.to_call, 4)
        self.assertEqual([s.amount for s in sit.rounds[0]], [2, 4, 2])
        with self.assertRaises(ValueError):
            replay(g.rules, "b", 0)  # player 1 is to act after one action

    def test_parse_action_word(self):
        legal = ("fold", "call", "raise")
        self.assertEqual(parse_action_word("Call.", legal), "call")
        self.assertEqual(parse_action_word("**raise**", legal), "raise")
        self.assertEqual(parse_action_word("  fold\n", legal), "fold")
        self.assertIsNone(parse_action_word("I call", legal))
        self.assertIsNone(parse_action_word("check", legal))
        self.assertIsNone(parse_action_word("", legal))

    def test_unknown_template(self):
        with self.assertRaises(ValueError):
            render(make_game("kuhn"), 0, "C")


if __name__ == "__main__":
    unittest.main()
