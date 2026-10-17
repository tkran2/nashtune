from __future__ import annotations

import itertools
import unittest

import numpy as np

from nashtune.games import best_response, counterfactual_values, exploitability, make_game, own_reach
from nashtune.games.analysis import expected_value


def random_profile(game, rng, floor=0.02):
    return game.normalize(rng.random(game.action_slots) + floor)


def player_value(game, profile, player):
    v = expected_value(game, profile)
    return v if player == 0 else -v


class TestBestResponse(unittest.TestCase):
    def test_best_response_is_optimal_by_brute_force_on_kuhn(self):
        """Enumerate all 64 pure strategies of the responder."""
        g = make_game("kuhn")
        rng = np.random.default_rng(0)
        for trial in range(5):
            profile = random_profile(g, rng)
            if trial == 4:  # ties everywhere: the uniform strategy
                profile = g.uniform_strategy()
            for me in (0, 1):
                value, br = best_response(g, profile, me)
                mine = g.player_infosets(me)
                best = -np.inf
                for choice in itertools.product(*[range(i.action_count) for i in mine]):
                    pure = profile.copy()
                    for info, a in zip(mine, choice):
                        pure[info.offset : info.offset + info.action_count] = 0.0
                        pure[info.offset + a] = 1.0
                    best = max(best, player_value(g, pure, me))
                self.assertAlmostEqual(value, best, places=12)
                # The returned strategy achieves the stated value and leaves the opponent untouched.
                self.assertAlmostEqual(player_value(g, br, me), value, places=12)
                other = g.slot_player != me
                np.testing.assert_array_equal(br[other], profile[other])

    def test_best_response_is_never_beaten_on_leduc(self):
        g = make_game("leduc")
        rng = np.random.default_rng(1)
        profile = random_profile(g, rng)
        for me in (0, 1):
            value, br = best_response(g, profile, me)
            self.assertAlmostEqual(player_value(g, br, me), value, places=10)
            for _ in range(20):
                other = g.with_player(profile, random_profile(g, rng, floor=0.0), me)
                self.assertLessEqual(player_value(g, other, me), value + 1e-12)

    def test_exploitability_nonnegative_and_brackets(self):
        rng = np.random.default_rng(2)
        for name in ("kuhn", "leduc"):
            g = make_game(name)
            for _ in range(5):
                e = exploitability(g, random_profile(g, rng))
                self.assertGreater(e.exploitability, 0)
                self.assertLessEqual(e.value_lower, e.value)
                self.assertLessEqual(e.value, e.value_upper)


class TestCounterfactualValues(unittest.TestCase):
    def test_finite_difference_against_a_fixed_opponent(self):
        """Changing the policy at I by eps*d changes the player's value by
        eps * own_reach(I) * sum_a d_a q(I, a), exactly (the value is linear in
        the policy at one infoset under perfect recall)."""
        g = make_game("leduc")
        rng = np.random.default_rng(4)
        profile = random_profile(g, rng)
        for player in (0, 1):
            cf = counterfactual_values(g, profile, player)
            reach = own_reach(g, profile, player)
            base = player_value(g, profile, player)
            self.assertAlmostEqual(cf.value, base, places=12)
            for info in rng.choice(g.player_infosets(player), size=15, replace=False):
                d = rng.normal(size=info.action_count)
                d -= d.mean()
                eps = 1e-3
                moved = profile.copy()
                moved[info.offset : info.offset + info.action_count] += eps * d
                predicted = eps * reach[info.index] * float(d @ cf.q[info.offset : info.offset + info.action_count])
                self.assertAlmostEqual(player_value(g, moved, player) - base, predicted, places=12)

    def test_danskin_gradient_of_worst_case_value(self):
        """Against the opponent's best response, the same q gives the first-order
        change of the player's worst-case value (the best response does not move
        for a small enough perturbation)."""
        g = make_game("kuhn")
        rng = np.random.default_rng(5)
        profile = random_profile(g, rng)
        for player in (0, 1):
            _, br_profile = best_response(g, profile, 1 - player)
            cf = counterfactual_values(g, br_profile, player)
            reach = own_reach(g, profile, player)

            def worst_case(p):
                return -best_response(g, p, 1 - player)[0]

            base = worst_case(profile)
            for info in g.player_infosets(player):
                d = np.array([1.0, -1.0])
                eps = 1e-7
                moved = profile.copy()
                moved[info.offset : info.offset + 2] += eps * d
                fd = (worst_case(moved) - base) / eps
                exact = reach[info.index] * float(d @ cf.q[info.offset : info.offset + 2])
                self.assertAlmostEqual(fd, exact, places=6)

    def test_conditional_values(self):
        g = make_game("leduc")
        rng = np.random.default_rng(6)
        profile = random_profile(g, rng)
        cf = counterfactual_values(g, profile, 1)
        q, defined = cf.conditional(g)
        self.assertTrue(np.all(defined[g.infoset_player == 1]))
        self.assertFalse(np.any(defined[g.infoset_player == 0]))
        for info in g.player_infosets(1)[:20]:
            s = slice(info.offset, info.offset + info.action_count)
            np.testing.assert_allclose(q[s] * cf.cf_reach[info.index], cf.q[s], rtol=1e-12, atol=1e-15)
        # Payoffs in Leduc are bounded by 13 chips.
        self.assertLessEqual(np.abs(q).max(), 13.0)


if __name__ == "__main__":
    unittest.main()
