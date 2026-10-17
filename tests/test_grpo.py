from __future__ import annotations

import unittest

import numpy as np

from nashtune.games import make_game
from nashtune.grpo import (
    Adam,
    BatchBuilder,
    expected_advantages,
    group_advantages,
    kl_categorical,
    log_softmax,
    logits_grad,
    loss_and_grad,
)
from nashtune.policy.tabular import TabularPolicy


def random_batch(rng, sizes, kl=0.1, ce=0.0, eps=0.2, spread=0.3):
    builder = BatchBuilder(sizes, eps)
    for i, n in enumerate(sizes):
        builder.set_old(i, log_softmax(rng.normal(size=n)))
        builder.set_ref(i, log_softmax(rng.normal(size=n)))
        for _ in range(6):
            builder.add_sample(i, int(rng.integers(n)), float(rng.normal()), float(rng.random()))
        builder.add_kl(i, kl)
        if ce:
            builder.add_ce(i, rng.dirichlet(np.ones(n)), ce)
    batch = builder.build()
    # Current log-probs near the old ones so some ratios fall inside and some outside the clip range.
    logs = [log_softmax(lp + rng.normal(scale=spread, size=len(lp))) for lp in batch.old_logp]
    return batch, logs


def per_sample_loss(logs, samples, old, eps):
    """The textbook clipped surrogate, one sample at a time."""
    total = 0.0
    for i, a, adv, w in samples:
        ratio = np.exp(logs[i][a] - old[i][a])
        total -= w * min(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)
    return total


class TestAdvantages(unittest.TestCase):
    def test_group_normalization(self):
        adv, used = group_advantages(np.array([1.0, 2.0, 3.0, 6.0]))
        self.assertTrue(used)
        self.assertAlmostEqual(adv.mean(), 0.0, places=12)
        self.assertAlmostEqual(adv.std(), 1.0, places=5)
        expected = (np.array([1, 2, 3, 6]) - 3.0) / (np.std([1, 2, 3, 6]) + 1e-6)
        np.testing.assert_allclose(adv, expected)

    def test_zero_variance_group_is_skipped(self):
        adv, used = group_advantages(np.array([0.5, 0.5, 0.5]))
        self.assertFalse(used)
        np.testing.assert_array_equal(adv, 0.0)

    def test_unnormalized(self):
        adv, used = group_advantages(np.array([1.0, 3.0]), norm="none")
        np.testing.assert_allclose(adv, [-1.0, 1.0])

    def test_expected_advantages_are_the_large_group_limit(self):
        rng = np.random.default_rng(0)
        q = np.array([-1.0, 0.5, 2.0])
        pi = np.array([0.2, 0.5, 0.3])
        exact, used = expected_advantages(q, pi)
        self.assertTrue(used)
        actions = rng.choice(3, size=400_000, p=pi)
        sampled, _ = group_advantages(q[actions])
        for a in range(3):
            self.assertAlmostEqual(sampled[actions == a][0], exact[a], places=2)
        # Expectation under pi is zero; variance under pi is one (up to the 1e-6).
        self.assertAlmostEqual(float(pi @ exact), 0.0, places=12)
        self.assertAlmostEqual(float(pi @ exact**2), 1.0, places=5)
        self.assertFalse(expected_advantages(q, np.array([0.0, 1.0, 0.0]))[1])


class TestClippedSurrogate(unittest.TestCase):
    def test_aggregated_loss_equals_per_sample_loss(self):
        rng = np.random.default_rng(1)
        sizes = [2, 3, 3, 2]
        eps = 0.2
        builder = BatchBuilder(sizes, eps)
        samples, old = [], []
        for i, n in enumerate(sizes):
            old.append(log_softmax(rng.normal(size=n)))
            builder.set_old(i, old[-1])
            for _ in range(10):
                s = (i, int(rng.integers(n)), float(rng.normal()), float(rng.random()))
                samples.append(s)
                builder.add_sample(*s)
        batch = builder.build()
        for spread in (0.0, 0.1, 0.5, 2.0):
            logs = [log_softmax(lp + rng.normal(scale=spread, size=len(lp))) for lp in old]
            loss, _, _ = loss_and_grad(logs, batch)
            self.assertAlmostEqual(loss, per_sample_loss(logs, samples, old, eps), places=10)

    def test_clipping_cases(self):
        builder = BatchBuilder([2], 0.2)
        builder.set_old(0, np.log([0.5, 0.5]))
        builder.add_sample(0, 0, 1.0, 1.0)  # positive advantage on action 0
        builder.add_sample(0, 1, -1.0, 1.0)  # negative advantage on action 1
        batch = builder.build()
        # ratio 1.5 on action 0 (clipped: no gradient), ratio 0.5 on action 1 (clipped: no gradient)
        logs = [np.log([0.75, 0.25])]
        loss, grads, stats = loss_and_grad(logs, batch)
        self.assertAlmostEqual(loss, -(1.2 * 1.0 + 0.8 * -1.0))
        np.testing.assert_allclose(grads[0], [0.0, 0.0])
        self.assertAlmostEqual(stats["clip_frac"], 1.0)
        # Moving the wrong way is never clipped: ratio 0.5 on the positive action keeps its gradient.
        logs = [np.log([0.25, 0.75])]
        loss, grads, _ = loss_and_grad(logs, batch)
        self.assertAlmostEqual(loss, -(0.5 * 1.0 + 1.5 * -1.0))
        np.testing.assert_allclose(grads[0], [-0.5, 1.5])

    def test_exact_kl(self):
        lp = np.log([0.7, 0.2, 0.1])
        ref = np.log([0.2, 0.3, 0.5])
        self.assertAlmostEqual(kl_categorical(lp, ref), float(np.sum(np.exp(lp) * (lp - ref))))
        self.assertAlmostEqual(kl_categorical(ref, ref), 0.0)

    def test_gradient_wrt_logp_by_finite_differences(self):
        rng = np.random.default_rng(2)
        batch, logs = random_batch(rng, [2, 3, 3], kl=0.3, ce=0.5)
        _, grads, _ = loss_and_grad(logs, batch)
        h = 1e-6
        for j, lp in enumerate(logs):
            for a in range(len(lp)):
                plus = [x.copy() for x in logs]
                minus = [x.copy() for x in logs]
                plus[j][a] += h
                minus[j][a] -= h
                fd = (loss_and_grad(plus, batch)[0] - loss_and_grad(minus, batch)[0]) / (2 * h)
                self.assertAlmostEqual(grads[j][a], fd, places=5)


class TestTabularGradient(unittest.TestCase):
    def test_logit_gradient_check(self):
        g = make_game("kuhn")
        rng = np.random.default_rng(3)
        policy = TabularPolicy(g, init="random", seed=3)
        sizes = [i.action_count for i in g.infosets]
        builder = BatchBuilder(sizes, 0.2)
        for i in range(g.num_infosets):
            n = sizes[i]
            builder.set_old(i, log_softmax(policy.logits[g.infosets[i].slots] + rng.normal(scale=0.2, size=n)))
            builder.set_ref(i, log_softmax(rng.normal(size=n)))
            builder.add_sample(i, int(rng.integers(n)), float(rng.normal()), 0.3)
            builder.add_kl(i, 0.2)
            builder.add_ce(i, rng.dirichlet(np.ones(n)), 0.1)
        batch = builder.build()
        _, grad, _ = policy.loss_and_logit_grad(batch)
        base = policy.logits.copy()
        h = 1e-6
        for k in range(g.action_slots):
            policy.logits = base.copy()
            policy.logits[k] += h
            up = policy.loss_and_logit_grad(batch)[0]
            policy.logits = base.copy()
            policy.logits[k] -= h
            down = policy.loss_and_logit_grad(batch)[0]
            self.assertAlmostEqual(grad[k], (up - down) / (2 * h), places=5)
        policy.logits = base

    def test_logits_grad_chain_rule(self):
        theta = np.array([0.3, -1.0, 2.0])
        lp = log_softmax(theta)
        g = np.array([0.5, -0.2, 0.1])
        out = logits_grad(g, lp)
        p = np.exp(lp)
        jac = np.eye(3) - p[None, :]  # d lp_a / d theta_b
        np.testing.assert_allclose(out, jac.T @ g)

    def test_adam_matches_reference_step(self):
        params = np.array([1.0, -2.0])
        opt = Adam(2, lr=0.1)
        opt.step(params, np.array([0.5, -0.5]))
        # First Adam step moves each coordinate by lr * sign(grad).
        np.testing.assert_allclose(params, [0.9, -1.9], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
