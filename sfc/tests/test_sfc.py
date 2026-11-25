"""Unit tests for the SFC implementation (standard library only)."""

from __future__ import annotations

import math
import unittest

import numpy as np

import sfc
from sfc.codebook import SharedCodebook
from sfc.monitor import (
    And,
    Atom,
    Eventually,
    Globally,
    Or,
    Run,
    Trend,
    Until,
    gap_bound,
    levels,
    rho,
    rtilde,
)
from sfc.parity import context, parity_dataset, parity_rule
from sfc.relax import smax, smin
from sfc.until_scan import until_direct, until_scan


class OneSidedBounds(unittest.TestCase):
    def test_lemma_1(self) -> None:
        rng = np.random.default_rng(0)
        beta = 8.0
        for n in (1, 2, 3, 5, 17):
            for _ in range(50):
                u = rng.normal(size=n)
                self.assertLessEqual(smin(u, beta), u.min() + 1e-12)
                self.assertGreaterEqual(smin(u, beta), u.min() - math.log(n) / beta - 1e-12)
                self.assertLessEqual(smax(u, beta), u.max() + 1e-12)
                self.assertGreaterEqual(smax(u, beta), u.max() - math.log(n) / beta - 1e-12)


class MonitorSemantics(unittest.TestCase):
    def setUp(self) -> None:
        self.beta = 8.0
        self.x = np.array([0.0, 1.0, 2.0, 3.0, 0.0, -1.0, 0.5])
        self.ctx = {"x": self.x}

    def test_atom_sign(self) -> None:
        self.assertAlmostEqual(rho(Atom("x", "<", 1.0), self.ctx, 0), 1.0)
        self.assertAlmostEqual(rho(Atom("x", ">", 1.0), self.ctx, 2), 1.0)

    def test_globally_is_minimum(self) -> None:
        node = Globally(0, 3, Atom("x", ">", -0.5))
        self.assertAlmostEqual(rho(node, self.ctx, 0), 0.5)  # x[0] = 0.0 -> 0.5
        self.assertAlmostEqual(rho(node, self.ctx, 0), min(self.x[0:4] - (-0.5)))

    def test_eventually_is_maximum(self) -> None:
        node = Eventually(0, 3, Atom("x", ">", 2.5))
        self.assertAlmostEqual(rho(node, self.ctx, 0), 0.5)

    def test_until_matches_definition(self) -> None:
        left = Atom("x", "<", 10.0)
        right = Atom("x", ">", 2.5)
        node = Until(0, 5, left, right)
        best = max(
            min(rho(right, self.ctx, s), min(rho(left, self.ctx, p) for p in range(0, s)) if s > 0 else math.inf)
            for s in range(0, 6)
        )
        self.assertAlmostEqual(rho(node, self.ctx, 0), best)

    def test_run_requires_consecutive_samples(self) -> None:
        node = Run(0, 4, 3, Atom("x", ">", 0.5))
        # samples 1,2,3 are 1.0, 2.0, 3.0 -> three consecutive
        self.assertGreater(rho(node, self.ctx, 0), 0)
        # only samples 2,3,4: 4 is 0.0 -> no run of three starting at 2
        self.assertLess(rho(node, self.ctx, 2), 0)

    def test_trend_net_change_and_excess(self) -> None:
        node = Trend("x", 0, 3, delta=0.5, epsilon=0.1)
        ctx = {"x": np.array([0.0, 1.0, 2.0, 3.0])}
        # net change 3.0 - 0.5 = 2.5; largest excess above the final value is -1.0
        self.assertAlmostEqual(rho(node, ctx, 0), min(2.5, 0.1 - (-1.0)))

    def test_globally_leaves_a_gap(self) -> None:
        node = Globally(0, 3, Atom("x", ">", -0.5))
        lower = rtilde(node, self.ctx, 0, self.beta)
        self.assertLessEqual(lower, rho(node, self.ctx, 0) + 1e-12)
        self.assertLessEqual(rho(node, self.ctx, 0) - lower, gap_bound(node, self.beta, len(self.x)) + 1e-12)


class UntilScan(unittest.TestCase):
    def test_scan_is_sound_and_tight(self) -> None:
        rng = np.random.default_rng(1)
        for horizon in (8, 16, 64, 256):
            for _ in range(8):
                left = rng.uniform(-1, 1, horizon)
                right = rng.uniform(-1, 1, horizon)
                for a, b in ((0, horizon - 1), (horizon // 2, horizon - 1), (0, 0)):
                    exact = until_direct(left, right, a, b)
                    lower = until_scan(left, right, a, b, 8.0)
                    self.assertLessEqual(lower, exact + 1e-12)
                    self.assertLessEqual(exact - lower, 2 * math.log(horizon) / 8.0 + 1e-12)


class Parity(unittest.TestCase):
    def test_rule_is_well_typed(self) -> None:
        self.assertTrue(sfc.well_typed(parity_rule()))
        self.assertEqual(levels(parity_rule()), 3)

    def test_exact_robustness_is_plus_minus_amplitude(self) -> None:
        rule = parity_rule()
        for amplitude in (0.1, 0.5, 1.0):
            X, y, _ = parity_dataset(amplitude)
            for x, label in zip(X, y):
                value = rho(rule, context(x), 0)
                self.assertAlmostEqual(value, amplitude if label == 1 else -amplitude)

    def test_all_eight_records_are_correct(self) -> None:
        rule = parity_rule()
        X, y, _ = parity_dataset(0.5)
        predicted = [rho(rule, context(x), 0) > 0 for x in X]
        self.assertEqual(sum(int(p == (label == 1)) for p, label in zip(predicted, y)), 8)

    def test_certified_count_matches_the_theorem(self) -> None:
        rule = parity_rule()
        certified = 0
        for amplitude in (0.1, 0.2, 0.3, 0.5, 0.8, 1.0):
            X, y, _ = parity_dataset(amplitude)
            for x, label in zip(X, y):
                if rtilde(rule, context(x), 0, 8.0) > 0:
                    certified += 1
                    self.assertEqual(label, 1)  # no false certificate
        self.assertEqual(certified, 12)  # four positive records at amplitudes >= 0.5


class Codebook(unittest.TestCase):
    def test_centres_are_strictly_increasing(self) -> None:
        book = SharedCodebook(0.0, 1.0, np.array([0.3, -0.2, 0.9, 0.1]))
        self.assertTrue(book.ordered())
        self.assertAlmostEqual(book.centres()[-1], 1.0)

    def test_constant_signal(self) -> None:
        book = SharedCodebook(2.0, 2.0, np.zeros(4))
        self.assertTrue(np.allclose(book.centres(), 2.0))


class Soundness(unittest.TestCase):
    """Theorem 1 on random formulas: a positive lower bound implies a positive exact score."""

    def test_random_formulas(self) -> None:
        rng = np.random.default_rng(3)
        beta = 8.0
        violations = 0
        for _ in range(120):
            x = rng.normal(size=32)
            ctx = {"x": x}
            node = self._random_formula(rng, depth=3)
            exact = rho(node, ctx, 0)
            lower = rtilde(node, ctx, 0, beta)
            if lower > exact + 1e-9:
                violations += 1
            if lower > 0 and exact <= 0:
                violations += 1
            if exact - lower > gap_bound(node, beta, 32) + 1e-9 and math.isfinite(exact):
                violations += 1
        self.assertEqual(violations, 0)

    def _random_formula(self, rng, depth: int):
        if depth <= 1:
            return Atom("x", rng.choice(["<", ">"]), float(rng.normal()))
        kind = rng.integers(0, 5)
        lo = int(rng.integers(0, 3))
        hi = lo + int(rng.integers(0, 4))
        if kind == 0:
            return And(tuple(self._random_formula(rng, depth - 1) for _ in range(int(rng.integers(2, 4)))))
        if kind == 1:
            return Or(tuple(self._random_formula(rng, depth - 1) for _ in range(int(rng.integers(2, 4)))))
        if kind == 2:
            return Globally(lo, hi, self._random_formula(rng, depth - 1))
        if kind == 3:
            return Eventually(lo, hi, self._random_formula(rng, depth - 1))
        return Until(lo, hi, self._random_formula(rng, depth - 1), self._random_formula(rng, depth - 1))


if __name__ == "__main__":
    unittest.main()
