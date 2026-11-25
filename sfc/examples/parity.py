"""Cross-channel parity and Gaussian perturbation (SFC, Sect. 5.6/5.7).

Reproduces the counts of Sect. 5.6. At sigma = 0 the sweep gives 384 exact
positives (4 positive records x 6 amplitudes x 16 draws) and 192 certified
firings (the same 4 records, certified at amplitudes 0.5, 0.8 and 1.0, x 16
draws), and no negative-parity record changes sign.

    python examples/parity.py
"""

from __future__ import annotations

import numpy as np

from sfc.monitor import rho, rtilde
from sfc.parity import context, parity_dataset, parity_rule
from sfc.until_scan import until_direct, until_scan

BETA = 8.0
AMPLITUDES = (0.10, 0.20, 0.30, 0.50, 0.80, 1.00)
DRAWS = 16


def main() -> None:
    rule = parity_rule()

    print("exact robustness by amplitude (48 records = 8 phases x 6 amplitudes)")
    for amplitude in AMPLITUDES:
        X, y, _ = parity_dataset(amplitude)
        values = np.array([rho(rule, context(x), 0) for x in X])
        correct = np.mean((values > 0) == (y == 1))
        certified = int(np.sum([rtilde(rule, context(x), 0, BETA) > 0 for x in X]))
        print(
            f"  a={amplitude:<4.2f}  exact in [{values.min():+.2f}, {values.max():+.2f}]"
            f"  accuracy {100 * correct:5.1f}%  certified {certified}/8"
        )

    print("\nGaussian perturbation (16 shared noise draws per amplitude)")
    print("  sigma  exact+  certified  agreement  false  label flips")
    for sigma in (0.0, 0.05, 0.10, 0.20):
        rng = np.random.default_rng(11)
        exact_pos = certified = agree = false = flips = total = 0
        for amplitude in AMPLITUDES:
            X, y, _ = parity_dataset(amplitude)
            for _ in range(DRAWS):
                noise = rng.normal(size=X.shape[1:]) * sigma
                for x, label in zip(X, y):
                    ctx = context(x + noise)
                    exact = rho(rule, ctx, 0)
                    lower = rtilde(rule, ctx, 0, BETA)
                    total += 1
                    exact_pos += int(exact > 0)
                    certified += int(lower > 0)
                    agree += int((exact > 0) == (lower > 0))
                    false += int(lower > 0 and exact <= 0)
                    flips += int(label == 0 and exact > 0)
        print(
            f"  {sigma:<6.2f} {exact_pos:6d}  {certified:9d}  "
            f"{100 * agree / total:8.1f}%  {false:5d}  {flips:10d}"
        )

    print("\nUntil microbenchmark (exact vs linear-time scan, beta = 8)")
    rng = np.random.default_rng(0)
    for horizon in (8, 16, 32, 64, 128, 256):
        worst = 0.0
        for _ in range(24):
            left = rng.uniform(-1, 1, horizon)
            right = rng.uniform(-1, 1, horizon)
            exact = until_direct(left, right, 0, horizon - 1)
            lower = until_scan(left, right, 0, horizon - 1, BETA)
            worst = max(worst, exact - lower)
        print(f"  T={horizon:<4d} max exact-minus-smooth gap {worst:.4f}  (bound {2 * np.log(horizon) / BETA:.4f})")


if __name__ == "__main__":
    main()
