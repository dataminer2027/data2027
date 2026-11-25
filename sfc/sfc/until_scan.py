"""Linear-time sound ``Until`` (SFC, Sect. 4.4, Theorem 2).

For a fixed start ``t`` every eligible release ``s in [t+a, t+b]`` shares the
prefix beginning at ``t``.  An exclusive prefix sum

    C(s) = sum_{p=t}^{s-1} exp(-beta rho1(p)),      C(t) = 0

gives, in log space,

    u(s) = -(1/beta) log( C(s) + exp(-beta rho2(s)) )
    rho(phi1 U_I phi2, t) >= smax_beta{ u(s) : s in t + I }

The scan visits each position once, keeps one accumulator, and adds at most
``2 log T / beta`` to the local gap (one soft minimum over the prefix and one
soft maximum over releases).
"""

from __future__ import annotations

import numpy as np

from .relax import smax

__all__ = ["until_direct", "until_scan", "local_gap"]


def until_direct(left: np.ndarray, right: np.ndarray, a: int, b: int) -> float:
    """Exact ``max_s min{ rho2(s), min_{p in [t, s)} rho1(p) }`` in O(b^2)."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    last = min(b, len(right) - 1, len(left) - 1)
    best = -np.inf
    for s in range(a, last + 1):
        prefix = float(np.min(left[:s])) if s > 0 else np.inf
        best = max(best, min(float(right[s]), prefix))
    return float(best)


def until_scan(left: np.ndarray, right: np.ndarray, a: int, b: int, beta: float) -> float:
    """Sound lower bound on :func:`until_direct` in O(b) time and memory."""
    if beta <= 0:
        raise ValueError("beta must be positive")
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    last = min(b, len(right) - 1, len(left) - 1)
    accumulator = -np.inf  # log C(s); -inf is the empty prefix
    releases: list[float] = []
    for s in range(0, last + 1):
        if s >= a:
            releases.append(float(-np.logaddexp(accumulator, -beta * right[s]) / beta))
        accumulator = float(np.logaddexp(accumulator, -beta * left[s]))
    if not releases:
        return -np.inf
    return smax(np.array(releases), beta)


def local_gap(beta: float, horizon: int) -> float:
    """``2 log T / beta`` for the two reductions of the scan."""
    if horizon <= 0:
        return 0.0
    return float(2.0 * np.log(max(horizon, 2)) / beta)
