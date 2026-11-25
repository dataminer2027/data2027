"""Conservative relaxations (SFC, Sect. 4.3, Definitions 3 and Lemma 1).

    smin_beta(u) = -(1/beta) log sum_i exp(-beta u_i)          <= min_i u_i
    smax_beta(u) =  (1/beta) ( log sum_i exp(beta u_i) - log n ) <= max_i u_i

Both are monotone non-decreasing and differentiable, and both are lower bounds:
the negative log-sum-exp needs no correction, while the unnormalised
log-sum-exp would be an *upper* bound on the max, which is why the ``-log n``
term in ``smax`` is essential.
"""

from __future__ import annotations

import numpy as np

__all__ = ["logsumexp", "smin", "smax", "smin_scalar", "smax_scalar", "gap"]


def logsumexp(values: np.ndarray, sign: float = 1.0) -> float:
    a = sign * np.asarray(values, dtype=float)
    if a.size == 0:
        return -np.inf
    m = float(np.max(a))
    if not np.isfinite(m):
        return m
    return float(m + np.log(np.sum(np.exp(a - m))))


def smin(values, beta: float) -> float:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return np.inf
    return float(-logsumexp(beta * a, -1.0) / beta)


def smax(values, beta: float) -> float:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return -np.inf
    return float((logsumexp(beta * a, 1.0) - np.log(a.size)) / beta)


def smin_scalar(u: float, beta: float) -> float:
    return float(u)


def smax_scalar(u: float, beta: float) -> float:
    return float(u)


def gap(n_terms: int, beta: float) -> float:
    """Per-reduction gap ``log n / beta`` of Lemma 1."""
    if n_terms <= 1:
        return 0.0
    return float(np.log(n_terms) / beta)
