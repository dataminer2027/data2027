"""Rule certificates (SFC, Sect. 4.6, Step 3).

At inference the certificate is the tuple

    (y_hat, phi*, rho, rtilde, delta_gap, [rho_lo, inf))

where ``rho`` is the exact robustness computed by an independent monitor,
``rtilde`` the conservative lower bound and ``delta_gap = k log T / beta`` the
analytical gap of Lemma 2.  ``rho_lo`` is a split-conformal lower bound on the
robustness of the extracted rule for exchangeable observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .monitor import gap_bound, rho, rtilde

__all__ = ["Certificate", "certify", "conformal_quantile", "conformal_lower_bound"]


@dataclass(frozen=True)
class Certificate:
    label: int | None
    formula: object
    exact: float
    lower: float
    gap: float
    interval: tuple[float, float]

    @property
    def certified(self) -> bool:
        return self.lower > 0

    @property
    def sound(self) -> bool:
        return self.lower <= self.exact + 1e-12

    def as_tuple(self) -> tuple:
        return (self.label, self.formula, self.exact, self.lower, self.gap, self.interval)


def certify(
    formula,
    ctx: dict,
    beta: float,
    label: int | None = None,
    origin: int = 0,
    rho_lo: float = -math.inf,
) -> Certificate:
    horizon = len(next(iter(ctx.values())))
    exact = rho(formula, ctx, origin)
    lower = rtilde(formula, ctx, origin, beta)
    return Certificate(
        label=label,
        formula=formula,
        exact=float(exact),
        lower=float(lower),
        gap=gap_bound(formula, beta, horizon),
        interval=(float(rho_lo), math.inf),
    )


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Order statistic of ``-rho`` at rank ``ceil((n+1)(1-alpha))``.

    Returns ``+inf`` when the rank exceeds ``n``, which yields no finite lower
    bound (the standard split-conformal convention for small calibration sets).
    """
    values = np.sort(-np.asarray(scores, dtype=float))
    n = len(values)
    if n == 0:
        return math.inf
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return math.inf
    return float(values[rank - 1])


def conformal_lower_bound(scores: Sequence[float], alpha: float) -> float:
    """``rho_lo = -q_{1-alpha}``; ``-inf`` when the rank exceeds ``n``."""
    q = conformal_quantile(scores, alpha)
    return -q if np.isfinite(q) else -math.inf
