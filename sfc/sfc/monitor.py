"""Exact and relaxed STL semantics over typed views (SFC, Sect. 3.2 and 4.3-4.5).

Exact robustness follows the standard quantitative semantics

    rho(pi, t)          = c - z_t   (for  z < c, z <= c),  z_t - c   (for z > c, z >= c)
    rho(phi1 ^ phi2)    = min{rho1, rho2}          rho(phi1 v phi2) = max{rho1, rho2}
    rho(G_I phi, t)     = min_{s in t+I} rho(phi, s)
    rho(F_I phi, t)     = max_{s in t+I} rho(phi, s)
    rho(phi1 U_I phi2,t)= max_{s in t+I} min{ rho2(s), min_{p in [t,s)} rho1(p) }

plus the two interval modalities of Sect. 4.5, ``Run`` (n consecutive satisfied
positions) and ``Trend`` (net change with a tolerance on the excess above the
final value).

The relaxations replace ``min``/``max`` by ``smin``/``smax``, ``Until`` by the
prefix scan of :mod:`sfc.until_scan`, ``Run`` by a soft sliding minimum and
``Trend`` by an upper bound on its inner maximum.  Every replacement is a lower
bound, so ``rtilde <= rho`` and a positive ``rtilde`` certifies the rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .relax import smax, smin
from .until_scan import until_scan

__all__ = [
    "Atom",
    "And",
    "Or",
    "Globally",
    "Eventually",
    "Until",
    "Run",
    "Trend",
    "rho",
    "rtilde",
    "levels",
    "gap_bound",
    "certified",
    "semantic_fidelity",
]


@dataclass(frozen=True)
class Atom:
    """``z^(v)_t  <op>  c`` with a threshold in the units of the view."""

    view: str
    op: str
    c: float
    sort: str = "Time"

    def margin(self, z: np.ndarray, t: int) -> float:
        value = float(z[t])
        if self.op in {"<", "<="}:
            return float(self.c - value)
        if self.op in {">", ">="}:
            return float(value - self.c)
        raise ValueError(f"unknown comparison: {self.op!r}")


@dataclass(frozen=True)
class And:
    children: tuple


@dataclass(frozen=True)
class Or:
    children: tuple


@dataclass(frozen=True)
class Globally:
    lo: int
    hi: int
    child: object


@dataclass(frozen=True)
class Eventually:
    lo: int
    hi: int
    child: object


@dataclass(frozen=True)
class Until:
    lo: int
    hi: int
    left: object
    right: object


@dataclass(frozen=True)
class Run:
    lo: int
    hi: int
    n: int
    child: object


@dataclass(frozen=True)
class Trend:
    view: str
    lo: int
    hi: int
    delta: float = 0.0
    epsilon: float = 0.0


def _bounds(lo: int, hi: int, t: int, length: int) -> tuple[int, int]:
    start = t + lo
    stop = min(t + hi, length - 1)
    return start, stop


def rho(node, ctx: Mapping[str, np.ndarray], t: int) -> float:
    """Exact robustness of ``node`` at origin ``t``."""
    if isinstance(node, Atom):
        return node.margin(ctx[node.view], t)
    if isinstance(node, (And, Or)):
        values = [rho(c, ctx, t) for c in node.children]
        return float(min(values) if isinstance(node, And) else max(values))
    if isinstance(node, Globally):
        length = len(next(iter(ctx.values())))
        start, stop = _bounds(node.lo, node.hi, t, length)
        if start > stop:
            return np.inf
        return float(min(rho(node.child, ctx, s) for s in range(start, stop + 1)))
    if isinstance(node, Eventually):
        length = len(next(iter(ctx.values())))
        start, stop = _bounds(node.lo, node.hi, t, length)
        if start > stop:
            return -np.inf
        return float(max(rho(node.child, ctx, s) for s in range(start, stop + 1)))
    if isinstance(node, Until):
        length = len(next(iter(ctx.values())))
        start, stop = _bounds(node.lo, node.hi, t, length)
        best = -np.inf
        for s in range(start, stop + 1):
            prefix = float(min(rho(node.left, ctx, p) for p in range(t, s))) if s > t else np.inf
            best = max(best, min(rho(node.right, ctx, s), prefix))
        return float(best)
    if isinstance(node, Run):
        length = len(next(iter(ctx.values())))
        start, stop = _bounds(node.lo, node.hi, t, length)
        best = -np.inf
        for j in range(start, stop - node.n + 2):
            best = max(best, min(rho(node.child, ctx, s) for s in range(j, j + node.n)))
        return float(best)
    if isinstance(node, Trend):
        x = ctx[node.view]
        start, stop = _bounds(node.lo, node.hi, t, len(x))
        if start >= stop:
            return -np.inf
        net = float(x[stop] - x[start]) - node.delta
        excess = float(np.max(x[start:stop] - x[stop])) if stop > start else -np.inf
        return float(min(net, node.epsilon - excess))
    raise TypeError(f"not a formula: {type(node)!r}")


def rtilde(node, ctx: Mapping[str, np.ndarray], t: int, beta: float) -> float:
    """Conservative lower bound on :func:`rho`."""
    if isinstance(node, Atom):
        return node.margin(ctx[node.view], t)
    if isinstance(node, (And, Or)):
        values = [rtilde(c, ctx, t, beta) for c in node.children]
        return float(smin(values, beta) if isinstance(node, And) else smax(values, beta))
    if isinstance(node, Globally):
        length = len(next(iter(ctx.values())))
        start, stop = _bounds(node.lo, node.hi, t, length)
        if start > stop:
            return np.inf
        return smin([rtilde(node.child, ctx, s, beta) for s in range(start, stop + 1)], beta)
    if isinstance(node, Eventually):
        length = len(next(iter(ctx.values())))
        start, stop = _bounds(node.lo, node.hi, t, length)
        if start > stop:
            return -np.inf
        return smax([rtilde(node.child, ctx, s, beta) for s in range(start, stop + 1)], beta)
    if isinstance(node, Until):
        length = len(next(iter(ctx.values())))
        start, stop = _bounds(node.lo, node.hi, t, length)
        left = np.array([rtilde(node.left, ctx, p, beta) for p in range(t, stop + 1)])
        right = np.array([rtilde(node.right, ctx, s, beta) for s in range(t, stop + 1)])
        return until_scan(left, right, start - t, stop - t, beta)
    if isinstance(node, Run):
        length = len(next(iter(ctx.values())))
        start, stop = _bounds(node.lo, node.hi, t, length)
        windows = [
            smin([rtilde(node.child, ctx, s, beta) for s in range(j, j + node.n)], beta)
            for j in range(start, stop - node.n + 2)
        ]
        return smax(windows, beta) if windows else -np.inf
    if isinstance(node, Trend):
        x = ctx[node.view]
        start, stop = _bounds(node.lo, node.hi, t, len(x))
        if start >= stop:
            return -np.inf
        net = float(x[stop] - x[start]) - node.delta
        deviations = x[start:stop] - x[stop]
        upper = smax(deviations, beta) + float(np.log(max(len(deviations), 1)) / beta)
        return float(smin([net, node.epsilon - upper], beta))
    raise TypeError(f"not a formula: {type(node)!r}")


def levels(node) -> int:
    """Reduction levels ``k`` on the longest evaluation path (Lemma 2)."""
    if isinstance(node, Atom):
        return 0
    if isinstance(node, (And, Or)):
        return 1 + max(levels(c) for c in node.children)
    if isinstance(node, (Globally, Eventually)):
        return 1 + levels(node.child)
    if isinstance(node, Until):
        return 2 + max(levels(node.left), levels(node.right))
    if isinstance(node, Run):
        return 2 + levels(node.child)
    if isinstance(node, Trend):
        return 2
    raise TypeError(f"not a formula: {type(node)!r}")


def gap_bound(node, beta: float, horizon: int) -> float:
    """``k log T / beta`` of Lemma 2."""
    if horizon < 2:
        return 0.0
    return float(levels(node) * np.log(horizon) / beta)


def certified(node, ctx, t: int, beta: float) -> bool:
    """A positive conservative score certifies satisfaction (Theorem 1)."""
    return rtilde(node, ctx, t, beta) > 0


def semantic_fidelity(node, series, beta: float, firing: Callable[[object], bool]) -> float:
    """Agreement between an audited firing decision and the exact monitor (Eq. 3)."""
    agree = 0
    for x in series:
        ctx = x if isinstance(x, dict) else {"x": x}
        exact = rho(node, ctx, 0) > 0
        agree += int(exact == bool(firing(x)))
    return agree / len(series)
