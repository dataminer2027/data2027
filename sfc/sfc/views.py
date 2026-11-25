"""Typed views and their sorts (SFC, Sect. 4.2, Definition 1).

A view is a deterministic transformation of the input series that carries a
sort drawn from ``{Time, Freq, Deriv, Stat}``.  Temporal operators are defined
only over ``Time``/``Deriv`` subformulas, a ``Freq`` subformula may only enter an
aggregation that reduces its frequency axis, and ``Stat`` predicates live at
patch level.  The lookup table in :data:`ALLOWED` filters candidate formulas
before learning, so an interval can never index a periodogram's frequency bins
(Proposition 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

__all__ = ["View", "SORTS", "TEMPORAL_ONLY", "ALLOWED", "well_typed", "default_views"]

SORTS = ("Time", "Freq", "Deriv", "Stat")
TEMPORAL_ONLY = ("Time", "Deriv")

#: which operators accept which child sorts
ALLOWED = {
    "G": TEMPORAL_ONLY,
    "F": TEMPORAL_ONLY,
    "U": TEMPORAL_ONLY,
    "R": TEMPORAL_ONLY,
    "Tr": TEMPORAL_ONLY,
    "aggregate": ("Freq", "Stat"),
}


@dataclass(frozen=True)
class View:
    """A named deterministic transformation with a sort."""

    name: str
    sort: str
    channel: int
    kind: str = "level"
    param: float = 0.0

    def series(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        channel = x[:, self.channel]
        if self.kind == "level":
            return channel.copy()
        if self.kind == "diff":
            return np.diff(channel, prepend=channel[0])
        if self.kind == "fft":
            spectrum = np.abs(np.fft.rfft(channel))
            return np.interp(np.linspace(0, len(spectrum) - 1, len(channel)), np.arange(len(spectrum)), spectrum)
        if self.kind == "patch_mean":
            width = max(int(self.param), 1)
            out = np.empty_like(channel)
            for t in range(len(channel)):
                start = max(0, t - width + 1)
                out[t] = channel[start : t + 1].mean()
            return out
        raise ValueError(f"unknown view kind: {self.kind!r}")


def default_views(n_channels: int, patch: int = 8) -> tuple[View, ...]:
    views: list[View] = []
    for v in range(n_channels):
        views.append(View(f"ch{v}:level", "Time", v))
        views.append(View(f"ch{v}:diff", "Deriv", v, "diff"))
        views.append(View(f"ch{v}:fft", "Freq", v, "fft"))
        views.append(View(f"ch{v}:mean", "Stat", v, "patch_mean", float(patch)))
    return tuple(views)


def _sort_of(node) -> str:
    from .monitor import And, Atom, Eventually, Globally, Or, Run, Trend, Until

    if isinstance(node, Atom):
        return node.sort
    if isinstance(node, (And, Or)):
        sorts = {_sort_of(c) for c in node.children}
        if len(sorts) != 1:
            raise ValueError("mixed sorts under a connective")
        return sorts.pop()
    if isinstance(node, (Globally, Eventually)):
        return _sort_of(node.child)
    if isinstance(node, (Until, Run)):
        return _sort_of(node.left) if isinstance(node, Until) else _sort_of(node.child)
    if isinstance(node, Trend):
        return "Deriv"
    raise TypeError(f"not a formula: {type(node)!r}")


def well_typed(node) -> bool:
    """Definition 1 as a predicate on a formula tree."""
    from .monitor import And, Atom, Eventually, Globally, Or, Run, Trend, Until

    try:
        if isinstance(node, Atom):
            return node.sort in SORTS
        if isinstance(node, (And, Or)):
            return all(well_typed(c) for c in node.children) and len({_sort_of(c) for c in node.children}) == 1
        if isinstance(node, (Globally, Eventually)):
            return well_typed(node.child) and _sort_of(node.child) in TEMPORAL_ONLY
        if isinstance(node, Until):
            return (
                well_typed(node.left)
                and well_typed(node.right)
                and _sort_of(node.left) in TEMPORAL_ONLY
                and _sort_of(node.right) in TEMPORAL_ONLY
            )
        if isinstance(node, Run):
            return well_typed(node.child) and _sort_of(node.child) in TEMPORAL_ONLY
        if isinstance(node, Trend):
            return True
        return False
    except ValueError:
        return False
