"""Cross-channel parity benchmark (SFC, Sect. 5.6/5.7).

Length-48 records with three channels; each channel carries one four-sample
pulse in ``I0 = [8, 11]`` or ``I1 = [32, 35]``.  The signal is ``+a`` inside the
pulse and ``-a`` elsewhere.  The label is positive when an odd number of
channels have late pulses, and the rule is the disjunction over the four
odd-parity phase assignments of a conjunction of ``G_I(x_i > 0)`` clauses:

    phi_odd = OR_{p odd} AND_i  G_{I_{p_i}} (x_i > 0)

Moving a single pulse between the two windows flips both scores at fixed
amplitude and duration, so the benchmark isolates joint timing rather than
signal strength.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from .monitor import And, Atom, Globally, Or

__all__ = ["EARLY", "LATE", "LENGTH", "PHASES", "parity_record", "parity_dataset", "parity_rule", "context"]

EARLY = (8, 11)
LATE = (32, 35)
LENGTH = 48
PHASES = tuple(product((0, 1), repeat=3))


def parity_record(amplitude: float, phase: tuple[int, ...], length: int = LENGTH) -> np.ndarray:
    x = np.full((length, len(phase)), -amplitude, dtype=float)
    for channel, late in enumerate(phase):
        lo, hi = LATE if late else EARLY
        x[lo : hi + 1, channel] = amplitude
    return x


def parity_dataset(amplitude: float, length: int = LENGTH):
    X = np.stack([parity_record(amplitude, p, length) for p in PHASES])
    y = np.array([sum(p) % 2 for p in PHASES])
    return X, y, PHASES


def parity_rule() -> Or:
    disjuncts = []
    for phase in PHASES:
        if sum(phase) % 2 == 0:
            continue
        clauses = []
        for channel, late in enumerate(phase):
            lo, hi = LATE if late else EARLY
            clauses.append(Globally(lo, hi, Atom(f"ch{channel}", ">", 0.0)))
        disjuncts.append(And(tuple(clauses)))
    return Or(tuple(disjuncts))


def context(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {f"ch{i}": x[:, i] for i in range(x.shape[1])}
