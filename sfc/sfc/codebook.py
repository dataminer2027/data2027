"""Shared threshold codebook (SFC, Sect. 4.2, Definition 2).

Bin centres are shared across all patches and all samples and are stored in
physical units:

    mu_k = x_min + (x_max - x_min) * sum_{j <= k} softmax(theta)_j

Because the cumulative softmax is strictly increasing, a bin index identifies
the same threshold everywhere; a predicate stores ``c`` in physical units and
the z-score conversion stays inside predicate evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SharedCodebook", "softmax"]


def softmax(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    shifted = theta - np.max(theta)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


@dataclass
class SharedCodebook:
    xmin: float
    xmax: float
    logits: np.ndarray

    def __post_init__(self) -> None:
        self.logits = np.asarray(self.logits, dtype=float)

    @property
    def n_bins(self) -> int:
        return len(self.logits)

    def centres(self) -> np.ndarray:
        if self.xmax <= self.xmin:
            return np.full(self.n_bins, self.xmin)
        return self.xmin + (self.xmax - self.xmin) * np.cumsum(softmax(self.logits))

    def threshold(self, index: int) -> float:
        if not 0 <= index < self.n_bins:
            raise IndexError("bin index out of range")
        return float(self.centres()[index])

    def ordered(self) -> bool:
        centres = self.centres()
        return bool(np.all(np.diff(centres) > 0))

    @classmethod
    def fit(cls, values: np.ndarray, n_bins: int = 8, seed: int = 0) -> "SharedCodebook":
        values = np.asarray(values, dtype=float).ravel()
        rng = np.random.default_rng(seed)
        return cls(float(values.min()), float(values.max()), rng.normal(size=n_bins))
