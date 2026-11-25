"""SFC: semantic-fidelity certification for temporal-logic rules.

Implements the method of *Certified Temporal Logic Rules for Interpretable Time
Series Classification*: typed views (Definition 1), the shared codebook
(Definition 2), the exact and relaxed semantics with their bounds (Eq. 1 to 15),
and certification with a conformal lower bound (Sect. 4.6).
"""

from .certify import Certificate, certify, conformal_lower_bound, conformal_quantile
from .codebook import SharedCodebook
from .monitor import (
    And,
    Atom,
    Eventually,
    Globally,
    Or,
    Run,
    Trend,
    Until,
    certified,
    gap_bound,
    levels,
    rho,
    rtilde,
    semantic_fidelity,
)
from .parity import EARLY, LATE, PHASES, parity_dataset, parity_record, parity_rule
from .relax import smax, smin
from .until_scan import until_direct, until_scan
from .views import View, default_views, well_typed

__version__ = "0.1.0"

__all__ = [
    "And",
    "Atom",
    "Certificate",
    "EARLY",
    "Eventually",
    "Globally",
    "LATE",
    "Or",
    "PHASES",
    "Run",
    "SharedCodebook",
    "Trend",
    "Until",
    "View",
    "certified",
    "certify",
    "conformal_lower_bound",
    "conformal_quantile",
    "default_views",
    "gap_bound",
    "levels",
    "parity_dataset",
    "parity_record",
    "parity_rule",
    "rho",
    "rtilde",
    "semantic_fidelity",
    "smax",
    "smin",
    "until_direct",
    "until_scan",
    "well_typed",
]
