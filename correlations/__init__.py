"""Correlation analysis for nova-research.

Stable public API (see correlations/estimators.py for details)::

    from correlations import load_returns, estimate_covariance

    data = load_returns(["SPY", "TLT", "SU.PA"], base="EUR")   # aligned log returns
    risk = estimate_covariance(data.returns)                   # annualised, 252 days,
    risk.covariance, risk.volatility, risk.warnings            # Ledoit-Wolf by default

    from correlations import estimate_correlation, CorrelationEstimate

Research output only -- nothing in this package places an order.
"""

from correlations.estimators import (
    COVARIANCE_METHODS,
    DEFAULT_COVARIANCE_METHOD,
    METHODS,
    CorrelationEstimate,
    CovarianceEstimate,
    PairStatistic,
    estimate_correlation,
    estimate_covariance,
)

__all__ = [
    "COVARIANCE_METHODS",
    "DEFAULT_COVARIANCE_METHOD",
    "METHODS",
    "CorrelationEstimate",
    "CovarianceEstimate",
    "PairStatistic",
    "estimate_correlation",
    "estimate_covariance",
    "load_returns",
]


def load_returns(*args, **kwargs):
    """Lazy wrapper for correlations.analysis.load_returns (avoids a heavy import)."""
    from correlations.analysis import load_returns as _load

    return _load(*args, **kwargs)
