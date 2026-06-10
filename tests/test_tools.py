# tests/test_tools.py
"""Unit tests for the tools layer (market data, risk calculators, news scanner)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.risk_calculators import (
    expected_shortfall,
    herfindahl_hirschman_index,
    historical_var,
    parametric_var,
    portfolio_log_returns,
    portfolio_volatility,
)
from tools.market_data import annualised_volatility, max_drawdown, moving_average_signal, sortino_ratio
from tools.news_scanner import classify_headline_severity, scan_credit_sentiment


# risk calculator tests

def _sample_returns(n: int = 252, seed: int = 42) -> pd.Series:
    """Generate realistic-looking daily log returns for testing."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0003, scale=0.012, size=n)
    return pd.Series(returns)


def test_historical_var_is_positive():
    returns = _sample_returns()
    var = historical_var(returns, confidence=0.95)
    assert var > 0, "VaR must be a positive loss estimate"


def test_var_99_greater_than_var_95():
    returns = _sample_returns()
    var95 = historical_var(returns, confidence=0.95)
    var99 = historical_var(returns, confidence=0.99)
    assert var99 > var95, "99% VaR should be greater than 95% VaR"


def test_cvar_greater_than_var():
    returns = _sample_returns()
    var = historical_var(returns, confidence=0.95)
    cvar = expected_shortfall(returns, confidence=0.95)
    assert cvar >= var, "CVaR (average tail loss) must be >= VaR threshold"


def test_parametric_var_similar_to_historical():
    """For normally distributed returns, parametric and historical VaR should be close."""
    rng = np.random.default_rng(99)
    returns = pd.Series(rng.normal(0, 0.01, 10_000))  # True normal distribution
    hist = historical_var(returns, confidence=0.95)
    param = parametric_var(returns, confidence=0.95)
    # Should be within 20% of each other for a large normal sample
    assert abs(hist - param) / param < 0.20, f"hist={hist:.4f} param={param:.4f} diverge too much for normal distribution"


def test_portfolio_log_returns_weighted():
    r1 = _sample_returns(seed=1)
    r2 = _sample_returns(seed=2)
    returns_dict = {"A": r1, "B": r2}
    weights = {"A": 0.6, "B": 0.4}
    port = portfolio_log_returns(returns_dict, weights)
    assert len(port) == len(r1)
    assert not port.isna().all()


def test_hhi_single_position():
    """A portfolio with one position should have HHI = 1."""
    hhi = herfindahl_hirschman_index({"SPY": 1.0})
    assert hhi == 1.0


def test_hhi_equal_weights():
    """Equal weights → minimum concentration."""
    weights = {str(i): 0.1 for i in range(10)}
    hhi = herfindahl_hirschman_index(weights)
    assert hhi < 0.05, f"Equal-weight HHI should be near 0, got {hhi}"


def test_hhi_concentration_detection():
    """A portfolio with 80% in one name should be flagged as concentrated."""
    from core.config import HHI_HIGH_CONCENTRATION
    weights = {"AAPL": 0.80, "GOOG": 0.20}
    hhi = herfindahl_hirschman_index(weights)
    assert hhi > HHI_HIGH_CONCENTRATION


# market data tests

def test_annualised_volatility_scale():
    """Daily vol of 1% should annualise to ~15.9% with sqrt(252)."""
    daily_returns = pd.Series([0.01] * 252)
    vol = annualised_volatility(daily_returns)
    # std of constant series is 0 — use a noisy series instead
    rng = np.random.default_rng(0)
    noisy = pd.Series(rng.normal(0, 0.01, 252))
    vol = annualised_volatility(noisy)
    expected = 0.01 * np.sqrt(252)
    assert abs(vol - expected) / expected < 0.05, f"Annualised vol {vol:.4f} should ≈ {expected:.4f}"


def test_max_drawdown_negative():
    prices = pd.Series([100, 110, 105, 90, 95])
    dd = max_drawdown(prices)
    assert dd < 0, "Drawdown must be negative"
    # Peak 110, trough 90 → drawdown = (90-110)/110 ≈ -18.2%
    assert abs(dd - (-20 / 110)) < 0.01


def test_max_drawdown_monotone_up():
    """Monotonically rising prices have zero drawdown."""
    prices = pd.Series([100, 101, 102, 103, 104])
    assert max_drawdown(prices) == 0.0


def test_moving_average_signal_insufficient_data():
    prices = pd.Series(range(100))
    assert moving_average_signal(prices) == "insufficient_data"


def test_sortino_ratio_finite():
    returns = _sample_returns()
    sortino = sortino_ratio(returns)
    assert np.isfinite(sortino)


# news scanner tests

def test_classify_headline_severity_critical():
    headline = "Bank faces DOJ indictment over fraud charges"
    assert classify_headline_severity(headline) in ("CRITICAL", "HIGH")


def test_classify_headline_severity_neutral():
    headline = "Company reports quarterly earnings in line with expectations"
    assert classify_headline_severity(headline) == "NEUTRAL"


def test_scan_credit_sentiment_severe():
    headlines = ["Company faces SEC charge for fraud", "DOJ investigation opened"]
    sentiment = scan_credit_sentiment(headlines)
    assert sentiment == "SEVERE"


def test_scan_credit_sentiment_empty():
    assert scan_credit_sentiment([]) == "NEUTRAL"
