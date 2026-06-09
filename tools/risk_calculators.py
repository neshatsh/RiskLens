# tools/risk_calculators.py
"""VaR, CVaR, portfolio volatility, and HHI — historical and parametric methods."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from core.config import TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)


def portfolio_log_returns(
    returns_by_ticker: Dict[str, pd.Series],
    weights: Dict[str, float],
) -> pd.Series:
    """Weighted portfolio log returns, aligned on common dates."""
    df = pd.DataFrame(returns_by_ticker).dropna(how="all")
    if df.empty:
        return pd.Series(dtype=float)

    # Build weight vector aligned to available columns
    weight_vec = np.array([weights.get(col, 0.0) for col in df.columns])
    # Renormalise in case some tickers were dropped
    total_w = weight_vec.sum()
    if total_w == 0:
        raise ValueError("All position weights are zero after alignment")
    weight_vec = weight_vec / total_w

    portfolio_returns = df.fillna(0).astype(float).values @ weight_vec
    return pd.Series(portfolio_returns, index=df.index)


def historical_var(
    portfolio_returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Historical simulation VaR — empirical distribution, no normality assumption."""
    if len(portfolio_returns) < 30:
        logger.warning("Too few observations for reliable VaR estimate")
        return float("nan")

    return float(-np.percentile(portfolio_returns, (1 - confidence) * 100))


def parametric_var(
    portfolio_returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Parametric VaR (Gaussian). Underestimates fat tails but useful as a benchmark."""
    if len(portfolio_returns) < 30:
        return float("nan")

    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    z = stats.norm.ppf(1 - confidence)
    return float(-(mu + z * sigma))


def expected_shortfall(
    portfolio_returns: pd.Series,
    confidence: float = 0.95,
) -> float:
    """Expected Shortfall — average loss in the tail. Basel III/FRTB preferred metric over VaR."""
    if len(portfolio_returns) < 30:
        return float("nan")

    var_threshold = historical_var(portfolio_returns, confidence)
    tail_losses = portfolio_returns[portfolio_returns <= -var_threshold]

    if tail_losses.empty:
        return var_threshold  # Fallback if no observations in tail

    return float(-tail_losses.mean())


def portfolio_volatility(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
) -> float:
    """Annualised portfolio vol: sqrt(w' Σ w) * sqrt(252)."""
    available = [t for t in weights if t in returns_df.columns]
    if not available:
        return float("nan")

    sub_df = returns_df[available].dropna()
    w = np.array([weights[t] for t in available])
    w = w / w.sum()

    cov = sub_df.cov().values * TRADING_DAYS_PER_YEAR
    port_var = float(w @ cov @ w)
    return float(np.sqrt(port_var))


def herfindahl_hirschman_index(weights: Dict[str, float]) -> float:
    """Normalised HHI: 0 = perfectly diversified, 1 = single position. >0.25 flags concentration."""
    w = np.array(list(weights.values()))
    n = len(w)
    if n <= 1:
        return 1.0
    raw_hhi = float(np.sum(w ** 2))
    normalised = (raw_hhi - 1 / n) / (1 - 1 / n)
    return float(np.clip(normalised, 0, 1))


def compute_all_var_metrics(
    returns_by_ticker: Dict[str, pd.Series],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """Compute all VaR/CVaR metrics and HHI in one call."""
    port_returns = portfolio_log_returns(returns_by_ticker, weights)
    returns_df = pd.DataFrame(returns_by_ticker)

    return {
        "var_95_hist":  historical_var(port_returns, 0.95),
        "var_99_hist":  historical_var(port_returns, 0.99),
        "var_95_param": parametric_var(port_returns, 0.95),
        "var_99_param": parametric_var(port_returns, 0.99),
        "cvar_95":      expected_shortfall(port_returns, 0.95),
        "portfolio_vol": portfolio_volatility(returns_df, weights),
        "hhi":          herfindahl_hirschman_index(weights),
    }
