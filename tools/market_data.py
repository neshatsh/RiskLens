# tools/market_data.py
"""Market data helpers — price fetch, log returns, vol, drawdown, beta, and MA signals via yfinance."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from core.config import (
    BENCHMARK_TICKER,
    LOOKBACK_DAYS,
    ROLLING_VOL_WINDOW,
    TRADING_DAYS_PER_YEAR,
)

logger = logging.getLogger(__name__)


def fetch_price_data(
    tickers: List[str],
    lookback_days: int = LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Download adjusted close prices. Returns DataFrame[ticker → price series]."""
    end_date = datetime.today()
    # Buffer extra calendar days to account for weekends/holidays
    start_date = end_date - timedelta(days=int(lookback_days * 1.5))

    logger.info("Fetching price data for %d tickers (%s to %s)", len(tickers), start_date.date(), end_date.date())

    try:
        raw = yf.download(
            tickers,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        logger.error("yfinance download failed: %s", exc)
        return pd.DataFrame()

    # yfinance returns MultiIndex columns when >1 ticker; flatten to 'Close' only
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].dropna(how="all")
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]}).dropna(how="all")

    # Trim to the requested lookback after aligning on trading days
    prices = prices.iloc[-lookback_days:]

    missing = set(tickers) - set(prices.columns)
    if missing:
        logger.warning("No price data returned for: %s", missing)

    logger.info("Price data shape: %s", prices.shape)
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns — additive over time, required for VaR/portfolio aggregation."""
    return np.log(prices / prices.shift(1)).dropna()


def annualised_volatility(log_returns: pd.Series) -> float:
    """Annualised volatility from a log-return series."""
    return float(log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def rolling_volatility(
    log_returns: pd.Series,
    window: int = ROLLING_VOL_WINDOW,
) -> pd.Series:
    """21-day rolling annualised volatility."""
    return log_returns.rolling(window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(prices: pd.Series) -> float:
    """Maximum peak-to-trough drawdown (negative number)."""
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return float(drawdown.min())


def compute_beta(
    ticker_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """OLS beta vs benchmark (cov / var)."""
    aligned = pd.concat([ticker_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 30:
        logger.warning("Insufficient data for beta calculation")
        return float("nan")
    cov_matrix = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return float(cov_matrix[0, 1] / cov_matrix[1, 1])


def sortino_ratio(log_returns: pd.Series, risk_free_rate: float = 0.04) -> float:
    """Sortino = (ann. return - rf) / downside vol. Ignores upside variance unlike Sharpe."""
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = log_returns - daily_rf
    downside = log_returns[log_returns < 0].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    if downside == 0:
        return float("nan")
    ann_return = float(log_returns.mean() * TRADING_DAYS_PER_YEAR)
    return float((ann_return - risk_free_rate) / downside)


def moving_average_signal(prices: pd.Series) -> str:
    """Golden cross / death cross from 50-day and 200-day MAs."""
    if len(prices) < 200:
        return "insufficient_data"

    ma50 = prices.rolling(50).mean().iloc[-1]
    ma200 = prices.rolling(200).mean().iloc[-1]
    ma50_prev = prices.rolling(50).mean().iloc[-2]
    ma200_prev = prices.rolling(200).mean().iloc[-2]

    if ma50_prev < ma200_prev and ma50 > ma200:
        return "golden_cross"
    if ma50_prev > ma200_prev and ma50 < ma200:
        return "death_cross"
    if ma50 > ma200:
        return "above_200ma"
    return "below_200ma"


def get_all_market_metrics(
    tickers: List[str],
    lookback_days: int = LOOKBACK_DAYS,
) -> Dict[str, Dict]:
    """Fetch prices and compute vol, drawdown, beta, sortino, MA signal for each ticker."""
    all_tickers = list(set(tickers + [BENCHMARK_TICKER]))
    prices = fetch_price_data(all_tickers, lookback_days)

    if prices.empty:
        logger.error("No price data available — market risk analysis will be empty")
        return {}

    log_returns = compute_log_returns(prices)
    results: Dict[str, Dict] = {}

    benchmark_col = BENCHMARK_TICKER if BENCHMARK_TICKER in log_returns.columns else None

    for ticker in tickers:
        if ticker not in prices.columns:
            logger.warning("Skipping %s — no price data", ticker)
            continue

        r = log_returns[ticker].dropna()
        p = prices[ticker].dropna()

        beta = float("nan")
        if benchmark_col and benchmark_col != ticker:
            beta = compute_beta(r, log_returns[benchmark_col].dropna())

        results[ticker] = {
            "annualised_vol":   annualised_volatility(r),
            "max_drawdown":     max_drawdown(p),
            "beta":             beta,
            "sortino":          sortino_ratio(r),
            "ma_signal":        moving_average_signal(p),
            "returns":          r,           # kept for portfolio-level VaR
            "prices":           p,
        }

    return results
