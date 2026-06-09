# agents/market_risk_agent.py
"""Market Risk Agent — VaR, CVaR, volatility, drawdown, beta, and MA signals for each position."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.config import (
    BETA_HIGH_THRESHOLD,
    DRAWDOWN_ALERT_THRESHOLD,
    DRAWDOWN_SEVERE_THRESHOLD,
    HHI_HIGH_CONCENTRATION,
    VAR_99_BREACH_THRESHOLD,
    VAR_95_WARNING_THRESHOLD,
    VOL_EXTREME_THRESHOLD,
    VOL_HIGH_THRESHOLD,
)
from core.portfolio import get_tickers
from core.state import RiskLensState
from tools.market_data import get_all_market_metrics
from tools.risk_calculators import compute_all_var_metrics

logger = logging.getLogger(__name__)


def market_risk_agent(state: RiskLensState) -> RiskLensState:
    """LangGraph node: fetch prices, compute per-ticker + portfolio metrics, flag breaches."""
    if "market_risk" not in state.get("active_agents", []):
        logger.info("Market risk agent not activated — skipping")
        return {"market_risk_score": 0.0, "completed_agents": ["market_risk"]}

    portfolio = state.get("portfolio", [])
    tickers = get_tickers(portfolio)
    weights = {p["ticker"]: p["weight"] for p in portfolio}

    logger.info("Market risk agent running for %d positions", len(tickers))

    # Fetch per-ticker metrics (price history, vol, drawdown, beta, signals)
    metrics = get_all_market_metrics(tickers)

    if not metrics:
        logger.error("Market data fetch returned empty — market risk analysis incomplete")
        return {
            "market_risk_score": 50.0,
            "market_risk_flags": ["MARKET DATA UNAVAILABLE: analysis incomplete"],
            "completed_agents": ["market_risk"],
        }

    # Extract per-ticker results
    volatility_by_ticker: Dict[str, float] = {}
    max_drawdown_by_ticker: Dict[str, float] = {}
    beta_by_ticker: Dict[str, float] = {}
    sortino_by_ticker: Dict[str, float] = {}
    ma_signals: Dict[str, str] = {}
    returns_by_ticker: Dict = {}
    flags: List[str] = []

    for ticker, data in metrics.items():
        vol = data["annualised_vol"]
        dd = data["max_drawdown"]
        beta = data["beta"]
        sortino = data["sortino"]
        signal = data["ma_signal"]

        volatility_by_ticker[ticker] = round(vol, 4)
        max_drawdown_by_ticker[ticker] = round(dd, 4)
        beta_by_ticker[ticker] = round(beta, 3) if beta == beta else 0.0  # NaN guard
        sortino_by_ticker[ticker] = round(sortino, 2) if sortino == sortino else 0.0
        ma_signals[ticker] = signal
        returns_by_ticker[ticker] = data["returns"]

        # Per-ticker flags
        if vol > VOL_EXTREME_THRESHOLD:
            flags.append(f"EXTREME VOLATILITY: {ticker} showing {vol*100:.1f}% annualised vol — exceeds 55% threshold")
        elif vol > VOL_HIGH_THRESHOLD:
            flags.append(f"HIGH VOLATILITY: {ticker} showing {vol*100:.1f}% annualised vol — exceeds 35% threshold")

        if dd < -DRAWDOWN_SEVERE_THRESHOLD:
            flags.append(f"SEVERE DRAWDOWN: {ticker} down {dd*100:.1f}% from peak — exceeds -50% threshold")
        elif dd < -DRAWDOWN_ALERT_THRESHOLD:
            flags.append(f"DRAWDOWN ALERT: {ticker} down {dd*100:.1f}% from peak — exceeds -30% threshold")

        if beta > BETA_HIGH_THRESHOLD:
            flags.append(f"HIGH BETA: {ticker} beta={beta:.2f} — amplifies market moves by {beta:.1f}x")

        if signal == "death_cross":
            flags.append(f"TECHNICAL SIGNAL: {ticker} 50-day MA crossed below 200-day MA (Death Cross)")

    # Portfolio-level VaR and concentration
    var_metrics = compute_all_var_metrics(returns_by_ticker, weights)

    var_99 = var_metrics.get("var_99_hist", 0)
    var_95 = var_metrics.get("var_95_hist", 0)
    cvar_95 = var_metrics.get("cvar_95", 0)
    hhi = var_metrics.get("hhi", 0)

    if var_99 and var_99 > VAR_99_BREACH_THRESHOLD:
        flags.append(
            f"PORTFOLIO VaR BREACH: 1-day 99% VaR at {var_99*100:.2f}% — exceeds {VAR_99_BREACH_THRESHOLD*100:.1f}% threshold (Basel III escalation)"
        )
    if var_95 and var_95 > VAR_95_WARNING_THRESHOLD:
        flags.append(
            f"PORTFOLIO VaR WARNING: 1-day 95% VaR at {var_95*100:.2f}% — exceeds {VAR_95_WARNING_THRESHOLD*100:.1f}% warning level"
        )

    if hhi > HHI_HIGH_CONCENTRATION:
        flags.append(
            f"CONCENTRATION RISK: Portfolio HHI={hhi:.3f} — exceeds 0.25 threshold, indicating insufficient diversification"
        )

    # Check for duration risk: inverted curve + TLT position
    if "TLT" in tickers and state.get("yield_curve_status") == "inverted":
        flags.append("DURATION RISK: Inverted yield curve detected with long-bond (TLT) exposure — significant mark-to-market risk")

    # Normalised market risk score 0–100
    market_risk_score = _compute_market_risk_score(var_99, var_95, volatility_by_ticker, max_drawdown_by_ticker, flags)

    logger.info(
        "Market risk complete: %d flags, VaR-99=%.2f%%, CVaR-95=%.2f%%, score=%.1f",
        len(flags), (var_99 or 0) * 100, (cvar_95 or 0) * 100, market_risk_score,
    )

    return {
        "var_95": var_metrics.get("var_95_hist"),
        "var_99": var_metrics.get("var_99_hist"),
        "cvar_95": var_metrics.get("cvar_95"),
        "var_parametric_95": var_metrics.get("var_95_param"),
        "var_parametric_99": var_metrics.get("var_99_param"),
        "portfolio_volatility": var_metrics.get("portfolio_vol"),
        "hhi_concentration": var_metrics.get("hhi"),
        "volatility_by_ticker": volatility_by_ticker,
        "max_drawdown_by_ticker": max_drawdown_by_ticker,
        "beta_by_ticker": beta_by_ticker,
        "sortino_by_ticker": sortino_by_ticker,
        "ma_signals": ma_signals,
        "market_risk_flags": flags,
        "market_risk_score": market_risk_score,
        "completed_agents": ["market_risk"],
    }


def _compute_market_risk_score(
    var_99: Any,
    var_95: Any,
    volatility_by_ticker: Dict,
    max_drawdown_by_ticker: Dict,
    flags: List[str],
) -> float:
    """Score 0–100: VaR breach 40%, avg vol 30%, worst drawdown 20%, flag count 10%."""
    score = 0.0

    # VaR component
    if var_99 and var_99 > 0:
        # Scale: 0% VaR → 0 score, 5% VaR → 100 score (linear)
        var_component = min(var_99 / 0.05 * 100, 100)
        score += var_component * 0.40

    # Average volatility component
    if volatility_by_ticker:
        avg_vol = sum(volatility_by_ticker.values()) / len(volatility_by_ticker)
        # Scale: 0% vol → 0, 60% vol → 100 (linear)
        vol_component = min(avg_vol / 0.60 * 100, 100)
        score += vol_component * 0.30

    # Worst drawdown component
    if max_drawdown_by_ticker:
        worst_dd = abs(min(max_drawdown_by_ticker.values()))
        # Scale: 0% drawdown → 0, 60% drawdown → 100
        dd_component = min(worst_dd / 0.60 * 100, 100)
        score += dd_component * 0.20

    # Flag count component
    flag_component = min(len(flags) * 10, 100)
    score += flag_component * 0.10

    return round(min(score, 100.0), 2)
