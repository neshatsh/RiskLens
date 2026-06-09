# core/portfolio.py
"""Portfolio loader — returns built-in 10-position sample or loads from a JSON file."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Canonical sample portfolio — 10 positions across asset classes / sectors
SAMPLE_PORTFOLIO: List[Dict[str, Any]] = [
    {
        "ticker": "RY.TO",
        "name": "Royal Bank of Canada",
        "weight": 0.12,
        "asset_class": "equity",
        "sector": "financials",
        "currency": "CAD",
    },
    {
        "ticker": "ENB.TO",
        "name": "Enbridge Inc",
        "weight": 0.10,
        "asset_class": "equity",
        "sector": "energy",
        "currency": "CAD",
    },
    {
        "ticker": "SHOP.TO",
        "name": "Shopify Inc",
        "weight": 0.08,
        "asset_class": "equity",
        "sector": "technology",
        "currency": "CAD",
    },
    {
        "ticker": "XBB.TO",
        "name": "iShares Core Canadian Bond ETF",
        "weight": 0.15,
        "asset_class": "bond",
        "sector": "fixed_income",
        "currency": "CAD",
    },
    {
        "ticker": "SPY",
        "name": "S&P 500 ETF (SPDR)",
        "weight": 0.10,
        "asset_class": "equity",
        "sector": "us_equities",
        "currency": "USD",
    },
    {
        "ticker": "GLD",
        "name": "Gold ETF (SPDR)",
        "weight": 0.08,
        "asset_class": "commodity",
        "sector": "commodities",
        "currency": "USD",
    },
    {
        "ticker": "TLT",
        "name": "iShares 20+ Year Treasury Bond ETF",
        "weight": 0.10,
        "asset_class": "bond",
        "sector": "fixed_income",
        "currency": "USD",
    },
    {
        "ticker": "XLF",
        "name": "Financial Select Sector SPDR ETF",
        "weight": 0.10,
        "asset_class": "equity",
        "sector": "financials",
        "currency": "USD",
    },
    {
        "ticker": "CNR.TO",
        "name": "Canadian National Railway",
        "weight": 0.09,
        "asset_class": "equity",
        "sector": "industrials",
        "currency": "CAD",
    },
    {
        "ticker": "BCE.TO",
        "name": "BCE Inc",
        "weight": 0.08,
        "asset_class": "equity",
        "sector": "telecom",
        "currency": "CAD",
    },
]


def load_portfolio(json_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load from json_path if given, otherwise return SAMPLE_PORTFOLIO."""
    if json_path and os.path.exists(json_path):
        try:
            with open(json_path, "r") as fh:
                portfolio = json.load(fh)
            logger.info("Loaded portfolio from %s (%d positions)", json_path, len(portfolio))
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to parse %s (%s) — falling back to sample portfolio", json_path, exc)
            portfolio = SAMPLE_PORTFOLIO
    else:
        portfolio = SAMPLE_PORTFOLIO
        logger.info("Using built-in sample portfolio (%d positions)", len(portfolio))

    _validate_portfolio(portfolio)
    return portfolio


def _validate_portfolio(portfolio: List[Dict[str, Any]]) -> None:
    """Raise ValueError if portfolio is malformed; log warnings for soft issues."""
    required_keys = {"ticker", "weight", "asset_class", "sector"}
    for pos in portfolio:
        missing = required_keys - pos.keys()
        if missing:
            raise ValueError(f"Position {pos.get('ticker', '?')} missing keys: {missing}")

    total_weight = sum(p["weight"] for p in portfolio)
    if not (0.99 <= total_weight <= 1.01):
        logger.warning("Portfolio weights sum to %.4f — expected ~1.0", total_weight)

    tickers = [p["ticker"] for p in portfolio]
    if len(tickers) != len(set(tickers)):
        logger.warning("Duplicate tickers detected in portfolio")


def get_tickers(portfolio: List[Dict[str, Any]]) -> List[str]:
    """Extract ticker symbols from a portfolio."""
    return [p["ticker"] for p in portfolio]


def get_equity_positions(portfolio: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter to equity positions only (for credit risk and news scanning)."""
    return [p for p in portfolio if p["asset_class"] == "equity"]


def save_portfolio(portfolio: List[Dict[str, Any]], path: str) -> None:
    """Persist a portfolio to JSON for reuse."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(portfolio, fh, indent=2)
    logger.info("Saved portfolio to %s", path)
