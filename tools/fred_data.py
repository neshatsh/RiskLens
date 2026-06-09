# tools/fred_data.py
"""FRED API wrapper — pulls macro indicators and derives yield curve / recession signals."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd
import requests

from core.config import (
    FRED_API_KEY,
    FRED_SERIES,
    HY_SPREAD_STRESS_THRESHOLD,
    INFLATION_HIGH_THRESHOLD,
    VIX_ELEVATED_THRESHOLD,
    VIX_EXTREME_THRESHOLD,
    YIELD_CURVE_INVERSION_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _fetch_fred_series(series_id: str, api_key: str, lookback_days: int = 90) -> Optional[float]:
    """Fetch most recent value of a FRED series. Returns None on any failure."""
    observation_start = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 5,
        "observation_start": observation_start,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        observations = resp.json().get("observations", [])
        for obs in observations:
            if obs["value"] != ".":   # FRED uses "." for missing values
                return float(obs["value"])
        logger.warning("No valid observation found for FRED series %s", series_id)
        return None
    except requests.RequestException as exc:
        logger.warning("FRED API request failed for %s: %s", series_id, exc)
        return None
    except (KeyError, ValueError) as exc:
        logger.warning("FRED response parsing error for %s: %s", series_id, exc)
        return None


def _fetch_cpi_yoy(api_key: str) -> Optional[float]:
    """Compute CPI YoY% = (current_index / index_12_months_ago - 1) * 100.

    CPIAUCSL is an index level (~330), not a percentage — we need two observations
    13 months apart to compute the year-over-year change correctly.
    """
    observation_start = (datetime.today() - timedelta(days=500)).strftime("%Y-%m-%d")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "CPIAUCSL",
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 16,
        "observation_start": observation_start,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        observations = [
            obs for obs in resp.json().get("observations", [])
            if obs["value"] != "."
        ]
        if len(observations) < 13:
            logger.warning("Not enough CPI observations to compute YoY (%d found)", len(observations))
            return None
        # observations are sorted desc — index 0 is most recent, index 12 is ~12 months ago
        current = float(observations[0]["value"])
        year_ago = float(observations[12]["value"])
        return round((current / year_ago - 1) * 100, 2)
    except requests.RequestException as exc:
        logger.warning("FRED CPI YoY request failed: %s", exc)
        return None
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        logger.warning("FRED CPI YoY parsing error: %s", exc)
        return None


def _fetch_gdp_growth(api_key: str) -> Optional[float]:
    """Annualised QoQ real GDP growth: ((current / prev_quarter) ** 4 - 1) * 100.

    GDPC1 is chained 2017 dollar levels (~24000), not a growth rate. We fetch
    the two most recent quarters and annualise the quarter-over-quarter change.
    """
    observation_start = (datetime.today() - timedelta(days=270)).strftime("%Y-%m-%d")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "GDPC1",
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 4,
        "observation_start": observation_start,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        observations = [
            obs for obs in resp.json().get("observations", [])
            if obs["value"] != "."
        ]
        if len(observations) < 2:
            logger.warning("Not enough GDP observations to compute growth (%d found)", len(observations))
            return None
        current = float(observations[0]["value"])
        previous = float(observations[1]["value"])
        annualised_growth = ((current / previous) ** 4 - 1) * 100
        return round(annualised_growth, 2)
    except requests.RequestException as exc:
        logger.warning("FRED GDP growth request failed: %s", exc)
        return None
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        logger.warning("FRED GDP growth parsing error: %s", exc)
        return None


def fetch_macro_indicators(api_key: str = FRED_API_KEY) -> Dict[str, Any]:
    """Pull all configured FRED series. Returns None values if FRED_API_KEY is not set."""
    if not api_key:
        logger.warning("FRED_API_KEY not set — macro indicators will be unavailable")
        return {sid: None for sid in FRED_SERIES}

    indicators: Dict[str, Any] = {}
    for series_id, label in FRED_SERIES.items():
        if series_id == "CPIAUCSL":
            value = _fetch_cpi_yoy(api_key)
        elif series_id == "GDPC1":
            value = _fetch_gdp_growth(api_key)
        else:
            value = _fetch_fred_series(series_id, api_key)
        indicators[series_id] = value
        logger.debug("FRED %s (%s): %s", series_id, label, value)

    return indicators


def fetch_boc_overnight_rate() -> Optional[float]:
    """Bank of Canada overnight rate via BoC Valet API (no key needed)."""
    url = "https://www.bankofcanada.ca/valet/observations/V122514/json"
    params = {"recent": 1}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        observations = data.get("observations", [])
        if observations:
            return float(observations[-1]["V122514"]["v"])
        return None
    except Exception as exc:
        logger.warning("Bank of Canada API request failed: %s", exc)
        return None


def derive_macro_signals(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """Compute yield curve status, recession probability, VIX regime, inflation regime from FRED data."""
    signals: Dict[str, Any] = {}

    # Yield curve status
    spread = indicators.get("T10Y2Y")
    if spread is not None:
        if spread < YIELD_CURVE_INVERSION_THRESHOLD:
            signals["yield_curve_status"] = "inverted"
        elif spread < 0.25:
            signals["yield_curve_status"] = "flat"
        else:
            signals["yield_curve_status"] = "normal"
    else:
        signals["yield_curve_status"] = "unknown"

    # Inflation regime
    cpi = indicators.get("CPIAUCSL")
    if cpi is not None:
        signals["inflation_regime"] = "HIGH" if cpi > INFLATION_HIGH_THRESHOLD else "NORMAL"
        signals["cpi_value"] = cpi
    else:
        signals["inflation_regime"] = "unknown"

    # Credit stress (high-yield spread)
    hy_spread = indicators.get("BAMLH0A0HYM2")
    if hy_spread is not None:
        signals["hy_credit_stress"] = "ELEVATED" if hy_spread > HY_SPREAD_STRESS_THRESHOLD else "NORMAL"
        signals["hy_spread_bps"] = hy_spread
    else:
        signals["hy_credit_stress"] = "unknown"

    # VIX regime
    vix = indicators.get("VIXCLS")
    if vix is not None:
        if vix > VIX_EXTREME_THRESHOLD:
            signals["vix_regime"] = "EXTREME_FEAR"
        elif vix > VIX_ELEVATED_THRESHOLD:
            signals["vix_regime"] = "RISK_OFF"
        else:
            signals["vix_regime"] = "NORMAL"
        signals["vix_value"] = vix
    else:
        signals["vix_regime"] = "unknown"

    # Recession probability proxy (inverted curve + rising unemployment + negative GDP)
    unrate = indicators.get("UNRATE")
    gdp = indicators.get("GDPC1")
    recession_signals = 0
    if signals["yield_curve_status"] == "inverted":
        recession_signals += 1
    if unrate is not None and unrate > 5.0:
        recession_signals += 1
    if gdp is not None and gdp < 0:
        recession_signals += 1

    if recession_signals >= 3:
        signals["recession_probability"] = "HIGH"
    elif recession_signals >= 2:
        signals["recession_probability"] = "ELEVATED"
    else:
        signals["recession_probability"] = "LOW"

    return signals
