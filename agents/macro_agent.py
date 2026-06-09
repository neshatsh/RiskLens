# agents/macro_agent.py
"""Macro Risk Agent — FRED indicators, yield curve status, and systemic risk flags."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.config import (
    FRED_SERIES,
    HY_SPREAD_STRESS_THRESHOLD,
    INFLATION_HIGH_THRESHOLD,
    VIX_ELEVATED_THRESHOLD,
    VIX_EXTREME_THRESHOLD,
)
from core.state import RiskLensState
from tools.fred_data import derive_macro_signals, fetch_boc_overnight_rate, fetch_macro_indicators

logger = logging.getLogger(__name__)


def macro_agent(state: RiskLensState) -> RiskLensState:
    """LangGraph node: FRED indicators, yield curve, VIX, and recession signals."""
    if "macro" not in state.get("active_agents", []):
        logger.info("Macro agent not activated — skipping")
        return {"macro_risk_score": 0.0, "completed_agents": ["macro"]}

    logger.info("Macro agent fetching FRED indicators")

    # Fetch raw FRED data
    indicators = fetch_macro_indicators()

    # Try to add Bank of Canada rate (graceful failure)
    boc_rate = fetch_boc_overnight_rate()
    if boc_rate is not None:
        indicators["BOC_RATE"] = boc_rate

    # Derive composite signals
    signals = derive_macro_signals(indicators)

    # Generate flags
    flags: List[str] = []

    yield_curve_status = signals.get("yield_curve_status", "unknown")
    if yield_curve_status == "inverted":
        spread = indicators.get("T10Y2Y")
        spread_str = f"{spread:.2f}%" if spread is not None else "N/A"
        flags.append(
            f"YIELD CURVE INVERTED: 10Y-2Y spread at {spread_str} — "
            "historical recession indicator with avg 12-18 month lead time"
        )

    recession_prob = signals.get("recession_probability", "LOW")
    if recession_prob in ("HIGH", "ELEVATED"):
        flags.append(
            f"RECESSION SIGNAL: {recession_prob} composite recession probability — "
            "inverted curve + macro deterioration signals"
        )

    inflation_regime = signals.get("inflation_regime", "NORMAL")
    cpi = signals.get("cpi_value")
    if inflation_regime == "HIGH" and cpi is not None:
        flags.append(
            f"HIGH INFLATION: CPI at {cpi:.1f}% YoY — real return compression expected; "
            "rate hike pressure on duration positions"
        )

    hy_stress = signals.get("hy_credit_stress", "NORMAL")
    hy_spread = signals.get("hy_spread_bps")
    if hy_stress == "ELEVATED" and hy_spread is not None:
        flags.append(
            f"CREDIT MARKET STRESS: US HY spread at {hy_spread:.0f}bps — "
            f"exceeds {HY_SPREAD_STRESS_THRESHOLD:.0f}bps threshold; risk-off conditions"
        )

    vix_regime = signals.get("vix_regime", "NORMAL")
    vix_val = signals.get("vix_value")
    if vix_regime == "EXTREME_FEAR" and vix_val is not None:
        flags.append(
            f"EXTREME MARKET FEAR: VIX at {vix_val:.1f} — exceeds 35; "
            "severe risk-off environment, elevated correlation across asset classes"
        )
    elif vix_regime == "RISK_OFF" and vix_val is not None:
        flags.append(
            f"RISK-OFF SIGNAL: VIX at {vix_val:.1f} — elevated uncertainty, "
            "reduced liquidity in risk assets"
        )

    vix = indicators.get("VIXCLS")
    hy = indicators.get("BAMLH0A0HYM2")
    if vix and hy and vix > VIX_ELEVATED_THRESHOLD and hy > HY_SPREAD_STRESS_THRESHOLD:
        flags.append(
            "DUAL STRESS SIGNAL: Simultaneously elevated VIX and HY spreads — "
            "systemic risk-off conditions similar to 2020/2022 stress periods"
        )

    if boc_rate:
        fed_rate = indicators.get("DFF")
        if fed_rate and abs(boc_rate - fed_rate) > 1.5:
            flags.append(
                f"CAD/USD RATE DIVERGENCE: BoC rate ({boc_rate:.2f}%) vs Fed rate ({fed_rate:.2f}%) "
                "divergence exceeds 150bps — CAD FX pressure on USD-denominated positions"
            )

    macro_risk_score = _compute_macro_risk_score(signals, indicators, flags)

    logger.info(
        "Macro agent complete: %d flags, yield_curve=%s, recession=%s, score=%.1f",
        len(flags), yield_curve_status, recession_prob, macro_risk_score,
    )

    return {
        "macro_indicators": indicators,
        "yield_curve_status": yield_curve_status,
        "recession_probability": recession_prob,
        "macro_risk_flags": flags,
        "macro_risk_score": macro_risk_score,
        "completed_agents": ["macro"],
    }


def _compute_macro_risk_score(
    signals: Dict,
    indicators: Dict,
    flags: List[str],
) -> float:
    """Score 0–100 with a proportional baseline so benign conditions score ~10-25, not 0.

    Binary threshold bonuses on top of continuous baseline contributions:
    - VIX: continuous 0-20pts based on level
    - Yield curve: continuous spread-based contribution + inversion bonus
    - Inflation: continuous distance from 2% target
    - HY spread: continuous level-based contribution
    - Short rate level: elevated rates add baseline macro risk
    """
    score = 0.0

    # VIX — continuous contribution: 0 at VIX=10, 20 at VIX=35+
    vix = signals.get("vix_value")
    if vix is not None:
        vix_contribution = min(20.0, max(0.0, (vix - 10.0) / 25.0 * 20.0))
        score += vix_contribution
        # Threshold bonuses on top
        vix_reg = signals.get("vix_regime", "NORMAL")
        if vix_reg == "EXTREME_FEAR":
            score += 15
        elif vix_reg == "RISK_OFF":
            score += 7

    # Yield curve — inversion is a large bonus; normal positive spread is low baseline
    spread = indicators.get("T10Y2Y")
    curve = signals.get("yield_curve_status", "unknown")
    if spread is not None:
        if curve == "inverted":
            score += 25
        elif curve == "flat":
            score += 10
        else:
            # Positive spread: narrower = higher risk baseline (0-5pts)
            score += max(0.0, (1.0 - spread) * 3.0)

    # Recession signal
    rec = signals.get("recession_probability", "LOW")
    if rec == "HIGH":
        score += 20
    elif rec == "ELEVATED":
        score += 10

    # CPI inflation — continuous: 2% target = 0 baseline, >4% = elevated, scales up to 10pts
    cpi = signals.get("cpi_value")
    if cpi is not None:
        inflation_contribution = min(10.0, max(0.0, abs(cpi - 2.0) / 3.0 * 6.0))
        score += inflation_contribution
        if signals.get("inflation_regime") == "HIGH":
            score += 5

    # HY spread — continuous: 300bps = 0 baseline, 600bps = elevated, scales to 15pts
    hy = signals.get("hy_spread_bps")
    if hy is not None:
        hy_contribution = min(15.0, max(0.0, (hy - 300.0) / 300.0 * 15.0))
        score += hy_contribution
        if signals.get("hy_credit_stress") == "ELEVATED":
            score += 10

    # Short rate level — very high or very low rates both add baseline macro risk
    fed_rate = indicators.get("DFF")
    if fed_rate is not None:
        if fed_rate > 4.5:
            score += 5
        elif fed_rate > 3.0:
            score += 3
        elif fed_rate < 0.5:
            score += 4  # ZIRP environment is abnormal, has its own risk

    return round(min(score, 100.0), 2)
