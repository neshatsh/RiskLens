# agents/operational_risk_agent.py
"""Operational Risk Agent — news-driven scan for fraud, sanctions, cyber, and legal events."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from core.config import SANCTIONED_COUNTRIES
from core.portfolio import get_tickers
from core.state import RiskLensState
from tools.news_scanner import (
    check_sanctions_exposure,
    fetch_company_news,
    scan_for_op_risk_events,
)

logger = logging.getLogger(__name__)


def operational_risk_agent(state: RiskLensState) -> RiskLensState:
    """LangGraph node: scan news for operational risk events per position."""
    if "operational_risk" not in state.get("active_agents", []):
        logger.info("Operational risk agent not activated — skipping")
        return {"operational_risk_score": 0.0, "completed_agents": ["operational_risk"]}

    portfolio = state.get("portfolio", [])
    logger.info("Operational risk agent scanning %d positions", len(portfolio))

    operational_events: Dict[str, List[Dict]] = {}
    all_sanctions_flags: List[str] = []
    flags: List[str] = []
    severity_tally = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    direct_sanctions_count = 0
    indirect_sanctions_count = 0

    for position in portfolio:
        ticker = position["ticker"]
        company_name = position.get("name", ticker)

        # Fetch headlines for operational risk scanning
        try:
            headlines = fetch_company_news(company_name, ticker, days_back=30)
        except Exception as exc:
            logger.warning("News fetch failed for %s: %s", ticker, exc)
            headlines = []

        # Detect operational risk events
        events = scan_for_op_risk_events(headlines, ticker)
        operational_events[ticker] = events

        # Check sanctions exposure — ETFs get indirect treatment
        is_etf = position.get("asset_class") in ("bond", "commodity") or ticker.endswith((".TO",)) is False and len(ticker) <= 4 and position.get("name", "").lower().find("etf") != -1
        is_etf = position.get("asset_class") in ("bond", "commodity") or "etf" in position.get("name", "").lower()
        sanctions = check_sanctions_exposure(company_name, headlines, is_etf=is_etf)
        all_sanctions_flags.extend(sanctions)
        if any("INDIRECT" in f for f in sanctions):
            indirect_sanctions_count += 1
        elif sanctions:
            direct_sanctions_count += 1

        if not events:
            continue

        # Aggregate events by severity for scoring
        ticker_max_severity = "LOW"
        for event in events:
            sev = event["severity"]
            severity_tally[sev] += 1
            if _severity_rank(sev) > _severity_rank(ticker_max_severity):
                ticker_max_severity = sev

        # Generate ticker-level flag
        event_count = len(events)
        categories = list({e["category"] for e in events})

        if ticker_max_severity == "CRITICAL":
            flags.append(
                f"CRITICAL OPERATIONAL RISK: {ticker} — {event_count} flagged event(s) "
                f"in categories: {', '.join(categories)}. Immediate escalation required."
            )
        elif ticker_max_severity == "HIGH":
            flags.append(
                f"HIGH OPERATIONAL RISK: {ticker} — {event_count} event(s) detected "
                f"({', '.join(categories)})"
            )
        elif ticker_max_severity == "MEDIUM":
            flags.append(
                f"MEDIUM OPERATIONAL RISK: {ticker} — {event_count} event(s) in monitoring period"
            )

    # Add sanctions flags — indirect ones use LOW label, direct use MEDIUM label
    for sf in all_sanctions_flags:
        if "INDIRECT" in sf:
            flags.append(f"LOW INDIRECT SANCTIONS EXPOSURE: {sf.split(': ', 1)[-1]}")
        else:
            flags.append(f"MEDIUM DIRECT SANCTIONS EXPOSURE: {sf.split(': ', 1)[-1]}")

    # Compute operational risk score 0–100
    op_score = _compute_op_risk_score(severity_tally, direct_sanctions_count, indirect_sanctions_count)

    logger.info(
        "Operational risk complete: %d flags (%d CRITICAL, %d HIGH, %d direct sanctions, %d indirect), score=%.1f",
        len(flags), severity_tally["CRITICAL"], severity_tally["HIGH"],
        direct_sanctions_count, indirect_sanctions_count, op_score,
    )

    return {
        "operational_events": operational_events,
        "sanctions_flags": all_sanctions_flags,
        "operational_risk_flags": flags,
        "operational_risk_score": op_score,
        "completed_agents": ["operational_risk"],
    }


def _severity_rank(severity: str) -> int:
    """Convert severity string to comparable integer."""
    return {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}.get(severity, 0)


def _compute_op_risk_score(
    severity_tally: Dict[str, int],
    direct_sanctions: int,
    indirect_sanctions: int,
) -> float:
    """Op risk score 0–100.

    Severity weights are per-event contributions capped so a single indirect
    sanctions hit on an ETF can't drive the whole portfolio to CRITICAL.
    - Indirect ETF-level sanctions → 10pts each (LOW risk, index constituent noise)
    - Direct company sanctions mention → 25pts each (MEDIUM risk, needs monitoring)
    - CRITICAL confirmed events → 40pts each
    - HIGH events → 15pts each
    - MEDIUM events → 6pts each
    - LOW events → 2pts each
    """
    score = 0.0

    score += severity_tally.get("CRITICAL", 0) * 40
    score += severity_tally.get("HIGH", 0) * 15
    score += severity_tally.get("MEDIUM", 0) * 6
    score += severity_tally.get("LOW", 0) * 2

    # Direct sanctions exposure — meaningful but not automatic CRITICAL
    score += direct_sanctions * 25

    # Indirect / ETF-level sanctions — low weight, this is index constituent noise
    score += indirect_sanctions * 10

    return round(min(score, 100.0), 2)
