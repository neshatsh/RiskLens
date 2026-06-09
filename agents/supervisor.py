# agents/supervisor.py
"""Supervisor agent — routes to specialist agents based on portfolio content, then aggregates scores."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    RISK_LEVEL_THRESHOLDS,
    RISK_WEIGHTS,
)
from core.state import RiskLensState

logger = logging.getLogger(__name__)

SUPERVISOR_SYSTEM_PROMPT = """You are a Chief Risk Officer (CRO) at a Canadian bank.
You are overseeing an automated risk analysis of a financial portfolio.

Your job is to decide which specialist risk agents should be activated for this analysis.
The available agents are:
- market_risk: Calculates VaR, volatility, drawdown, beta, moving averages
- credit_risk: Assesses creditworthiness via news sentiment and financial proxies
- operational_risk: Scans for fraud, sanctions, cyber, legal, and key person risk
- macro: Pulls FRED macro data (rates, inflation, yield curve, VIX)
- rag: Retrieves relevant Basel III/IV regulatory passages for the detected risks

For most portfolios, all agents should run. Only exclude an agent if it is clearly
irrelevant (e.g., no equity positions means credit risk may have limited value).

Respond with ONLY valid JSON in this exact format:
{
  "agents": ["market_risk", "credit_risk", "operational_risk", "macro", "rag"],
  "reasoning": "Brief explanation of routing decision"
}"""


def _get_llm():
    """Instantiate the configured LLM backend."""
    if LLM_PROVIDER == "anthropic" and ANTHROPIC_API_KEY:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=ANTHROPIC_MODEL,
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=LLM_TEMPERATURE,
        )
    elif OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENAI_MODEL,
            openai_api_key=OPENAI_API_KEY,
            temperature=LLM_TEMPERATURE,
        )
    else:
        raise EnvironmentError("No LLM API key configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env")


def supervisor_route(state: RiskLensState) -> RiskLensState:
    """LangGraph node: LLM-based routing decision — picks which agents to activate."""
    portfolio = state.get("portfolio", [])
    analysis_date = state.get("analysis_date", "unknown")

    # Summarise portfolio for the LLM (avoid sending the full JSON to keep prompt short)
    asset_classes = list({p["asset_class"] for p in portfolio})
    sectors = list({p["sector"] for p in portfolio})
    tickers = [p["ticker"] for p in portfolio]

    user_message = f"""Portfolio summary for {analysis_date}:
- Positions: {len(portfolio)} holdings
- Tickers: {', '.join(tickers)}
- Asset classes: {', '.join(asset_classes)}
- Sectors: {', '.join(sectors)}

Which specialist agents should be activated for a complete risk analysis?"""

    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = llm.invoke(messages)
        content = response.content.strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        routing = json.loads(content)
        active_agents = routing.get("agents", ["market_risk", "credit_risk", "operational_risk", "macro", "rag"])
        reasoning = routing.get("reasoning", "")
        logger.info("Supervisor activated agents: %s | Reasoning: %s", active_agents, reasoning)

    except Exception as exc:
        # On any LLM failure, default to running all agents — fail safe
        logger.warning("Supervisor LLM call failed (%s) — defaulting to all agents", exc)
        active_agents = ["market_risk", "credit_risk", "operational_risk", "macro", "rag"]

    return {
        **state,
        "active_agents": active_agents,
        "completed_agents": [],
        # Initialise all output fields to empty defaults
        "market_risk_flags": [],
        "credit_risk_flags": [],
        "operational_risk_flags": [],
        "macro_risk_flags": [],
        "sanctions_flags": [],
        "regulatory_citations": [],
        "volatility_by_ticker": {},
        "max_drawdown_by_ticker": {},
        "beta_by_ticker": {},
        "sortino_by_ticker": {},
        "ma_signals": {},
        "credit_scores": {},
        "credit_news": {},
        "operational_events": {},
        "macro_indicators": {},
        "yield_curve_status": "unknown",
        "recession_probability": "unknown",
        "hitl_triggered": False,
        "messages": state.get("messages", []),
    }


def aggregator_node(state: RiskLensState) -> RiskLensState:
    """LangGraph node: weighted composite score (market 40%, credit 35%, op 15%, macro 10%)."""
    market_score = state.get("market_risk_score") or 0.0
    credit_score = state.get("credit_risk_score") or 0.0
    op_score = state.get("operational_risk_score") or 0.0
    macro_score = state.get("macro_risk_score") or 0.0

    composite = (
        market_score * RISK_WEIGHTS["market_risk"]
        + credit_score * RISK_WEIGHTS["credit_risk"]
        + op_score * RISK_WEIGHTS["operational_risk"]
        + macro_score * RISK_WEIGHTS["macro_risk"]
    )

    if composite >= RISK_LEVEL_THRESHOLDS["CRITICAL"]:
        level = "CRITICAL"
    elif composite >= RISK_LEVEL_THRESHOLDS["HIGH"]:
        level = "HIGH"
    elif composite >= RISK_LEVEL_THRESHOLDS["MEDIUM"]:
        level = "MEDIUM"
    else:
        level = "LOW"

    logger.info(
        "Risk aggregation: market=%.1f credit=%.1f op=%.1f macro=%.1f → composite=%.1f (%s)",
        market_score, credit_score, op_score, macro_score, composite, level,
    )

    # Determine trend vs previous briefing if available
    risk_trend = _compute_risk_trend(composite)

    return {
        **state,
        "overall_risk_score": round(composite, 2),
        "overall_risk_level": level,
        "risk_trend": risk_trend,
        # Trigger HITL for HIGH or CRITICAL risk levels
        "hitl_triggered": level in ("HIGH", "CRITICAL"),
    }


def _compute_risk_trend(current_score: float) -> str:
    """Compare current score to last briefing. Returns IMPROVING, STABLE, or DETERIORATING."""
    import glob
    import os
    from core.config import BRIEFINGS_DIR

    try:
        briefing_files = sorted(glob.glob(os.path.join(BRIEFINGS_DIR, "*.json")))
        if not briefing_files:
            return "STABLE"  # No history to compare

        with open(briefing_files[-1], "r") as fh:
            last_briefing = json.load(fh)

        previous_score = last_briefing.get("overall_risk", {}).get("score")
        if previous_score is None:
            return "STABLE"

        delta = current_score - previous_score
        if delta > 5:
            return "DETERIORATING"
        if delta < -5:
            return "IMPROVING"
        return "STABLE"

    except Exception:
        return "STABLE"


def should_activate_hitl(state: RiskLensState) -> str:
    """Route to HITL for HIGH/CRITICAL, skip straight to report otherwise."""
    if state.get("hitl_triggered"):
        return "hitl_review"
    return "report"
