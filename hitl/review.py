# hitl/review.py
"""HITL review node — calls interrupt() to pause the graph when risk is HIGH or CRITICAL."""

from __future__ import annotations

import logging
from typing import Any, Dict

from langgraph.types import interrupt

from core.state import RiskLensState

logger = logging.getLogger(__name__)


def hitl_review_node(state: RiskLensState) -> RiskLensState:
    """LangGraph node: pause at interrupt() for analyst review; resume with analyst_notes + analyst_approved."""
    risk_level = state.get("overall_risk_level", "UNKNOWN")
    risk_score = state.get("overall_risk_score", 0.0)

    # Build a concise preliminary summary for the analyst
    all_flags = (
        state.get("market_risk_flags", [])[:3]
        + state.get("credit_risk_flags", [])[:2]
        + state.get("operational_risk_flags", [])[:2]
        + state.get("macro_risk_flags", [])[:2]
    )

    interrupt_payload = {
        "message": (
            f"RISK ALERT — {risk_level} RISK DETECTED\n"
            f"Composite Risk Score: {risk_score:.1f}/100\n\n"
            f"Top flags requiring review:\n"
            + "\n".join(f"  • {flag}" for flag in all_flags[:7])
            + "\n\nPlease review the preliminary findings and provide analyst notes before the final briefing is generated."
        ),
        "overall_risk_level": risk_level,
        "overall_risk_score": risk_score,
        "market_risk_score": state.get("market_risk_score"),
        "credit_risk_score": state.get("credit_risk_score"),
        "operational_risk_score": state.get("operational_risk_score"),
        "macro_risk_score": state.get("macro_risk_score"),
        "top_flags": all_flags[:7],
        "instructions": "Set analyst_notes (str) and analyst_approved (bool) in the state update to resume.",
    }

    logger.info("HITL interrupt triggered for %s risk (score=%.1f)", risk_level, risk_score)

    # This is the real LangGraph interrupt — execution halts here
    # The analyst's response is passed back via .invoke() with Command(resume=...) or state update
    analyst_input = interrupt(interrupt_payload)

    # When resumed, analyst_input contains the values passed back by the caller
    analyst_notes = None
    analyst_approved = True

    if isinstance(analyst_input, dict):
        analyst_notes = analyst_input.get("analyst_notes")
        analyst_approved = analyst_input.get("analyst_approved", True)

    logger.info(
        "HITL review completed: approved=%s, notes_provided=%s",
        analyst_approved, analyst_notes is not None,
    )

    return {
        **state,
        "analyst_notes": analyst_notes,
        "analyst_approved": analyst_approved,
    }


def format_hitl_summary(state: RiskLensState) -> str:
    """Format a plain-text preliminary risk summary for the analyst review panel."""
    lines = [
        f"{'='*60}",
        f"PRELIMINARY RISK ASSESSMENT — ANALYST REVIEW REQUIRED",
        f"{'='*60}",
        f"Overall Risk Level: {state.get('overall_risk_level', 'UNKNOWN')}",
        f"Composite Risk Score: {state.get('overall_risk_score', 0):.1f} / 100",
        f"",
        f"PILLAR SCORES:",
        f"  Market Risk:      {state.get('market_risk_score', 0):.1f}/100",
        f"  Credit Risk:      {state.get('credit_risk_score', 0):.1f}/100",
        f"  Operational Risk: {state.get('operational_risk_score', 0):.1f}/100",
        f"  Macro Risk:       {state.get('macro_risk_score', 0):.1f}/100",
        f"",
    ]

    all_flags = (
        state.get("market_risk_flags", [])
        + state.get("credit_risk_flags", [])
        + state.get("operational_risk_flags", [])
        + state.get("macro_risk_flags", [])
    )

    if all_flags:
        lines.append("KEY FLAGS:")
        for flag in all_flags[:10]:
            lines.append(f"  ⚠  {flag}")

    lines += [
        f"",
        f"{'='*60}",
        f"Please enter analyst notes and approve/reject to continue.",
        f"{'='*60}",
    ]

    return "\n".join(lines)
