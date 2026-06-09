# agents/report_agent.py
"""Report Agent — assembles the final briefing JSON and generates the executive summary via LLM."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from core.config import BRIEFINGS_DIR
from core.state import RiskLensState
from agents.supervisor import _get_llm

logger = logging.getLogger(__name__)

REPORT_SYSTEM_PROMPT = """You are a Chief Risk Officer at a major Canadian bank.
Write a concise executive summary (3-4 sentences) for an internal Risk Intelligence Briefing.

You will be given structured risk data. Your summary must:
1. State the overall risk level and score
2. Identify the single most important risk driver
3. Mention any regulatory concern (Basel III)
4. Recommend the most urgent action

Score interpretation — all pillar scores are RISK scores (higher = more risk):
- Pillar score 0-30: LOW risk — do NOT describe as concerning or critical
- Pillar score 31-55: MEDIUM risk — "warrants monitoring" or "elevated"
- Pillar score 56-75: HIGH risk — "requires attention"
- Pillar score 76+: CRITICAL risk — "immediate action"

Language calibration:
- Only use "critical credit risk" if the credit pillar score is above 70
- Op risk from indirect ETF sanctions: "minor indirect sanctions exposure" — NOT an escalation item
- Only use "immediate escalation" or "urgent" for pillar scores above 70

Write in formal, direct bank risk report style. No markdown, no bullet points — prose only."""


def _analyst_review_label(state: RiskLensState) -> str:
    """Return a human-readable analyst review status for the PDF cover page."""
    hitl_triggered = state.get("hitl_triggered", False)
    analyst_approved = state.get("analyst_approved")
    analyst_notes = state.get("analyst_notes")
    overall_level = state.get("overall_risk_level", "LOW")

    if not hitl_triggered:
        return f"Not required (Overall Risk: {overall_level})"
    if analyst_approved:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        note_flag = " — notes on file" if analyst_notes else ""
        return f"Yes — analyst approved {ts}{note_flag}"
    return "Pending analyst approval"


def report_agent(state: RiskLensState) -> RiskLensState:
    """LangGraph node: assemble final briefing, generate executive summary, save to disk."""
    logger.info("Report agent generating final briefing")

    analysis_date = state.get("analysis_date", datetime.today().strftime("%Y-%m-%d"))
    overall_score = state.get("overall_risk_score", 0.0)
    overall_level = state.get("overall_risk_level", "UNKNOWN")
    risk_trend = state.get("risk_trend", "STABLE")

    # Assemble risk breakdown
    risk_breakdown = _build_risk_breakdown(state)

    # Rank top risks
    top_risks = _rank_top_risks(state)

    # Generate recommended actions
    recommendations = _generate_recommendations(state)

    # Build per-position details
    position_details = _build_position_details(state)

    # Generate executive summary via LLM
    exec_summary = _generate_executive_summary(
        overall_score=overall_score,
        overall_level=overall_level,
        top_risks=top_risks,
        risk_breakdown=risk_breakdown,
    )

    briefing: Dict[str, Any] = {
        "metadata": {
            "date": analysis_date,
            "portfolio_id": "DEMO-001",
            "generated_by": "RiskLens v1.0",
            "analyst_reviewed": _analyst_review_label(state),
            "generated_at": datetime.now().isoformat(),
        },
        "executive_summary": exec_summary,
        "overall_risk": {
            "score": overall_score,
            "level": overall_level,
            "trend": risk_trend,
        },
        "risk_breakdown": risk_breakdown,
        "top_risks": top_risks,
        "recommended_actions": recommendations,
        "regulatory_citations": state.get("regulatory_citations", []),
        "analyst_notes": state.get("analyst_notes"),
        "position_details": position_details,
        "var_summary": {
            "var_95_historical": state.get("var_95"),
            "var_99_historical": state.get("var_99"),
            "var_95_parametric": state.get("var_parametric_95"),
            "var_99_parametric": state.get("var_parametric_99"),
            "cvar_95": state.get("cvar_95"),
            "portfolio_volatility": state.get("portfolio_volatility"),
            "hhi_concentration": state.get("hhi_concentration"),
        },
        "macro_summary": {
            "yield_curve_status": state.get("yield_curve_status", "unknown"),
            "recession_probability": state.get("recession_probability", "unknown"),
            "key_indicators": _format_macro_indicators(state.get("macro_indicators", {})),
        },
    }

    # Persist briefing to disk
    _save_briefing(briefing, analysis_date)

    logger.info(
        "Briefing complete: %s risk (score=%.1f), %d top risks, %d recommendations",
        overall_level, overall_score, len(top_risks), len(recommendations),
    )

    return {
        "final_briefing": briefing,
        "completed_agents": ["report"],
    }


def _build_risk_breakdown(state: RiskLensState) -> Dict[str, Any]:
    """Build the four-pillar risk breakdown dict."""
    def level_from_score(score: Optional[float]) -> str:
        if score is None: return "UNKNOWN"
        if score >= 75: return "CRITICAL"
        if score >= 55: return "HIGH"
        if score >= 35: return "MEDIUM"
        return "LOW"

    market_score = state.get("market_risk_score", 0)
    credit_score = state.get("credit_risk_score", 0)
    op_score = state.get("operational_risk_score", 0)
    macro_score = state.get("macro_risk_score", 0)

    return {
        "market_risk": {
            "score": round(market_score, 1),
            "level": level_from_score(market_score),
            "flags": state.get("market_risk_flags", []),
        },
        "credit_risk": {
            "score": round(credit_score, 1),
            "level": level_from_score(credit_score),
            "flags": state.get("credit_risk_flags", []),
        },
        "operational_risk": {
            "score": round(op_score, 1),
            "level": level_from_score(op_score),
            "flags": state.get("operational_risk_flags", []) + state.get("sanctions_flags", []),
        },
        "macro_risk": {
            "score": round(macro_score, 1),
            "level": level_from_score(macro_score),
            "flags": state.get("macro_risk_flags", []),
        },
    }


def _rank_top_risks(state: RiskLensState) -> List[Dict[str, Any]]:
    """Top 5 flags across pillars — CRITICAL first, then HIGH, with regulatory refs."""
    all_flags = []
    for pillar, flags in [
        ("Market Risk", state.get("market_risk_flags", [])),
        ("Credit Risk", state.get("credit_risk_flags", [])),
        ("Operational Risk", state.get("operational_risk_flags", []) + state.get("sanctions_flags", [])),
        ("Macro Risk", state.get("macro_risk_flags", [])),
    ]:
        for flag in flags:
            if "CRITICAL" in flag:
                severity = 4
            elif "HIGH" in flag:
                severity = 3
            elif "LOW INDIRECT" in flag or "INDIRECT SANCTIONS EXPOSURE (LOW)" in flag:
                severity = 1  # genuinely low — indirect ETF sanctions only
            else:
                severity = 2  # MEDIUM is the default for untagged market/credit flags
            all_flags.append((severity, pillar, flag))

    all_flags.sort(key=lambda x: -x[0])

    citations = state.get("regulatory_citations", [])
    top_risks = []
    for i, (severity, pillar, flag) in enumerate(all_flags[:5], 1):
        reg_ref = citations[i - 1][:200] if i <= len(citations) else "Basel III Pillar 1 — Risk-Weighted Assets"
        top_risks.append({
            "rank": i,
            "risk": flag,
            "driver": pillar,
            "severity": "CRITICAL" if severity == 4 else ("HIGH" if severity == 3 else ("MEDIUM" if severity == 2 else "LOW")),
            "regulatory_ref": reg_ref,
        })

    return top_risks


def _generate_recommendations(state: RiskLensState) -> List[Dict[str, Any]]:
    """Generate actionable recommendations based on detected risk flags."""
    recommendations = []
    flags = (
        state.get("market_risk_flags", [])
        + state.get("credit_risk_flags", [])
        + state.get("operational_risk_flags", [])
        + state.get("macro_risk_flags", [])
    )

    # VaR breach → position reduction
    if any("VaR BREACH" in f for f in flags):
        recommendations.append({
            "action": "Review and reduce highest-volatility positions",
            "rationale": "Portfolio 99% VaR exceeds 2.5% threshold — reduce tail risk exposure",
            "urgency": "HIGH",
            "pillar": "Market Risk",
        })

    # Yield curve inversion + TLT → duration reduction
    if any("DURATION RISK" in f or "YIELD CURVE INVERTED" in f for f in flags):
        recommendations.append({
            "action": "Reduce duration exposure — consider trimming TLT position by 20-30%",
            "rationale": "Inverted yield curve creates mark-to-market risk for long-duration bonds",
            "urgency": "HIGH",
            "pillar": "Market Risk / Macro",
        })

    # High volatility individual positions
    if any("HIGH VOLATILITY" in f or "EXTREME VOLATILITY" in f for f in flags):
        recommendations.append({
            "action": "Apply position limits to high-volatility holdings",
            "rationale": "Individual position volatility exceeds risk limits — may breach VaR thresholds",
            "urgency": "MEDIUM",
            "pillar": "Market Risk",
        })

    # Death cross signal
    if any("Death Cross" in f for f in flags):
        recommendations.append({
            "action": "Review technical stop-loss levels for death-cross flagged positions",
            "rationale": "50-day MA crossing below 200-day MA signals potential trend reversal",
            "urgency": "MEDIUM",
            "pillar": "Market Risk",
        })

    # Credit flags
    if any("CREDIT RISK" in f for f in flags):
        recommendations.append({
            "action": "Initiate enhanced credit monitoring for flagged positions",
            "rationale": "Proxy credit scores below threshold — increase review frequency",
            "urgency": "HIGH",
            "pillar": "Credit Risk",
        })

    # Operational risk — scale urgency to actual severity
    if any("CRITICAL OPERATIONAL" in f for f in flags):
        recommendations.append({
            "action": "Escalate to Legal and Compliance immediately",
            "rationale": "Critical operational risk event confirmed — regulatory reporting may be required",
            "urgency": "CRITICAL",
            "pillar": "Operational Risk",
        })
    elif any("HIGH OPERATIONAL" in f or "MEDIUM DIRECT SANCTIONS" in f for f in flags):
        recommendations.append({
            "action": "Review and document direct sanctions or high-severity operational exposure",
            "rationale": "Direct sanctions exposure or high-severity operational event requires compliance review",
            "urgency": "MEDIUM",
            "pillar": "Operational Risk",
        })
    elif any("LOW INDIRECT SANCTIONS" in f for f in flags):
        recommendations.append({
            "action": "Monitor for changes in ETF constituent sanctions exposure at next review cycle",
            "rationale": "Indirect index-level sanctions exposure is low-severity — routine monitoring is sufficient",
            "urgency": "LOW",
            "pillar": "Operational Risk",
        })

    # Macro recession signal
    if any("RECESSION" in f for f in flags):
        recommendations.append({
            "action": "Increase portfolio defensiveness — overweight bonds/gold, reduce cyclicals",
            "rationale": "Elevated recession probability signal — defensive positioning reduces downside",
            "urgency": "MEDIUM",
            "pillar": "Macro Risk",
        })

    if not recommendations:
        recommendations.append({
            "action": "Maintain current portfolio positioning",
            "rationale": "No material risk threshold breaches detected — routine monitoring continues",
            "urgency": "LOW",
            "pillar": "All",
        })

    return recommendations


def _build_position_details(state: RiskLensState) -> List[Dict[str, Any]]:
    """Build per-position summary table for the briefing appendix."""
    portfolio = state.get("portfolio", [])
    details = []
    for pos in portfolio:
        ticker = pos["ticker"]
        details.append({
            "ticker": ticker,
            "name": pos.get("name", ticker),
            "weight": pos["weight"],
            "asset_class": pos["asset_class"],
            "sector": pos["sector"],
            "annualised_vol": state.get("volatility_by_ticker", {}).get(ticker),
            "max_drawdown": state.get("max_drawdown_by_ticker", {}).get(ticker),
            "beta": state.get("beta_by_ticker", {}).get(ticker),
            "sortino": state.get("sortino_by_ticker", {}).get(ticker),
            "ma_signal": state.get("ma_signals", {}).get(ticker),
            "credit_score": state.get("credit_scores", {}).get(ticker, {}).get("credit_score"),
            "credit_grade": state.get("credit_scores", {}).get(ticker, {}).get("credit_grade"),
            "op_risk_events": len(state.get("operational_events", {}).get(ticker, [])),
        })
    return details


def _format_macro_indicators(indicators: Dict) -> List[Dict[str, str]]:
    """Format raw FRED indicator dict for display in the briefing."""
    from core.config import FRED_SERIES
    # Series that are already expressed as rates/percentages vs index levels
    pct_series = {"CPIAUCSL", "GDPC1", "DFF", "GS10", "GS2", "T10Y2Y", "UNRATE", "BAMLH0A0HYM2"}
    formatted = []
    for series_id, label in FRED_SERIES.items():
        value = indicators.get(series_id)
        if value is None:
            formatted_value = "N/A"
        elif series_id in pct_series:
            formatted_value = f"{value:.2f}%"
        else:
            formatted_value = f"{value:.2f}"
        formatted.append({
            "indicator": label,
            "series_id": series_id,
            "value": formatted_value,
        })
    return formatted


def _generate_executive_summary(
    overall_score: float,
    overall_level: str,
    top_risks: List[Dict],
    risk_breakdown: Dict,
) -> str:
    """Use LLM to generate a concise executive summary paragraph."""
    top_risk_text = top_risks[0]["risk"] if top_risks else "No material risks detected"
    highest_pillar = max(
        risk_breakdown.items(),
        key=lambda x: x[1].get("score", 0),
    )[0].replace("_", " ").title() if risk_breakdown else "Unknown"

    prompt = f"""Generate a 3-4 sentence executive summary for this risk briefing:

Overall Risk Score: {overall_score:.1f}/100 — {overall_level}
Highest Risk Pillar: {highest_pillar}
Primary Risk Driver: {top_risk_text}
Market Risk Score: {risk_breakdown.get('market_risk', {}).get('score', 0):.1f}
Credit Risk Score: {risk_breakdown.get('credit_risk', {}).get('score', 0):.1f}
Operational Risk Score: {risk_breakdown.get('operational_risk', {}).get('score', 0):.1f}
Macro Risk Score: {risk_breakdown.get('macro_risk', {}).get('score', 0):.1f}"""

    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as exc:
        logger.warning("Executive summary LLM call failed: %s", exc)
        return (
            f"Portfolio risk assessment for {datetime.today().strftime('%B %d, %Y')} indicates an "
            f"overall risk score of {overall_score:.1f}/100, classified as {overall_level}. "
            f"The primary risk driver is identified in {highest_pillar}. "
            "Please review the detailed findings in the sections below and consult with the risk committee."
        )


def _save_briefing(briefing: Dict[str, Any], analysis_date: str) -> None:
    """Save briefing JSON to disk for trend analysis in future runs."""
    import os
    os.makedirs(BRIEFINGS_DIR, exist_ok=True)
    filename = f"briefing_{analysis_date}_{datetime.now().strftime('%H%M%S')}.json"
    filepath = os.path.join(BRIEFINGS_DIR, filename)
    try:
        with open(filepath, "w") as fh:
            json.dump(briefing, fh, indent=2, default=str)
        logger.info("Briefing saved to %s", filepath)
    except Exception as exc:
        logger.warning("Failed to save briefing: %s", exc)
