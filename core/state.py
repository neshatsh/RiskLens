# core/state.py
"""Shared LangGraph state — single TypedDict that flows through the entire pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage


class RiskLensState(TypedDict):

    # input
    portfolio: List[Dict[str, Any]]
    analysis_date: str

    # routing
    active_agents: List[str]
    completed_agents: Annotated[List[str], operator.add]  # add reducer so parallel agents can each append

    # market risk
    var_95: Optional[float]
    var_99: Optional[float]
    cvar_95: Optional[float]
    var_parametric_95: Optional[float]
    var_parametric_99: Optional[float]
    portfolio_volatility: Optional[float]
    hhi_concentration: Optional[float]
    volatility_by_ticker: Dict[str, float]
    max_drawdown_by_ticker: Dict[str, float]
    beta_by_ticker: Dict[str, float]
    sortino_by_ticker: Dict[str, float]
    ma_signals: Dict[str, str]
    market_risk_flags: List[str]
    market_risk_score: Optional[float]

    # credit risk
    credit_scores: Dict[str, Any]
    credit_news: Dict[str, List[str]]
    credit_risk_flags: List[str]
    credit_risk_score: Optional[float]

    # operational risk
    operational_events: Dict[str, List[Dict[str, str]]]
    sanctions_flags: List[str]
    operational_risk_flags: List[str]
    operational_risk_score: Optional[float]

    # macro
    macro_indicators: Dict[str, Any]
    yield_curve_status: str
    recession_probability: str
    macro_risk_flags: List[str]
    macro_risk_score: Optional[float]

    # rag
    regulatory_citations: List[str]

    # aggregate
    overall_risk_score: Optional[float]
    overall_risk_level: Optional[str]
    risk_trend: Optional[str]

    # hitl
    hitl_triggered: bool
    analyst_notes: Optional[str]
    analyst_approved: Optional[bool]

    # output
    final_briefing: Optional[Dict[str, Any]]
    pdf_path: Optional[str]
    messages: List[BaseMessage]
