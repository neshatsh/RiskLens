# graph/builder.py
"""Builds and compiles the RiskLens StateGraph — supervisor fans out to parallel agents, then rag → aggregator → HITL → report."""

from __future__ import annotations

import logging
from typing import List, Literal

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.types import Send

from agents.credit_risk_agent import credit_risk_agent
from agents.macro_agent import macro_agent
from agents.market_risk_agent import market_risk_agent
from agents.operational_risk_agent import operational_risk_agent
from agents.rag_agent import rag_agent
from agents.report_agent import report_agent
from agents.supervisor import aggregator_node, should_activate_hitl, supervisor_route
from core.state import RiskLensState
from graph.checkpointer import get_checkpointer
from hitl.review import hitl_review_node

logger = logging.getLogger(__name__)


def _route_after_supervisor(state: RiskLensState) -> List[Send]:
    """Fan out active agents in parallel using the Send API."""
    active_agents = state.get("active_agents", [])
    sends = []

    agent_map = {
        "market_risk":      "market_risk_node",
        "credit_risk":      "credit_risk_node",
        "operational_risk": "operational_risk_node",
        "macro":            "macro_node",
    }

    for agent_name in active_agents:
        if agent_name in agent_map:
            sends.append(Send(agent_map[agent_name], state))
            logger.debug("Dispatching %s via Send API", agent_name)

    # If no specialist agents were activated (shouldn't happen), go directly to rag
    if not sends:
        logger.warning("No specialist agents activated — routing directly to RAG")
        sends.append(Send("rag_node", state))

    return sends


def build_graph(use_checkpointer: bool = True):
    """Compile the RiskLens StateGraph. Pass use_checkpointer=False for tests."""
    workflow = StateGraph(RiskLensState)

    # Add all nodes
    workflow.add_node("supervisor", supervisor_route)
    workflow.add_node("market_risk_node", market_risk_agent)
    workflow.add_node("credit_risk_node", credit_risk_agent)
    workflow.add_node("operational_risk_node", operational_risk_agent)
    workflow.add_node("macro_node", macro_agent)
    workflow.add_node("rag_node", rag_agent)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("hitl_review", hitl_review_node)
    workflow.add_node("report", report_agent)

    # Entry point
    workflow.set_entry_point("supervisor")

    # Supervisor fans out to specialist agents in parallel
    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        ["market_risk_node", "credit_risk_node", "operational_risk_node", "macro_node"],
    )

    # All specialist agents converge to RAG (RAG needs their flags as input)
    for node in ["market_risk_node", "credit_risk_node", "operational_risk_node", "macro_node"]:
        workflow.add_edge(node, "rag_node")

    # RAG → aggregator → conditional HITL check
    workflow.add_edge("rag_node", "aggregator")

    workflow.add_conditional_edges(
        "aggregator",
        should_activate_hitl,
        {
            "hitl_review": "hitl_review",
            "report":      "report",
        },
    )

    # HITL review node → report (after analyst input)
    workflow.add_edge("hitl_review", "report")

    # Report is the terminal node
    workflow.add_edge("report", END)

    # Compile with or without checkpointer
    if use_checkpointer:
        checkpointer = get_checkpointer()
        # interrupt() inside hitl_review_node handles the pause — no interrupt_before needed
        graph = workflow.compile(checkpointer=checkpointer)
        logger.info("Graph compiled with SQLite checkpointer")
    else:
        graph = workflow.compile()
        logger.info("Graph compiled without checkpointer (test mode)")

    return graph


def get_initial_state(portfolio, analysis_date: str) -> RiskLensState:
    """Build a minimal initial state to start the graph."""
    return {
        "portfolio": portfolio,
        "analysis_date": analysis_date,
        "active_agents": [],
        "completed_agents": [],
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
        "analyst_notes": None,
        "analyst_approved": None,
        "var_95": None,
        "var_99": None,
        "cvar_95": None,
        "var_parametric_95": None,
        "var_parametric_99": None,
        "portfolio_volatility": None,
        "hhi_concentration": None,
        "market_risk_score": None,
        "credit_risk_score": None,
        "operational_risk_score": None,
        "macro_risk_score": None,
        "overall_risk_score": None,
        "overall_risk_level": None,
        "risk_trend": None,
        "final_briefing": None,
        "pdf_path": None,
        "messages": [HumanMessage(content=f"Begin risk analysis for {analysis_date}")],
    }
