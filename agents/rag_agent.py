# agents/rag_agent.py
"""RAG Agent — retrieves Basel III/IV passages relevant to the flags raised by other agents."""

from __future__ import annotations

import logging
from typing import List

from core.state import RiskLensState
from rag.retriever import build_rag_query_from_flags, retrieve_regulatory_context

logger = logging.getLogger(__name__)


def rag_agent(state: RiskLensState) -> RiskLensState:
    """LangGraph node: synthesise a query from risk flags and retrieve Basel III passages."""
    if "rag" not in state.get("active_agents", []):
        logger.info("RAG agent not activated — skipping")
        return {"completed_agents": ["rag"]}

    market_flags = state.get("market_risk_flags", [])
    credit_flags = state.get("credit_risk_flags", [])
    op_flags = state.get("operational_risk_flags", [])
    macro_flags = state.get("macro_risk_flags", [])

    logger.info(
        "RAG agent building query from %d flags (%d market, %d credit, %d op, %d macro)",
        len(market_flags) + len(credit_flags) + len(op_flags) + len(macro_flags),
        len(market_flags), len(credit_flags), len(op_flags), len(macro_flags),
    )

    query = build_rag_query_from_flags(market_flags, credit_flags, op_flags, macro_flags)
    logger.debug("RAG query: %s", query)

    try:
        citations = retrieve_regulatory_context(query)
        logger.info("RAG retrieved %d regulatory passages", len(citations))
    except Exception as exc:
        logger.error("RAG retrieval failed: %s", exc)
        citations = [
            "[RAG unavailable — ensure OPENAI_API_KEY is set and run: python -m rag.ingest]"
        ]

    return {
        "regulatory_citations": citations,
        "completed_agents": ["rag"],
    }
