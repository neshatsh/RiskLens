# tests/test_graph.py
"""
Integration tests for the LangGraph graph structure.

Tests graph compilation, node wiring, and routing logic without
making real API calls. Mocks all external data sources.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.portfolio import SAMPLE_PORTFOLIO
from graph.builder import build_graph, get_initial_state


class TestGraphStructure:
    def test_graph_compiles_without_checkpointer(self):
        """Graph should compile cleanly in test mode (no SQLite)."""
        graph = build_graph(use_checkpointer=False)
        assert graph is not None

    def test_initial_state_has_required_fields(self):
        """Initial state should include all required TypedDict fields."""
        state = get_initial_state(SAMPLE_PORTFOLIO, "2025-06-09")
        required = ["portfolio", "analysis_date", "active_agents", "completed_agents",
                    "market_risk_flags", "hitl_triggered", "messages"]
        for field in required:
            assert field in state, f"Missing required state field: {field}"

    def test_initial_state_portfolio_loaded(self):
        """Portfolio should be populated from SAMPLE_PORTFOLIO."""
        state = get_initial_state(SAMPLE_PORTFOLIO, "2025-06-09")
        assert len(state["portfolio"]) == 10
        assert state["hitl_triggered"] is False
        assert state["messages"] != []


class TestSupervisorRouting:
    def test_should_activate_hitl_for_high_risk(self):
        """Conditional edge should route to hitl_review for HIGH risk."""
        from agents.supervisor import should_activate_hitl
        state = get_initial_state(SAMPLE_PORTFOLIO, "2025-06-09")
        state["hitl_triggered"] = True
        assert should_activate_hitl(state) == "hitl_review"

    def test_should_skip_hitl_for_low_risk(self):
        """Conditional edge should route directly to report for LOW risk."""
        from agents.supervisor import should_activate_hitl
        state = get_initial_state(SAMPLE_PORTFOLIO, "2025-06-09")
        state["hitl_triggered"] = False
        assert should_activate_hitl(state) == "report"


class TestPortfolioLoader:
    def test_sample_portfolio_weights_sum_to_one(self):
        """Portfolio weights must sum to ~1.0."""
        total = sum(p["weight"] for p in SAMPLE_PORTFOLIO)
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total:.4f}"

    def test_sample_portfolio_has_all_required_keys(self):
        """Each position must have ticker, weight, asset_class, sector."""
        required = {"ticker", "weight", "asset_class", "sector"}
        for pos in SAMPLE_PORTFOLIO:
            missing = required - pos.keys()
            assert not missing, f"Position {pos.get('ticker')} missing: {missing}"

    def test_load_portfolio_returns_sample_when_no_file(self):
        """load_portfolio() with None path should return SAMPLE_PORTFOLIO."""
        from core.portfolio import load_portfolio
        portfolio = load_portfolio(None)
        assert len(portfolio) == len(SAMPLE_PORTFOLIO)
