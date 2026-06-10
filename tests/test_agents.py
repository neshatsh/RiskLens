# tests/test_agents.py
"""
Unit tests for individual agents using mock state.

Tests run without any API keys by mocking external calls.
Each agent is tested as a pure function: given a state dict, verify outputs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from core.portfolio import SAMPLE_PORTFOLIO
from graph.builder import get_initial_state


def _base_state() -> dict:
    """Minimal valid state for agent tests."""
    state = get_initial_state(SAMPLE_PORTFOLIO, "2025-06-09")
    # Pre-populate fields that later-stage agents expect from earlier agents
    state["active_agents"] = ["market_risk", "credit_risk", "operational_risk", "macro", "rag"]
    return state


# market risk agent tests

class TestMarketRiskAgent:
    @patch("agents.market_risk_agent.get_all_market_metrics")
    @patch("agents.market_risk_agent.compute_all_var_metrics")
    def test_populates_volatility_fields(self, mock_var, mock_metrics):
        """Agent should populate volatility_by_ticker and market_risk_flags."""
        import numpy as np

        mock_metrics.return_value = {
            "RY.TO": {
                "annualised_vol": 0.18,
                "max_drawdown": -0.12,
                "beta": 0.9,
                "sortino": 1.2,
                "ma_signal": "above_200ma",
                "returns": pd.Series(np.random.normal(0, 0.01, 252)),
                "prices": pd.Series(range(100, 352)),
            }
        }
        mock_var.return_value = {
            "var_95_hist": 0.012,
            "var_99_hist": 0.018,
            "var_95_param": 0.011,
            "var_99_param": 0.017,
            "cvar_95": 0.022,
            "portfolio_vol": 0.15,
            "hhi": 0.12,
        }

        from agents.market_risk_agent import market_risk_agent
        state = _base_state()
        result = market_risk_agent(state)

        assert "RY.TO" in result["volatility_by_ticker"]
        assert result["var_99"] == pytest.approx(0.018)
        assert "market_risk" in result["completed_agents"]

    def test_skips_when_not_activated(self):
        """Agent should be a no-op if not in active_agents."""
        from agents.market_risk_agent import market_risk_agent
        state = _base_state()
        state["active_agents"] = []
        result = market_risk_agent(state)
        assert result["market_risk_score"] == 0.0
        assert "market_risk" in result["completed_agents"]

    @patch("agents.market_risk_agent.get_all_market_metrics")
    def test_handles_empty_data_gracefully(self, mock_metrics):
        """Agent should not crash when market data fetch returns empty."""
        mock_metrics.return_value = {}
        from agents.market_risk_agent import market_risk_agent
        state = _base_state()
        result = market_risk_agent(state)
        assert result["market_risk_score"] > 0
        assert len(result["market_risk_flags"]) > 0


# credit risk agent tests

class TestCreditRiskAgent:
    @patch("agents.credit_risk_agent.fetch_company_news")
    def test_scores_all_equity_positions(self, mock_news):
        """Credit scores should be generated for all equity positions."""
        mock_news.return_value = ["Stable earnings beat expectations"]
        from agents.credit_risk_agent import credit_risk_agent
        state = _base_state()
        # Set some volatility data that the agent reads
        state["volatility_by_ticker"] = {p["ticker"]: 0.20 for p in SAMPLE_PORTFOLIO}
        state["max_drawdown_by_ticker"] = {p["ticker"]: -0.10 for p in SAMPLE_PORTFOLIO}
        state["beta_by_ticker"] = {p["ticker"]: 1.0 for p in SAMPLE_PORTFOLIO}

        result = credit_risk_agent(state)

        equity_tickers = [p["ticker"] for p in SAMPLE_PORTFOLIO if p["asset_class"] == "equity"]
        for ticker in equity_tickers:
            assert ticker in result["credit_scores"], f"{ticker} missing from credit_scores"

    @patch("agents.credit_risk_agent.fetch_company_news")
    def test_flags_severe_news(self, mock_news):
        """SEVERE news should trigger credit risk flag."""
        mock_news.return_value = ["Company faces DOJ indictment for fraud", "SEC investigation opened"]
        from agents.credit_risk_agent import credit_risk_agent
        state = _base_state()
        state["volatility_by_ticker"] = {p["ticker"]: 0.20 for p in SAMPLE_PORTFOLIO}
        state["max_drawdown_by_ticker"] = {p["ticker"]: -0.10 for p in SAMPLE_PORTFOLIO}
        state["beta_by_ticker"] = {p["ticker"]: 1.0 for p in SAMPLE_PORTFOLIO}

        result = credit_risk_agent(state)
        assert len(result["credit_risk_flags"]) > 0


# aggregator tests

class TestAggregator:
    def test_critical_risk_triggers_hitl(self):
        """Composite score >= 75 should trigger HITL."""
        from agents.supervisor import aggregator_node
        state = _base_state()
        state["market_risk_score"] = 90.0
        state["credit_risk_score"] = 80.0
        state["operational_risk_score"] = 70.0
        state["macro_risk_score"] = 60.0
        result = aggregator_node(state)
        assert result["hitl_triggered"] is True
        assert result["overall_risk_level"] == "CRITICAL"

    def test_low_risk_no_hitl(self):
        """Composite score < 35 should not trigger HITL."""
        from agents.supervisor import aggregator_node
        state = _base_state()
        state["market_risk_score"] = 20.0
        state["credit_risk_score"] = 15.0
        state["operational_risk_score"] = 10.0
        state["macro_risk_score"] = 5.0
        result = aggregator_node(state)
        assert result["hitl_triggered"] is False
        assert result["overall_risk_level"] == "LOW"

    def test_composite_score_formula(self):
        """Verify the Basel III weighting formula: 40/35/15/10."""
        from agents.supervisor import aggregator_node
        state = _base_state()
        state["market_risk_score"] = 100.0
        state["credit_risk_score"] = 0.0
        state["operational_risk_score"] = 0.0
        state["macro_risk_score"] = 0.0
        result = aggregator_node(state)
        # Should equal 100 * 0.40 = 40
        assert result["overall_risk_score"] == pytest.approx(40.0)
