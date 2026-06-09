# agents/credit_risk_agent.py
"""Credit Risk Agent — proxy credit scoring from market signals and news sentiment."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.config import CREDIT_GRADES, CREDIT_SCORE_CRITICAL, CREDIT_SCORE_HIGH_RISK
from core.portfolio import get_equity_positions
from core.state import RiskLensState
from tools.market_data import get_all_market_metrics
from tools.news_scanner import fetch_company_news, scan_credit_sentiment, scan_for_op_risk_events

logger = logging.getLogger(__name__)


def credit_risk_agent(state: RiskLensState) -> RiskLensState:
    """LangGraph node: proxy credit score from vol/drawdown/beta + news sentiment per position."""
    if "credit_risk" not in state.get("active_agents", []):
        logger.info("Credit risk agent not activated — skipping")
        return {"credit_risk_score": 0.0, "completed_agents": ["credit_risk"]}

    portfolio = state.get("portfolio", [])
    equity_positions = get_equity_positions(portfolio)

    logger.info("Credit risk agent running for %d equity positions", len(equity_positions))

    # Fetch market metrics directly — credit runs in parallel with market_risk so
    # state fields like volatility_by_ticker haven't been written yet when we execute.
    tickers = [p["ticker"] for p in equity_positions]
    market_metrics = get_all_market_metrics(tickers) if tickers else {}

    credit_scores: Dict[str, Any] = {}
    credit_news: Dict[str, List[str]] = {}
    flags: List[str] = []
    scores_list: List[float] = []

    for position in equity_positions:
        ticker = position["ticker"]
        company_name = position.get("name", ticker)

        # Fetch news headlines for sentiment analysis
        try:
            headlines = fetch_company_news(company_name, ticker, days_back=14)
        except Exception as exc:
            logger.warning("News fetch failed for %s: %s", ticker, exc)
            headlines = []

        credit_news[ticker] = headlines

        # Compute market-implied credit stress score (0 = stress, 100 = healthy)
        m = market_metrics.get(ticker, {})
        vol = m.get("annualised_vol") or 0.0
        drawdown = m.get("max_drawdown") or 0.0
        beta = m.get("beta") or 1.0
        ma_signal = m.get("ma_signal", "unknown")

        market_credit_score = _compute_market_credit_score(
            ticker=ticker,
            vol=vol,
            drawdown=drawdown,
            beta=beta,
            ma_signal=ma_signal,
        )

        # Compute news sentiment adjustment
        sentiment = scan_credit_sentiment(headlines)
        sentiment_adjustment = _sentiment_to_adjustment(sentiment)

        # Composite credit score (0-100)
        raw_score = market_credit_score + sentiment_adjustment
        final_score = max(0, min(100, raw_score))
        grade = _score_to_grade(int(final_score))

        # Risk flag if below threshold
        risk_flag = None
        if final_score < CREDIT_SCORE_CRITICAL:
            risk_flag = f"CRITICAL CREDIT RISK: {ticker} proxy score {final_score:.0f}/100 ({grade})"
            flags.append(risk_flag)
        elif final_score < CREDIT_SCORE_HIGH_RISK:
            risk_flag = f"ELEVATED CREDIT RISK: {ticker} proxy score {final_score:.0f}/100 ({grade})"
            flags.append(risk_flag)

        if sentiment == "SEVERE":
            flags.append(f"SEVERE NEWS SIGNAL: {ticker} — critical negative headlines detected")
        elif sentiment == "NEGATIVE":
            flags.append(f"NEGATIVE NEWS SENTIMENT: {ticker} — multiple adverse headlines in past 14 days")

        credit_scores[ticker] = {
            "credit_score": round(final_score, 1),
            "credit_grade": grade,
            "market_credit_score": round(market_credit_score, 1),
            "news_sentiment": sentiment,
            "key_signals": _extract_key_signals(
                vol=vol,
                drawdown=drawdown,
                beta=beta,
                ma_signal=ma_signal,
                sentiment=sentiment,
            ),
            "risk_flag": risk_flag,
            "headline_count": len(headlines),
        }
        scores_list.append(final_score)

    # For bond and commodity positions, assign a neutral credit score (credit risk less relevant)
    non_equity = [p for p in portfolio if p["asset_class"] != "equity"]
    for position in non_equity:
        ticker = position["ticker"]
        credit_scores[ticker] = {
            "credit_score": 75.0,
            "credit_grade": "A",
            "market_credit_score": 75.0,
            "news_sentiment": "NEUTRAL",
            "key_signals": ["non-equity position — standardised approach"],
            "risk_flag": None,
            "headline_count": 0,
        }
        scores_list.append(75.0)

    avg_score = sum(scores_list) / len(scores_list) if scores_list else 50.0
    # Invert: low credit score = high credit risk
    credit_risk_score = round(100 - avg_score, 2)

    logger.info(
        "Credit risk complete: %d flags, avg credit score=%.1f, risk score=%.1f",
        len(flags), avg_score, credit_risk_score,
    )

    return {
        "credit_scores": credit_scores,
        "credit_news": credit_news,
        "credit_risk_flags": flags,
        "credit_risk_score": credit_risk_score,
        "completed_agents": ["credit_risk"],
    }


def _compute_market_credit_score(
    ticker: str,
    vol: float,
    drawdown: float,
    beta: float,
    ma_signal: str = "unknown",
) -> float:
    """Proxy credit score 0–100. Higher = better credit quality.

    Calibration principle: volatility and drawdown are market risk signals, not
    solvency signals. A profitable, cash-rich growth company with high vol should
    land in speculative-grade (BB, ~45-55), not distressed (CCC, <30). Only
    genuinely distressed metrics — combined extreme vol + deep drawdown + high beta
    — should push below 40. CCC (<30) is reserved for companies near default.

    Target bands:
      BB  (40-54): beta>2 AND vol>50% AND drawdown<-40% — elevated market risk
      BBB (55-69): beta>1.5 OR vol>35% OR drawdown<-30% — below-average quality
      A   (70-84): normal metrics, reasonable vol, above 200MA
      AA  (85-95): low vol, defensive, strong MA signal
    """
    score = 80.0  # start at solid A-range baseline

    # Volatility — reflects credit spread widening risk, not insolvency
    # High vol for a growth company is expected; penalty is moderate, not catastrophic
    if vol > 0.55:
        score -= 14
    elif vol > 0.40:
        score -= 9
    elif vol > 0.30:
        score -= 5
    elif vol > 0.20:
        score -= 2

    # Drawdown — deep drawdowns signal balance sheet stress only in combination
    # -47% for Shopify is a market correction, not near-default
    dd_mag = abs(drawdown)
    if dd_mag > 0.55:
        score -= 14
    elif dd_mag > 0.40:
        score -= 8
    elif dd_mag > 0.25:
        score -= 4
    elif dd_mag > 0.12:
        score -= 2

    # Beta — systematic sensitivity, not a solvency indicator
    if beta > 2.0:
        score -= 6
    elif beta > 1.5:
        score -= 3
    elif beta > 1.2:
        score -= 1
    elif 0 < beta < 0.5:
        score += 4  # defensive positioning is a mild credit positive

    # MA signal — trend context adds/removes a few points at the margin
    if ma_signal == "death_cross":
        score -= 4
    elif ma_signal == "below_200ma":
        score -= 2
    elif ma_signal == "golden_cross":
        score += 3
    elif ma_signal == "above_200ma":
        score += 1

    return max(0.0, min(100.0, round(score, 1)))


def _sentiment_to_adjustment(sentiment: str) -> float:
    """Map news sentiment to a score adjustment (+/-)."""
    return {
        "POSITIVE": +5.0,
        "NEUTRAL":   0.0,
        "NEGATIVE": -10.0,
        "SEVERE":   -25.0,
    }.get(sentiment, 0.0)


def _score_to_grade(score: int) -> str:
    """Map numeric credit score to a proxy rating grade."""
    for (low, high), grade in CREDIT_GRADES.items():
        if low <= score <= high:
            return grade
    return "D"


def _extract_key_signals(
    vol: float,
    drawdown: float,
    beta: float,
    ma_signal: str,
    sentiment: str,
) -> List[str]:
    """Generate a concise list of key credit risk signals for the briefing."""
    signals = []

    if vol > 0.35:
        signals.append(f"elevated volatility ({vol*100:.1f}%)")
    else:
        signals.append(f"normal volatility ({vol*100:.1f}%)")

    dd_pct = abs(drawdown) * 100
    if dd_pct > 20:
        signals.append(f"significant drawdown (-{dd_pct:.1f}%)")
    else:
        signals.append(f"moderate drawdown (-{dd_pct:.1f}%)")

    if beta > 1.5:
        signals.append(f"high market sensitivity (β={beta:.2f})")
    elif beta < 0.5 and beta > 0:
        signals.append(f"defensive positioning (β={beta:.2f})")

    if ma_signal in ("death_cross", "below_200ma"):
        signals.append(f"bearish MA signal ({ma_signal.replace('_', ' ')})")
    elif ma_signal == "golden_cross":
        signals.append("bullish MA signal (golden cross)")

    if sentiment != "NEUTRAL":
        signals.append(f"news sentiment: {sentiment.lower()}")

    return signals
