# tools/news_scanner.py
"""News scanner — fetches headlines via NewsAPI (falls back to GNews) and classifies risk events."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

from core.config import (
    NEWS_API_KEY,
    OP_RISK_KEYWORDS,
    OP_RISK_SEVERITY_KEYWORDS,
    SANCTIONED_COUNTRIES,
)

logger = logging.getLogger(__name__)

# Number of articles to retrieve per company
MAX_ARTICLES = 10


def _fetch_newsapi(query: str, api_key: str, days_back: int = 7) -> List[str]:
    """Fetch headlines from NewsAPI for a given query string."""
    from_date = (datetime.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "relevancy",
        "pageSize": MAX_ARTICLES,
        "language": "en",
        "apiKey": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        articles = resp.json().get("articles", [])
        return [a.get("title", "") + " — " + (a.get("description") or "") for a in articles]
    except requests.RequestException as exc:
        logger.warning("NewsAPI request failed for '%s': %s", query, exc)
        return []


def _fetch_gnews(query: str, days_back: int = 7) -> List[str]:
    """GNews fallback — free, no key needed, rate-limited."""
    url = "https://gnews.io/api/v4/search"
    params = {
        "q": query,
        "lang": "en",
        "max": MAX_ARTICLES,
        "token": "free",  # GNews allows limited free queries without a key
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            articles = resp.json().get("articles", [])
            return [a.get("title", "") + " — " + (a.get("description") or "") for a in articles]
    except requests.RequestException as exc:
        logger.warning("GNews request failed for '%s': %s", query, exc)
    return []


def fetch_company_news(
    company_name: str,
    ticker: str,
    days_back: int = 7,
) -> List[str]:
    """Fetch recent headlines via NewsAPI (or GNews fallback). Returns list of headline strings."""
    query = f'"{company_name}" OR "{ticker}"'

    if NEWS_API_KEY:
        headlines = _fetch_newsapi(query, NEWS_API_KEY, days_back)
    else:
        logger.debug("NEWS_API_KEY not set — using GNews fallback for %s", ticker)
        headlines = _fetch_gnews(query, days_back)

    headlines = [h for h in headlines if h.strip() and len(h) > 10]
    logger.info("Fetched %d headlines for %s", len(headlines), ticker)
    return headlines


def classify_headline_severity(headline: str) -> str:
    """Keyword-based severity: CRITICAL / HIGH / MEDIUM / LOW / NEUTRAL. Checks worst-first."""
    h_lower = headline.lower()
    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        for kw in OP_RISK_SEVERITY_KEYWORDS[severity]:
            if kw.lower() in h_lower:
                return severity
    return "NEUTRAL"


def classify_headline_category(headline: str) -> str:
    """Map a headline to an operational risk category (or 'other')."""
    h_lower = headline.lower()
    for category, keywords in OP_RISK_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in h_lower:
                return category
    return "other"


def scan_for_op_risk_events(
    headlines: List[str],
    ticker: str,
) -> List[Dict[str, str]]:
    """Filter headlines to non-neutral severity events with category classification."""
    events = []
    for headline in headlines:
        severity = classify_headline_severity(headline)
        if severity != "NEUTRAL":
            events.append({
                "headline": headline[:300],  # Cap length
                "severity": severity,
                "category": classify_headline_category(headline),
                "ticker": ticker,
            })
    return events


def check_sanctions_exposure(
    company_name: str,
    headlines: List[str],
    is_etf: bool = False,
) -> List[str]:
    """Flag sanctions exposure. ETF/index-level exposure is flagged as indirect (LOW severity)."""
    flags = []
    combined_text = company_name + " " + " ".join(headlines)
    combined_lower = combined_text.lower()

    for country in SANCTIONED_COUNTRIES:
        if country.lower() in combined_lower:
            if is_etf:
                flags.append(
                    f"INDIRECT SANCTIONS EXPOSURE (LOW): {company_name} — "
                    f"index constituent may have minor {country} exposure"
                )
            else:
                flags.append(
                    f"SANCTIONS EXPOSURE: {company_name} — potential direct exposure to {country}"
                )

    return flags


def scan_credit_sentiment(headlines: List[str]) -> str:
    """Overall credit sentiment: POSITIVE / NEUTRAL / NEGATIVE / SEVERE."""
    if not headlines:
        return "NEUTRAL"

    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NEUTRAL": 0}
    for h in headlines:
        sev = classify_headline_severity(h)
        severity_counts[sev] += 1

    if severity_counts["CRITICAL"] > 0:
        return "SEVERE"
    if severity_counts["HIGH"] >= 2:
        return "NEGATIVE"
    if severity_counts["MEDIUM"] >= 3:
        return "NEGATIVE"
    if severity_counts["LOW"] >= 1 or severity_counts["MEDIUM"] >= 1:
        return "NEUTRAL"
    return "POSITIVE"
