# core/config.py
"""Central config for RiskLens — all thresholds, API keys, and constants in one place."""

from __future__ import annotations

import os
from typing import Dict

from dotenv import load_dotenv

load_dotenv()


# API keys
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # "openai" | "anthropic"
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
LLM_TEMPERATURE: float = 0.0  # keep outputs deterministic for risk analysis


# Market data
LOOKBACK_DAYS: int = 252          # one full trading year for VaR and vol
BENCHMARK_TICKER: str = "SPY"     # beta benchmark
TRADING_DAYS_PER_YEAR: int = 252  # always 252, never 365
ROLLING_VOL_WINDOW: int = 21      # ~1 trading month


# Market risk thresholds
VOL_HIGH_THRESHOLD: float = 0.35     # annualised vol > 35% → flag
VOL_EXTREME_THRESHOLD: float = 0.55  # > 55% → escalate

DRAWDOWN_ALERT_THRESHOLD: float = 0.30   # -30% from peak → alert
DRAWDOWN_SEVERE_THRESHOLD: float = 0.50  # -50% → severe

VAR_99_BREACH_THRESHOLD: float = 0.025   # 1-day 99% VaR > 2.5% of AUM
VAR_95_WARNING_THRESHOLD: float = 0.015  # 1-day 95% VaR > 1.5% of AUM

HHI_HIGH_CONCENTRATION: float = 0.25  # normalised HHI > 0.25 = too concentrated

BETA_HIGH_THRESHOLD: float = 1.5
BETA_DEFENSIVE_THRESHOLD: float = 0.5


# Credit risk thresholds
CREDIT_SCORE_HIGH_RISK: int = 45   # below 45 = elevated (BB, speculative grade)
CREDIT_SCORE_CRITICAL: int = 30    # below 30 = distressed (CCC/near-default only)

CREDIT_GRADES = {
    (80, 100): "AA",
    (65, 79):  "A",
    (55, 64):  "BBB",   # lower investment grade
    (42, 54):  "BB",    # speculative grade — elevated market risk, not distressed
    (28, 41):  "B",     # high yield, meaningful stress signals
    (0, 27):   "CCC",   # near-distressed only; reserve for confirmed solvency risk
}


# Operational risk — keyword lists by event category
OP_RISK_KEYWORDS: Dict[str, list] = {
    "fraud_misconduct": ["fraud", "misconduct", "fine", "penalty", "SEC", "OSC", "OSFI",
                         "investigation", "bribery", "corruption", "embezzlement"],
    "sanctions":        ["sanction", "OFAC", "SDN list", "export control", "embargo",
                         "designated entity"],
    "cyber":            ["data breach", "cyberattack", "ransomware", "hack", "outage",
                         "IT failure", "system failure", "zero-day"],
    "legal":            ["lawsuit", "class action", "regulatory action", "compliance failure",
                         "SEC charge", "DOJ", "settlement", "indictment"],
    "key_person":       ["CEO resigned", "CFO departure", "leadership change", "fired",
                         "terminated", "board resignation"],
}

# Countries with active OFAC primary sanctions
SANCTIONED_COUNTRIES = [
    "Russia", "Iran", "North Korea", "Cuba", "Syria", "Venezuela",
    "Belarus", "Myanmar", "Zimbabwe",
]

OP_RISK_SEVERITY_KEYWORDS = {
    "CRITICAL": ["sanction", "fraud", "indictment", "SEC charge", "DOJ"],
    "HIGH":     ["fine", "penalty", "data breach", "ransomware", "lawsuit"],
    "MEDIUM":   ["investigation", "cyberattack", "class action", "compliance failure"],
    "LOW":      ["leadership change", "CEO resigned", "CFO departure"],
}


# Macro thresholds (FRED-sourced)
FRED_SERIES: Dict[str, str] = {
    "DFF":          "Federal Funds Rate",
    "GS10":         "10-Year Treasury Yield",
    "GS2":          "2-Year Treasury Yield",
    "T10Y2Y":       "10Y-2Y Yield Spread",
    "CPIAUCSL":     "CPI Inflation YoY",
    "UNRATE":       "US Unemployment Rate",
    "GDPC1":        "Real GDP Growth (QoQ ann.)",
    "BAMLH0A0HYM2": "US HY Credit Spread (bps)",
    "VIXCLS":       "VIX Volatility Index",
}

YIELD_CURVE_INVERSION_THRESHOLD: float = 0.0   # T10Y2Y < 0 → inverted
INFLATION_HIGH_THRESHOLD: float = 4.0
HY_SPREAD_STRESS_THRESHOLD: float = 500.0       # bps
VIX_ELEVATED_THRESHOLD: float = 25.0
VIX_EXTREME_THRESHOLD: float = 35.0


# Aggregation weights — Basel III pillar weighting
RISK_WEIGHTS = {
    "market_risk":      0.40,
    "credit_risk":      0.35,
    "operational_risk": 0.15,
    "macro_risk":       0.10,
}

RISK_LEVEL_THRESHOLDS = {
    "CRITICAL": 75,
    "HIGH":     55,
    "MEDIUM":   35,
    # below 35 → LOW
}


# RAG / vector store
RAG_DOCS_DIR: str = os.path.join(os.path.dirname(__file__), "..", "rag", "docs")
FAISS_INDEX_PATH: str = os.path.join(os.path.dirname(__file__), "..", "rag", "faiss_index")
RAG_TOP_K: int = 3
RAG_CHUNK_SIZE: int = 500
RAG_CHUNK_OVERLAP: int = 50


# Checkpointer and output paths
SQLITE_DB_PATH: str = os.path.join(os.path.dirname(__file__), "..", "data", "risklens.db")
DEFAULT_THREAD_ID: str = "risklens-main"
BRIEFINGS_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data", "briefings")
PDF_LOGO_TEXT: str = "RiskLens"
PDF_CONFIDENTIAL_WATERMARK: str = "CONFIDENTIAL — FOR INTERNAL USE ONLY"


# Logging
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
