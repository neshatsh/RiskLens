# RiskLens: Agentic AI Risk Intelligence for Banks

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-orange.svg)

[//]: # (![License]&#40;https://img.shields.io/badge/license-MIT-lightgrey.svg&#41;)

**RiskLens** is a production-grade agentic AI system built on LangGraph and LangChain that monitors a financial portfolio across all three Basel III risk pillars, runs specialist agents in parallel, and generates a structured Risk Intelligence Briefing with regulatory citations and a human-in-the-loop review checkpoint.

[//]: # (---)

[//]: # ()
[//]: # (![RiskLens Demo]&#40;demo.gif&#41;)

[//]: # ()
[//]: # (---)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RiskLens Pipeline                        │
│                                                                 │
│  Portfolio Input                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────┐         Conditional routing via LLM               │
│  │Supervisor│ ──────────────────────────────────────────────┐   │
│  └──────────┘                                               │   │
│       │                                                     │   │
│  ┌────┴──────────────────────────────────────────┐          │   │
│  │            Parallel Execution (Send API)      │          │   │
│  │                                               │          │   │
│  │  ┌─────────────┐  ┌─────────────┐             │          │   │
│  │  │ Market Risk │  │ Credit Risk │             │          │   │
│  │  │   Agent     │  │   Agent     │             │          │   │
│  │  └─────────────┘  └─────────────┘             │          │   │
│  │  ┌─────────────┐  ┌─────────────┐             │          │   │
│  │  │ Operational │  │    Macro    │             │          │   │
│  │  │ Risk Agent  │  │   Agent     │             │          │   │
│  │  └─────────────┘  └─────────────┘             │          │   │
│  └────────────────────────┬──────────────────────┘          │   │
│                           │                                 │   │
│                           ▼                                 │   │
│                    ┌─────────────┐                          │   │
│                    │  RAG Agent  │ ← Basel III citations    │   │
│                    └─────────────┘                          │   │
│                           │                                 │   │
│                           ▼                                 │   │
│                    ┌─────────────┐                          │   │
│                    │ Aggregator  │ ← Composite risk score   │   │
│                    └─────────────┘                          │   │
│                           │                                 │   │
│              ┌────────────┴──────────────┐                  │   │
│         HIGH/CRITICAL?             LOW/MEDIUM?              │   │
│              ▼                           ▼                  │   │
│        ┌───────────┐               ┌──────────┐             │   │
│        │   HITL    │               │  Report  │◄────────────┘   │
│        │interrupt()│               │  Agent   │                 │
│        └───────────┘               └──────────┘                 │
│              │                         │                        │
│      analyst approves                  │                        │
│              └─────────────────────────▼                        │
│                                  PDF + JSON Briefing            │
└─────────────────────────────────────────────────────────────────┘
```


## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent Orchestration | LangGraph 0.2+ | StateGraph, Send API, interrupt() |
| LLM Backbone | GPT-4o / Claude Sonnet | Supervisor routing, report generation |
| Market Data | yfinance | Price history, returns, volatility |
| Macro Data | FRED API | Rates, inflation, yield curve, VIX |
| News Intelligence | NewsAPI | Credit and operational risk scanning |
| RAG | FAISS + OpenAI Embeddings | Basel III regulatory retrieval |
| Checkpointing | SQLite (LangGraph MemorySaver) | HITL state persistence across sessions |
| Dashboard | Streamlit + Plotly | Interactive risk monitoring UI |
| PDF Reports | ReportLab | Bank-style risk briefing output |
| Structured Outputs | Pydantic v2 | Type-safe LLM responses |


## Key Features

- **True Multi-Agent Orchestration:** Supervisor uses LLM-powered conditional routing at runtime, not a fixed sequential chain. Agents activate only when relevant to the portfolio.
- **Parallel Execution:** Market risk, credit risk, operational risk, and macro agents run concurrently via LangGraph's `Send` API.
- **All Three Basel III Pillars:** Market risk (VaR/CVaR, volatility, beta, drawdown), credit risk (proxy scoring, news sentiment), operational risk (sanctions, fraud, cyber event detection).
- **FRTB-Compliant Metrics:** Expected Shortfall (CVaR) at 95% is the primary output metric, matching the Basel IV Fundamental Review of the Trading Book requirement.
- **Real HITL with State Persistence:** Graph pauses via `interrupt()` for HIGH/CRITICAL risk. SQLite checkpointer preserves full state across the interrupt. Analyst notes are embedded in the final briefing and marked in the PDF.
- **Regulatory RAG:** FAISS vector index over Basel III/IV documents provides verbatim regulatory citations for every risk flag raised. The system knows *why* a risk matters, not just that it exists.
- **Production-Ready Error Handling:** Every external API call has fallback behaviour. One agent failing never crashes the pipeline.
- **Downloadable PDF Briefings:** ReportLab-generated bank-style reports with cover page, risk heatmap, position details, recommended actions, and regulatory citations.


## Methodology

### Value at Risk (VaR)

VaR is computed using **historical simulation** on log returns over the prior 252 trading days, with no distributional assumption:

```
Historical VaR (95%) = -percentile(daily_log_returns, 5%)
```

Both **historical** and **parametric** (Gaussian) VaR are computed for comparison. The divergence between the two reveals non-normality in the return distribution.

**Expected Shortfall (CVaR)** is the Basel III / FRTB primary metric:
```
CVaR (95%) = -mean(returns where return ≤ -VaR_95)
```

CVaR captures the average severity of losses *beyond* the VaR threshold, giving a more complete picture of tail risk. It is also sub-additive (diversification always helps), unlike VaR.

**Annualisation** uses 252 trading days (industry convention, not 365):
```
Annualised Volatility = Daily_Std × √252
```

### Credit Risk Scoring

Portfolio monitoring uses **proxy credit scoring** from market data, mirroring the logic in CDS spread-implied ratings and equity-derived PD models:

- High volatility + large drawdown + high beta → elevated credit stress
- News sentiment scan: headlines classified as CRITICAL/HIGH/MEDIUM/LOW
- Moving average signal: `below_200ma` flags deteriorating trend
- Composite score (0-100) maps to proxy grade (AAA to CCC)

### Supervisor Routing

The supervisor makes an LLM call with the portfolio summary and returns structured JSON specifying which agents to activate and why, creating an audit trail of routing decisions. This makes it trivial to add new specialist agents without changing any routing logic.

### Aggregation Weights

```python
composite_score = (
    market_risk_score  × 0.40 +  # FRTB emphasis on market risk capital
    credit_risk_score  × 0.35 +  # Credit dominates bank RWA (~75-80%)
    operational_score  × 0.15 +  # Often underweighted in practice
    macro_risk_score   × 0.10    # Systemic context
)
```

Weights approximate the typical RWA distribution in a Canadian bank's investment book.


## Design Notes

- **LangGraph over chains:** `SequentialChain` runs everything in order with no branching. LangGraph's `Send` API lets the four specialist agents run in parallel, the supervisor routes conditionally at runtime, and `interrupt()` pauses the graph mid-execution with full SQLite-persisted state — none of that is possible with a simple chain.
- **HITL is a compliance requirement, not a UX choice:** OSFI and SR 11-7 require human accountability in automated risk systems. The checkpoint exists so an analyst signs off before the briefing is released, with their notes embedded in the output as an audit trail.
- **RAG for citations:** A briefing that flags "high market risk" is less useful than one that quotes the specific Basel III article driving the capital requirement. The FAISS index over Basel III/IV documents pulls verbatim passages relevant to each flag raised.


## How to Run

### 1. Clone and set up

```bash
git clone https://github.com/neshatsh/RiskLens.git
cd RiskLens
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env:
#   OPENAI_API_KEY=sk-...      (required)
#   FRED_API_KEY=...           (free at fred.stlouisfed.org, enables macro tab)
#   NEWS_API_KEY=...           (free at newsapi.org, enables operational risk)
```

### 3. Build the RAG index (one-time)

```bash
python -m rag.ingest
```

Embeds Basel III reference documents into a local FAISS index. Takes ~10 seconds.

### 4. Run via CLI

```bash
python main.py                                    # sample portfolio
python main.py --pdf                              # with PDF output
python main.py --portfolio data/my_portfolio.json # custom portfolio
python main.py --no-hitl                          # skip HITL for testing
```

### 5. Launch dashboard

```bash
streamlit run app/dashboard.py
# Open http://localhost:8501
```

### 6. Run tests

```bash
pytest tests/ -v
```


## Project Structure

```
risklens/
├── agents/
│   ├── supervisor.py            # LLM-powered conditional routing
│   ├── market_risk_agent.py     # VaR, CVaR, volatility, beta, drawdown
│   ├── credit_risk_agent.py     # Proxy scoring, news sentiment
│   ├── operational_risk_agent.py# Sanctions, fraud, cyber detection
│   ├── macro_agent.py           # FRED indicators, yield curve
│   ├── rag_agent.py             # Basel III regulatory retrieval
│   └── report_agent.py          # Structured briefing generation
├── core/
│   ├── state.py                 # LangGraph shared TypedDict state
│   └── config.py                # Thresholds, constants, settings
├── graph/
│   ├── builder.py               # StateGraph construction
│   └── checkpointer.py          # SQLite MemorySaver setup
├── hitl/
│   └── review.py                # interrupt() handler
├── tools/
│   ├── market_data.py           # yfinance wrapper
│   ├── fred_data.py             # FRED API client
│   ├── news_scanner.py          # NewsAPI scanner
│   └── risk_calculators.py      # VaR, CVaR, Sortino, HHI
├── rag/
│   ├── ingest.py                # Document embedding pipeline
│   ├── retriever.py             # Semantic search
│   └── docs/                    # Basel III/IV source documents
├── output/
│   ├── pdf_generator.py         # ReportLab PDF briefing
│   └── formatter.py             # JSON briefing formatter
├── app/
│   └── dashboard.py             # Streamlit dashboard
├── notebooks/
│   └── 01_risk_methodology.ipynb# VaR walkthrough, LangGraph intro
└── tests/
```

[//]: # ()
[//]: # (## Known Limitations & Future Work)

[//]: # ()
[//]: # (**Current limitations:**)

[//]: # (- Credit scoring is a proxy model &#40;market-implied&#41;, not a true PD/LGD model)

[//]: # (- News coverage gaps without a premium API subscription)

[//]: # (- FRED macro data has release lags &#40;some series update monthly/quarterly&#41;)

[//]: # (- Mixed CAD/USD portfolio without FX risk modelling)

[//]: # (- RAG uses cosine similarity only; a cross-encoder reranking step would improve precision)

[//]: # ()
[//]: # (**Planned:**)

[//]: # (- Stress testing module: 2008 crisis, COVID March 2020, 2022 rate shock scenarios)

[//]: # (- Full Merton model for distance-to-default)

[//]: # (- Liquidity risk module &#40;bid-ask spread, market impact&#41;)

[//]: # (- OSFI-specific capital ratio monitoring for Canadian bank holdings)

[//]: # (- Automated Slack/email alerts for CRITICAL events)

[//]: # (- Bloomberg/Refinitiv integration for institutional-grade data)


[//]: # (## Background)

[//]: # ()
[//]: # (RiskLens is the second project in a two-part portfolio. The first, **[DocuLens]&#40;https://github.com/neshatsh/Doculens-RAG-platform&#41;**, built a production RAG pipeline from scratch &#40;without frameworks&#41; for legal and financial documents: BERT reranking, VLM extraction, FastAPI, Docker, 78% test coverage. RiskLens introduces agentic orchestration with LangGraph and LangChain, applying the same document intelligence principles to live portfolio risk monitoring.)

[//]: # ()
[//]: # (Together they demonstrate both levels: building AI systems from first principles, and orchestrating them at production scale with modern frameworks.)


[//]: # (## Author)

[//]: # ()
[//]: # (**Neshat Sharbatdar**)

[//]: # (Toronto, ON · [LinkedIn]&#40;https://www.linkedin.com/in/neshat-sharbatdar/&#41; · [GitHub]&#40;https://github.com/neshatsh&#41; · [DocuLens]&#40;https://github.com/neshatsh/Doculens-RAG-platform&#41;)
