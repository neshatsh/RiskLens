# app/dashboard.py
"""
RiskLens Streamlit Dashboard — interactive risk intelligence UI.

Dark-themed, professional dashboard with four tabs covering market risk,
credit/operational risk, and macro/regulatory risk. Includes a HITL review
panel that appears when risk is HIGH or CRITICAL, and a PDF download button.

Run: streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import streamlit as st

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# page config must come before any other st call
st.set_page_config(
    page_title="RiskLens — AI Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp { background-color: #0d0d18; color: #e8e8f0; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px; padding: 16px; margin: 6px 0;
        border-left: 4px solid #3355bb;
    }
    .risk-badge-critical { background: #dc2626; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
    .risk-badge-high { background: #f97316; color: white; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
    .risk-badge-medium { background: #eab308; color: #111; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
    .risk-badge-low { background: #22c55e; color: #111; padding: 4px 12px; border-radius: 20px; font-weight: bold; }
    .flag-box { background: #1f0808; border-left: 3px solid #dc2626; padding: 8px 12px; margin: 4px 0; border-radius: 4px; font-size: 13px; }
    .citation-box { background: #0a1628; border-left: 3px solid #3355bb; padding: 10px 14px; margin: 8px 0; border-radius: 4px; font-size: 12px; font-family: monospace; }
    .stButton > button { background: #3355bb; color: white; border: none; border-radius: 8px; padding: 8px 20px; }
    .stButton > button:hover { background: #4466cc; }
</style>
""", unsafe_allow_html=True)


# lazy imports so the dashboard loads even without API keys configured

@st.cache_resource
def _load_graph():
    from graph.builder import build_graph
    return build_graph(use_checkpointer=True)


@st.cache_data(ttl=60)
def _load_portfolio(json_path: Optional[str] = None):
    from core.portfolio import load_portfolio
    return load_portfolio(json_path)


def _risk_badge(level: str) -> str:
    css_class = f"risk-badge-{level.lower()}" if level else "risk-badge-low"
    return f'<span class="{css_class}">{level}</span>'


def _score_color(score: float) -> str:
    if score >= 75: return "#dc2626"
    if score >= 55: return "#f97316"
    if score >= 35: return "#eab308"
    return "#22c55e"


# sidebar

with st.sidebar:
    st.title("🔍 RiskLens")
    st.caption("AI-powered risk intelligence for institutional portfolios")
    st.divider()

    st.subheader("Portfolio")
    portfolio_option = st.radio("Select portfolio", ["Sample Portfolio", "Upload JSON"])
    uploaded_portfolio = None
    if portfolio_option == "Upload JSON":
        uploaded_file = st.file_uploader("Upload portfolio JSON", type=["json"])
        if uploaded_file:
            uploaded_portfolio = json.loads(uploaded_file.read())

    analysis_date = st.date_input("Analysis date", value=datetime.today()).strftime("%Y-%m-%d")

    st.subheader("Model")
    llm_choice = st.selectbox("LLM Backend", ["OpenAI GPT-4o", "Anthropic Claude Sonnet"])
    if llm_choice == "Anthropic Claude Sonnet":
        os.environ["LLM_PROVIDER"] = "anthropic"
    else:
        os.environ["LLM_PROVIDER"] = "openai"

    st.divider()
    run_button = st.button("▶  Run Full Analysis", use_container_width=True, type="primary")
    st.divider()
    st.caption("v1.0 — RiskLens Portfolio")


# session state defaults
if "briefing" not in st.session_state:
    st.session_state.briefing = None
if "running" not in st.session_state:
    st.session_state.running = False
if "hitl_pending" not in st.session_state:
    st.session_state.hitl_pending = False
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"risklens-{datetime.now().strftime('%Y%m%d%H%M%S')}"


# run analysis on button click
if run_button:
    from core.portfolio import load_portfolio, SAMPLE_PORTFOLIO
    from graph.builder import build_graph, get_initial_state

    portfolio = uploaded_portfolio or SAMPLE_PORTFOLIO
    initial_state = get_initial_state(portfolio, analysis_date)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    st.session_state.running = True
    st.session_state.briefing = None
    st.session_state.hitl_pending = False

    with st.spinner("Running multi-agent risk analysis... (this may take 30–60 seconds)"):
        try:
            graph = _load_graph()
            result = graph.invoke(initial_state, config)

            # Check if graph paused at HITL interrupt
            if result.get("hitl_triggered") and result.get("final_briefing") is None:
                st.session_state.hitl_pending = True
                st.session_state.preliminary_state = result
                st.warning("**HITL Review Required** — Risk level is HIGH or CRITICAL. Please review below.")
            else:
                st.session_state.briefing = result.get("final_briefing")
                st.success("Analysis complete!")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            st.session_state.running = False


# HITL review panel — shown when risk is HIGH/CRITICAL
if st.session_state.get("hitl_pending"):
    prelim = st.session_state.get("preliminary_state", {})
    st.markdown("---")
    st.markdown("## ⚠️ Analyst Review Required")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Score", f"{prelim.get('overall_risk_score', 0):.1f}/100")
    with col2:
        st.metric("Market Risk", f"{prelim.get('market_risk_score', 0):.1f}/100")
    with col3:
        st.metric("Credit Risk", f"{prelim.get('credit_risk_score', 0):.1f}/100")
    with col4:
        st.metric("Macro Risk", f"{prelim.get('macro_risk_score', 0):.1f}/100")

    all_flags = (
        prelim.get("market_risk_flags", [])[:3]
        + prelim.get("credit_risk_flags", [])[:2]
        + prelim.get("operational_risk_flags", [])[:2]
    )
    for flag in all_flags:
        st.markdown(f'<div class="flag-box">⚠️ {flag}</div>', unsafe_allow_html=True)

    with st.form("hitl_form"):
        analyst_notes = st.text_area("Analyst Notes", placeholder="Enter your risk assessment notes...", height=120)
        analyst_approved = st.checkbox("Approve briefing for final release", value=True)
        submitted = st.form_submit_button("Resume Analysis")

        if submitted:
            from graph.builder import build_graph
            from langgraph.types import Command

            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            try:
                graph = _load_graph()
                result = graph.invoke(
                    Command(resume={"analyst_notes": analyst_notes, "analyst_approved": analyst_approved}),
                    config,
                )
                st.session_state.briefing = result.get("final_briefing")
                st.session_state.hitl_pending = False
                st.success("Analysis resumed and finalised!")
                st.rerun()
            except Exception as exc:
                st.error(f"Resume failed: {exc}")


# main content
briefing = st.session_state.briefing

if briefing is None:
    st.title("RiskLens — AI Risk Intelligence Platform")
    st.markdown("""
    **Welcome to RiskLens** — an autonomous, multi-agent risk monitoring system built on LangGraph.

    This system monitors a portfolio of financial positions and generates a structured Risk Intelligence
    Briefing covering all three Basel III risk pillars: market risk, credit risk, and operational risk.

    ### How it works:
    1. **Supervisor Agent** decides which specialist agents to activate based on the portfolio
    2. **Specialist Agents** run in parallel: Market Risk, Credit Risk, Operational Risk, Macro
    3. **RAG Agent** retrieves relevant Basel III/IV regulatory context for the detected risks
    4. **HITL Checkpoint** pauses for analyst review if risk is HIGH or CRITICAL
    5. **Report Agent** generates the final structured briefing with PDF download

    ### To get started:
    - Configure API keys in `.env` (see `.env.example`)
    - Select a portfolio in the sidebar
    - Click **Run Full Analysis**
    """)
    st.stop()


# tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Market Risk", "🏦 Credit & Operational", "🌍 Macro & Regulatory"])

overall = briefing.get("overall_risk", {})
breakdown = briefing.get("risk_breakdown", {})

# tab 1: overview
with tab1:
    st.markdown(f"## {briefing.get('metadata', {}).get('date', '')} Risk Briefing")

    col_score, col_trend, col_pillar = st.columns([2, 1, 3])

    with col_score:
        score = overall.get("score", 0)
        level = overall.get("level", "UNKNOWN")
        color = _score_color(score)
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {color};">
            <div style="font-size: 13px; color: #888;">Overall Risk Score</div>
            <div style="font-size: 48px; font-weight: bold; color: {color};">{score:.1f}</div>
            <div style="font-size: 12px; color: #888;">out of 100</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(_risk_badge(level), unsafe_allow_html=True)

    with col_trend:
        trend = overall.get("trend", "STABLE")
        trend_icon = {"DETERIORATING": "📈⬆️", "IMPROVING": "📉⬇️", "STABLE": "➡️"}.get(trend, "➡️")
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 13px; color: #888;">Trend</div>
            <div style="font-size: 24px;">{trend_icon}</div>
            <div style="font-size: 14px; font-weight: bold;">{trend}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_pillar:
        st.markdown("**Pillar Breakdown**")
        pillar_cols = st.columns(4)
        pillar_icons = {"market_risk": "📈", "credit_risk": "🏦", "operational_risk": "⚙️", "macro_risk": "🌍"}
        for i, (pillar, pdata) in enumerate(breakdown.items()):
            with pillar_cols[i % 4]:
                pscore = pdata.get("score", 0)
                plevel = pdata.get("level", "")
                pcolor = _score_color(pscore)
                icon = pillar_icons.get(pillar, "📊")
                st.markdown(f"""
                <div class="metric-card" style="border-left-color: {pcolor}; text-align: center;">
                    <div style="font-size: 20px;">{icon}</div>
                    <div style="font-size: 11px; color: #888;">{pillar.replace('_', ' ').title()}</div>
                    <div style="font-size: 22px; color: {pcolor}; font-weight: bold;">{pscore:.0f}</div>
                    <div style="font-size: 10px;">{plevel}</div>
                </div>
                """, unsafe_allow_html=True)

    # Executive summary
    st.divider()
    st.markdown("### Executive Summary")
    st.markdown(f"> {briefing.get('executive_summary', '')}")

    # Top risk alerts
    st.markdown("### Top Risk Alerts")
    top_risks = briefing.get("top_risks", [])
    for risk in top_risks[:5]:
        sev = risk.get("severity", "MEDIUM")
        icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(sev, "🟡")
        st.markdown(f'<div class="flag-box">{icon} <b>[{sev}]</b> {risk.get("risk", "")}</div>', unsafe_allow_html=True)

    # Download PDF
    st.divider()
    if st.button("📄 Generate & Download PDF Report"):
        with st.spinner("Generating PDF..."):
            try:
                from output.pdf_generator import generate_pdf
                pdf_path = generate_pdf(briefing)
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="⬇️ Download PDF",
                            data=f.read(),
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf",
                        )
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")


# tab 2: market risk
with tab2:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown("## Market Risk Analysis")

    # VaR table
    var_summary = briefing.get("var_summary", {})
    st.markdown("### Value at Risk & CVaR Summary")
    var_data = {
        "Method": ["Historical 95%", "Historical 99%", "Parametric 95%", "Parametric 99%", "CVaR 95% (Basel III)"],
        "VaR (% of AUM)": [
            f"{var_summary.get('var_95_historical', 0)*100:.3f}%" if var_summary.get('var_95_historical') else "N/A",
            f"{var_summary.get('var_99_historical', 0)*100:.3f}%" if var_summary.get('var_99_historical') else "N/A",
            f"{var_summary.get('var_95_parametric', 0)*100:.3f}%" if var_summary.get('var_95_parametric') else "N/A",
            f"{var_summary.get('var_99_parametric', 0)*100:.3f}%" if var_summary.get('var_99_parametric') else "N/A",
            f"{var_summary.get('cvar_95', 0)*100:.3f}%" if var_summary.get('cvar_95') else "N/A",
        ],
        "Threshold": ["1.5%", "2.5% ⚠️", "—", "—", "—"],
    }
    st.dataframe(pd.DataFrame(var_data), use_container_width=True, hide_index=True)

    # Portfolio metrics
    col_a, col_b = st.columns(2)
    with col_a:
        port_vol = var_summary.get("portfolio_volatility")
        st.metric("Portfolio Volatility (Ann.)", f"{port_vol*100:.2f}%" if port_vol else "N/A")
    with col_b:
        hhi = var_summary.get("hhi_concentration")
        st.metric("Concentration (HHI)", f"{hhi:.3f}" if hhi else "N/A", help="0=diversified, 1=concentrated. >0.25 is high.")

    # Position details table
    st.markdown("### Position Details")
    positions = briefing.get("position_details", [])
    if positions:
        pos_df = pd.DataFrame(positions)
        display_cols = ["ticker", "name", "weight", "annualised_vol", "max_drawdown", "beta", "sortino", "ma_signal", "credit_grade"]
        pos_df_display = pos_df[[c for c in display_cols if c in pos_df.columns]].copy()
        for pct_col in ["weight", "annualised_vol", "max_drawdown"]:
            if pct_col in pos_df_display.columns:
                pos_df_display[pct_col] = pos_df_display[pct_col].apply(
                    lambda x: f"{x*100:.1f}%" if x is not None else "N/A"
                )
        st.dataframe(pos_df_display, use_container_width=True, hide_index=True)

        # Volatility chart
        vol_data = {p["ticker"]: (p.get("annualised_vol") or 0) * 100 for p in positions if p.get("annualised_vol")}
        if vol_data:
            vol_df = pd.DataFrame({"Ticker": list(vol_data.keys()), "Volatility (%)": list(vol_data.values())})
            vol_df["Color"] = vol_df["Volatility (%)"].apply(
                lambda x: "HIGH" if x > 35 else ("MEDIUM" if x > 20 else "LOW")
            )
            fig_vol = px.bar(
                vol_df, x="Ticker", y="Volatility (%)", color="Color",
                color_discrete_map={"HIGH": "#dc2626", "MEDIUM": "#eab308", "LOW": "#22c55e"},
                title="Annualised Volatility by Position",
                template="plotly_dark",
            )
            fig_vol.add_hline(y=35, line_dash="dash", line_color="orange", annotation_text="35% threshold")
            st.plotly_chart(fig_vol, use_container_width=True)

    # Market risk flags
    market_flags = breakdown.get("market_risk", {}).get("flags", [])
    if market_flags:
        st.markdown("### Market Risk Flags")
        for flag in market_flags:
            st.markdown(f'<div class="flag-box">⚠️ {flag}</div>', unsafe_allow_html=True)


# tab 3: credit and operational
with tab3:
    import pandas as pd

    st.markdown("## Credit & Operational Risk")

    # Credit scores table
    credit_scores = briefing.get("position_details", [])
    if credit_scores:
        st.markdown("### Credit Quality by Position")
        credit_df = pd.DataFrame(
            [{"Ticker": p["ticker"], "Name": p.get("name", ""), "Credit Score": p.get("credit_score"), "Grade": p.get("credit_grade", "N/A")}
             for p in credit_scores if p.get("credit_score") is not None]
        )
        if not credit_df.empty:
            st.dataframe(credit_df, use_container_width=True, hide_index=True)

    # Credit flags
    credit_flags = breakdown.get("credit_risk", {}).get("flags", [])
    if credit_flags:
        st.markdown("### Credit Risk Flags")
        for flag in credit_flags:
            st.markdown(f'<div class="flag-box">🏦 {flag}</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown("### Operational Risk Events")

    op_events = {}
    for pos in briefing.get("position_details", []):
        ticker = pos["ticker"]
        event_count = pos.get("op_risk_events", 0)
        if event_count:
            op_events[ticker] = event_count

    if op_events:
        import plotly.express as px
        op_df = pd.DataFrame({"Ticker": list(op_events.keys()), "Events": list(op_events.values())})
        fig_op = px.bar(op_df, x="Ticker", y="Events", title="Operational Risk Events by Position", template="plotly_dark", color_discrete_sequence=["#f97316"])
        st.plotly_chart(fig_op, use_container_width=True)

    op_flags = breakdown.get("operational_risk", {}).get("flags", [])
    if op_flags:
        for flag in op_flags:
            if "CRITICAL" in flag:
                sev_icon = "🔴"
            elif "HIGH" in flag:
                sev_icon = "🟠"
            elif "LOW" in flag:
                sev_icon = "🟢"
            else:
                sev_icon = "🟡"
            st.markdown(f'<div class="flag-box">{sev_icon} {flag}</div>', unsafe_allow_html=True)
    else:
        st.success("No significant operational risk events detected in the monitoring period.")


# tab 4: macro and regulatory
with tab4:
    import pandas as pd

    st.markdown("## Macro Environment & Regulatory Context")

    macro_summary = briefing.get("macro_summary", {})
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        curve = macro_summary.get("yield_curve_status", "unknown")
        color = "#dc2626" if curve == "inverted" else ("#eab308" if curve == "flat" else "#22c55e")
        st.markdown(f'<div class="metric-card" style="border-left-color:{color}"><div style="font-size:12px;color:#888">Yield Curve</div><div style="font-size:22px;font-weight:bold;color:{color}">{curve.upper()}</div></div>', unsafe_allow_html=True)
    with col_m2:
        rec = macro_summary.get("recession_probability", "unknown")
        color = "#dc2626" if rec == "HIGH" else ("#eab308" if rec == "ELEVATED" else "#22c55e")
        st.markdown(f'<div class="metric-card" style="border-left-color:{color}"><div style="font-size:12px;color:#888">Recession Signal</div><div style="font-size:22px;font-weight:bold;color:{color}">{rec}</div></div>', unsafe_allow_html=True)
    with col_m3:
        macro_score = breakdown.get("macro_risk", {}).get("score", 0)
        macro_level = breakdown.get("macro_risk", {}).get("level", "")
        color = _score_color(macro_score)
        st.markdown(f'<div class="metric-card" style="border-left-color:{color}"><div style="font-size:12px;color:#888">Macro Risk Score</div><div style="font-size:22px;font-weight:bold;color:{color}">{macro_score:.0f}/100</div><div style="font-size:11px">{macro_level}</div></div>', unsafe_allow_html=True)

    # FRED indicators table
    st.markdown("### Key Macro Indicators (FRED)")
    macro_indicators = macro_summary.get("key_indicators", [])
    if macro_indicators:
        st.dataframe(pd.DataFrame(macro_indicators), use_container_width=True, hide_index=True)

    # Macro flags
    macro_flags = breakdown.get("macro_risk", {}).get("flags", [])
    if macro_flags:
        st.markdown("### Macro Risk Flags")
        for flag in macro_flags:
            st.markdown(f'<div class="flag-box">🌍 {flag}</div>', unsafe_allow_html=True)

    # Regulatory citations
    st.divider()
    st.markdown("### Basel III / IV Regulatory Context")
    citations = briefing.get("regulatory_citations", [])
    if citations:
        for i, citation in enumerate(citations, 1):
            st.markdown(f"**Citation {i}**")
            st.markdown(f'<div class="citation-box">{citation[:600]}</div>', unsafe_allow_html=True)
    else:
        st.info("Regulatory citations unavailable — build the RAG index with `python -m rag.ingest`")

    # Analyst notes (if present)
    if briefing.get("analyst_notes"):
        st.divider()
        st.markdown("### Analyst Notes")
        st.info(briefing["analyst_notes"])
