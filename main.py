# main.py
"""RiskLens CLI — runs the full multi-agent pipeline. See --help for options."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

# Load .env before any other imports that read config
load_dotenv()

from core.config import LOG_FORMAT, LOG_LEVEL
from core.portfolio import load_portfolio
from graph.builder import build_graph, get_initial_state
from hitl.review import format_hitl_summary
from output.formatter import briefing_to_summary_text


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format=LOG_FORMAT,
    )
    # Quiet noisy third-party loggers
    for lib in ["httpx", "httpcore", "openai", "anthropic", "yfinance", "urllib3"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RiskLens — AI-powered portfolio risk analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--portfolio", type=str, default=None, help="Path to portfolio JSON file")
    parser.add_argument("--date", type=str, default=datetime.today().strftime("%Y-%m-%d"), help="Analysis date (YYYY-MM-DD)")
    parser.add_argument("--no-hitl", action="store_true", help="Skip HITL review step")
    parser.add_argument("--pdf", action="store_true", help="Generate PDF briefing after analysis")
    parser.add_argument("--thread-id", type=str, default=None, help="LangGraph thread ID for resuming a run")
    return parser.parse_args()


def run_analysis(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    logger.info("RiskLens starting — analysis date: %s", args.date)

    # Load portfolio
    portfolio = load_portfolio(args.portfolio)
    logger.info("Portfolio loaded: %d positions", len(portfolio))

    # Build graph
    use_checkpointer = not args.no_hitl
    graph = build_graph(use_checkpointer=use_checkpointer)

    # Initial state
    initial_state = get_initial_state(portfolio, args.date)
    thread_id = args.thread_id or f"risklens-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'='*60}")
    print(f"  RiskLens — AI Risk Intelligence Platform")
    print(f"  Analysis Date: {args.date}")
    print(f"  Portfolio: {len(portfolio)} positions")
    print(f"  Thread ID: {thread_id}")
    print(f"{'='*60}\n")

    print("Running multi-agent risk analysis...")
    print("(Specialist agents run in parallel: market | credit | operational | macro)\n")

    # First graph invocation — runs until HITL interrupt or completion
    try:
        result = graph.invoke(initial_state, config)
    except Exception as exc:
        logger.error("Graph invocation failed: %s", exc, exc_info=True)
        print(f"\nError: {exc}")
        sys.exit(1)

    # Check if graph paused at HITL interrupt
    if result.get("hitl_triggered") and result.get("final_briefing") is None and not args.no_hitl:
        print("\n" + format_hitl_summary(result))

        # Interactive HITL
        try:
            analyst_notes = input("\nEnter analyst notes (or press Enter to skip): ").strip()
            approve_str = input("Approve briefing? [Y/n]: ").strip().lower()
            analyst_approved = approve_str not in ("n", "no")
        except (KeyboardInterrupt, EOFError):
            print("\nAnalysis cancelled by user.")
            sys.exit(0)

        # Resume graph with analyst input
        from langgraph.types import Command
        print("\nResuming analysis with analyst input...")
        try:
            result = graph.invoke(
                Command(resume={"analyst_notes": analyst_notes or None, "analyst_approved": analyst_approved}),
                config,
            )
        except Exception as exc:
            logger.error("Graph resume failed: %s", exc, exc_info=True)
            print(f"\nResume failed: {exc}")
            sys.exit(1)

    # Extract and display final briefing
    briefing = result.get("final_briefing")
    if not briefing:
        print("\nWarning: No final briefing generated.")
        sys.exit(1)

    print(briefing_to_summary_text(briefing))

    # Optional PDF generation
    if args.pdf:
        print("\nGenerating PDF report...")
        try:
            from output.pdf_generator import generate_pdf
            pdf_path = generate_pdf(briefing)
            if pdf_path:
                print(f"PDF saved to: {pdf_path}")
            else:
                print("PDF generation failed — install reportlab: pip install reportlab")
        except Exception as exc:
            logger.warning("PDF generation failed: %s", exc)
            print(f"PDF error: {exc}")

    print(f"\nAnalysis complete. Thread ID: {thread_id}")
    print("View past briefings in: data/briefings/")
    print("Launch dashboard: streamlit run app/dashboard.py\n")


def main() -> None:
    setup_logging()
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
