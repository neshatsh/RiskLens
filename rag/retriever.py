# rag/retriever.py
"""Semantic retrieval over the Basel III FAISS index — returns top-K passages for a given query."""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_community.vectorstores import FAISS

from core.config import FAISS_INDEX_PATH, OPENAI_API_KEY, RAG_TOP_K
from rag.ingest import build_faiss_index

logger = logging.getLogger(__name__)

_vectorstore: Optional[FAISS] = None


def _clean_passage(text: str, source: str) -> str:
    """Strip separator lines and metadata noise, prepend a clean source label."""
    import re
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Drop lines that are purely = or - characters (doc formatting artifacts)
        if re.fullmatch(r"[=\-]{3,}", stripped):
            continue
        cleaned.append(line)
    body = "\n".join(cleaned).strip()
    # Use a clean source label without the raw filename extension
    label = source.replace(".txt", "").replace("_", " ").title()
    return f"[{label}] {body}"


def get_vectorstore() -> FAISS:
    """Singleton FAISS load — cached at module scope to avoid re-embedding on every call."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = build_faiss_index(force_rebuild=False)
    return _vectorstore


def retrieve_regulatory_context(
    query: str,
    top_k: int = RAG_TOP_K,
) -> List[str]:
    """Return top-K relevant passages with source metadata prepended."""
    try:
        store = get_vectorstore()
        docs = store.similarity_search(query, k=top_k)
        passages = []
        for doc in docs:
            source = doc.metadata.get("source", "regulatory_docs")
            clean = _clean_passage(doc.page_content, source)
            passages.append(clean)
            logger.debug("Retrieved passage from %s (len=%d)", source, len(doc.page_content))
        return passages
    except Exception as exc:
        logger.error("RAG retrieval failed: %s", exc)
        return [f"[RAG unavailable: {exc}]"]


def build_rag_query_from_flags(
    market_flags: List[str],
    credit_flags: List[str],
    op_flags: List[str],
    macro_flags: List[str],
) -> str:
    """Build a targeted Basel III query from the flags raised by other agents."""
    query_parts = []

    if any("VaR" in f or "volatility" in f.lower() for f in market_flags):
        query_parts.append("Basel III VaR requirements Expected Shortfall capital charges market risk")

    if any("drawdown" in f.lower() for f in market_flags):
        query_parts.append("Basel III trading book drawdown stress testing capital buffers")

    if any("credit" in f.lower() or "score" in f.lower() for f in credit_flags):
        query_parts.append("Basel III credit risk probability of default capital requirements IRB")

    if any("sanctions" in f.lower() or "fraud" in f.lower() or "fine" in f.lower() for f in op_flags):
        query_parts.append("Basel III operational risk regulatory fines legal risk capital")

    if any("inverted" in f.lower() or "yield" in f.lower() for f in macro_flags):
        query_parts.append("yield curve inversion duration risk interest rate risk Basel III")

    if any("inflation" in f.lower() for f in macro_flags):
        query_parts.append("inflation regime real return compression interest rate risk")

    if not query_parts:
        query_parts = ["Basel III risk management thresholds capital requirements monitoring"]

    return " | ".join(query_parts)
