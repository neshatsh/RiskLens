# rag/ingest.py
"""Ingests Basel III docs into a FAISS index. Run once: python -m rag.ingest"""

from __future__ import annotations

import logging
import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from core.config import (
    FAISS_INDEX_PATH,
    OPENAI_API_KEY,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_DOCS_DIR,
)

logger = logging.getLogger(__name__)


def load_documents() -> List:
    """Load all .txt files from the rag/docs/ directory."""
    docs = []
    if not os.path.exists(RAG_DOCS_DIR):
        logger.error("RAG docs directory not found: %s", RAG_DOCS_DIR)
        return docs

    for filename in os.listdir(RAG_DOCS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(RAG_DOCS_DIR, filename)
            loader = TextLoader(filepath, encoding="utf-8")
            file_docs = loader.load()
            # Tag each document with its source file for citation
            for doc in file_docs:
                doc.metadata["source"] = filename
            docs.extend(file_docs)
            logger.info("Loaded %d document(s) from %s", len(file_docs), filename)

    logger.info("Total documents loaded: %d", len(docs))
    return docs


def chunk_documents(docs: List) -> List:
    """Split documents into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks (size=%d, overlap=%d)", len(chunks), RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
    return chunks


def build_faiss_index(force_rebuild: bool = False) -> FAISS:
    """Build or load the FAISS index. Saves to disk on first call; loads on subsequent calls."""
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is required for RAG embeddings")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )

    if os.path.exists(FAISS_INDEX_PATH) and not force_rebuild:
        logger.info("Loading existing FAISS index from %s", FAISS_INDEX_PATH)
        try:
            return FAISS.load_local(
                FAISS_INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            logger.warning("Failed to load FAISS index (%s) — rebuilding", exc)

    logger.info("Building FAISS index from regulatory documents...")
    docs = load_documents()
    if not docs:
        raise FileNotFoundError(f"No documents found in {RAG_DOCS_DIR}")

    chunks = chunk_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(os.path.dirname(FAISS_INDEX_PATH) if os.path.dirname(FAISS_INDEX_PATH) else ".", exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    logger.info("FAISS index saved to %s (%d vectors)", FAISS_INDEX_PATH, len(chunks))

    return vectorstore


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_faiss_index(force_rebuild=True)
    print("RAG index built successfully.")
