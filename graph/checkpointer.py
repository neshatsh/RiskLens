# graph/checkpointer.py
"""SQLite checkpointer — persists graph state for HITL interrupts and run history."""

from __future__ import annotations

import logging
import os

from langgraph.checkpoint.sqlite import SqliteSaver

from core.config import SQLITE_DB_PATH

logger = logging.getLogger(__name__)


def get_checkpointer() -> SqliteSaver:
    """Open (or create) the SQLite checkpoint DB. Uses sqlite3.connect() directly — from_conn_string() is a context manager in v2.x."""
    import sqlite3
    os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)
    logger.info("Using SQLite checkpointer at %s", SQLITE_DB_PATH)
    conn = sqlite3.connect(SQLITE_DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)
