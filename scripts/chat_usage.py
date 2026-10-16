"""SQLite-backed local tracking of chat and graph-extraction token usage."""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DB = Path(__file__).resolve().parents[1] / "chat_usage.db"


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(DB, timeout=15)
    db.execute("PRAGMA busy_timeout=15000")
    existing = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chat_requests'"
    ).fetchone()
    if existing:
        columns = {
            row[1] for row in db.execute("PRAGMA table_info(chat_requests)").fetchall()
        }
        required_columns = {
            "id", "created_utc", "model", "prompt_tokens", "completion_tokens",
            "total_tokens", "estimated_cost_usd", "scheme_id", "source_count",
        }
        if columns != required_columns:
            legacy = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='chat_requests_legacy'"
            ).fetchone()
            if not legacy:
                db.execute("ALTER TABLE chat_requests RENAME TO chat_requests_legacy")
    db.execute(
        """CREATE TABLE IF NOT EXISTS chat_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_utc TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        completion_tokens INTEGER NOT NULL,
        total_tokens INTEGER NOT NULL,
        estimated_cost_usd REAL NOT NULL,
        scheme_id TEXT,
        source_count INTEGER NOT NULL
        )"""
    )
    legacy = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chat_requests_legacy'"
    ).fetchone()
    migrated = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chat_usage_migration'"
    ).fetchone()
    if legacy and not migrated:
        db.execute(
            """INSERT INTO chat_requests
            (created_utc, model, prompt_tokens, completion_tokens, total_tokens,
             estimated_cost_usd, scheme_id, source_count)
            SELECT created_utc, model, prompt_tokens, completion_tokens, total_tokens,
                   estimated_cost_usd, scheme_id, source_count
            FROM chat_requests_legacy"""
        )
        db.execute("CREATE TABLE chat_usage_migration (version INTEGER PRIMARY KEY)")
        db.execute("INSERT INTO chat_usage_migration VALUES (1)")
    db.commit()
    return db


@contextmanager
def database() -> Iterator[sqlite3.Connection]:
    db = connect()
    try:
        with db:
            yield db
    finally:
        db.close()


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Use optional configured chat-model prices; zero when not configured."""
    prompt_rate = float(os.getenv("CHAT_INPUT_USD_PER_M", "0"))
    completion_rate = float(os.getenv("CHAT_OUTPUT_USD_PER_M", "0"))
    return round(
        (prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1_000_000,
        8,
    )


def record(*, model, prompt_tokens, completion_tokens, scheme_id, source_count):
    prompt_tokens, completion_tokens = int(prompt_tokens), int(completion_tokens)
    estimated_cost = estimate_cost(prompt_tokens, completion_tokens)
    with database() as db:
        db.execute(
            """INSERT INTO chat_requests
            (created_utc, model, prompt_tokens, completion_tokens, total_tokens,
             estimated_cost_usd, scheme_id, source_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                str(model),
                prompt_tokens,
                completion_tokens,
                prompt_tokens + completion_tokens,
                estimated_cost,
                scheme_id,
                int(source_count),
            ),
        )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": estimated_cost,
        "model": str(model),
    }


def totals():
    with database() as db:
        row = db.execute(
            """SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0),
               COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(total_tokens), 0),
               COALESCE(SUM(estimated_cost_usd), 0) FROM chat_requests"""
        ).fetchone()
    return dict(zip(
        ("requests", "prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_usd"),
        row,
    ))


def recent(limit=100):
    with database() as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT * FROM chat_requests ORDER BY id DESC LIMIT ?", (int(limit),)
        ).fetchall()
    return [dict(row) for row in rows]
