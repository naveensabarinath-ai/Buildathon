"""SQLite-backed local tracking of chatbot usage. No model calls are made here."""
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / 'chat_usage.db'


def connect():
    db = sqlite3.connect(DB, timeout=15)
    db.execute('PRAGMA busy_timeout=15000')
    db.execute('''CREATE TABLE IF NOT EXISTS chat_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_utc TEXT NOT NULL,
        model TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        completion_tokens INTEGER NOT NULL,
        total_tokens INTEGER NOT NULL,
        query_embedding_tokens_estimate INTEGER NOT NULL,
        estimated_cost_usd REAL NOT NULL,
        scheme_id TEXT,
        source_count INTEGER NOT NULL
    )''')
    db.commit()
    return db


def estimate_embedding_tokens(question):
    """Local encoding estimate; may differ from API metering."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'))
        return len(enc.encode(question))
    except Exception:
        return max(1, (len(question) + 3) // 4)


def estimate_cost(prompt_tokens, completion_tokens, embedding_tokens):
    """User-configured USD-per-million rates; zero when not configured."""
    pin = float(os.getenv('CHAT_INPUT_USD_PER_M', '0'))
    pout = float(os.getenv('CHAT_OUTPUT_USD_PER_M', '0'))
    emb = float(os.getenv('EMBEDDING_USD_PER_M', '0'))
    return round((prompt_tokens * pin + completion_tokens * pout + embedding_tokens * emb) / 1_000_000, 8)


def record(*, model, prompt_tokens, completion_tokens, embedding_estimate, scheme_id, source_count):
    prompt_tokens, completion_tokens = int(prompt_tokens), int(completion_tokens)
    estimated_cost = estimate_cost(prompt_tokens, completion_tokens, embedding_estimate)
    with connect() as db:
        db.execute('''INSERT INTO chat_requests
        (created_utc,model,prompt_tokens,completion_tokens,total_tokens,
        query_embedding_tokens_estimate,estimated_cost_usd,scheme_id,source_count)
        VALUES (?,?,?,?,?,?,?,?,?)''',
        (datetime.now(timezone.utc).isoformat(), str(model), prompt_tokens,
         completion_tokens, prompt_tokens + completion_tokens, int(embedding_estimate),
         estimated_cost, scheme_id, int(source_count)))
    return {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'query_embedding_tokens_estimate': int(embedding_estimate),
            'estimated_cost_usd': estimated_cost, 'model': str(model)}


def totals():
    with connect() as db:
        row = db.execute('''SELECT COUNT(*), COALESCE(SUM(prompt_tokens),0),
           COALESCE(SUM(completion_tokens),0), COALESCE(SUM(total_tokens),0),
           COALESCE(SUM(query_embedding_tokens_estimate),0),
           COALESCE(SUM(estimated_cost_usd),0) FROM chat_requests''').fetchone()
    return dict(zip(('requests','prompt_tokens','completion_tokens','total_tokens',
                     'query_embedding_tokens_estimate','estimated_cost_usd'), row))


def recent(limit=100):
    with connect() as db:
        db.row_factory = sqlite3.Row
        rows = db.execute('SELECT * FROM chat_requests ORDER BY id DESC LIMIT ?', (int(limit),)).fetchall()
    return [dict(row) for row in rows]
