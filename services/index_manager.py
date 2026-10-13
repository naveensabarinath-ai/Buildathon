"""Cost-aware index builder; no OpenAI calls when vectors are already cached."""
import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
DATA = ROOT / 'data' / 'all_tn_agriculture_schemes.json'
INDEX = ROOT / 'indexes' / 'faiss_tn_agriculture'
DB = ROOT / 'data' / 'admin_metrics.db'
MANIFEST = ROOT / 'indexes' / 'index_manifest.json'
MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
BATCH_SIZE = 80


def connect():
    con = sqlite3.connect(DB, timeout=30)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('''CREATE TABLE IF NOT EXISTS vectors (
      hash TEXT PRIMARY KEY, model TEXT NOT NULL, embedding TEXT NOT NULL,
      token_estimate INTEGER NOT NULL, created_at TEXT NOT NULL)''')
    con.execute('''CREATE TABLE IF NOT EXISTS runs (
      id TEXT PRIMARY KEY, started_at TEXT NOT NULL, status TEXT NOT NULL,
      schemes INTEGER NOT NULL, chunks INTEGER NOT NULL, new_vectors INTEGER NOT NULL,
      reused_vectors INTEGER NOT NULL, estimated_tokens INTEGER NOT NULL,
      actual_tokens INTEGER, message TEXT)''')
    con.commit()
    return con


def load_chunks():
    if not DATA.exists():
        raise FileNotFoundError(f'Missing {DATA}; run pipeline/scrape_tn_agriculture_schemes.py first.')
    spec = importlib.util.spec_from_file_location(
        'scheme_builder', ROOT / 'pipeline' / 'build_faiss_index.py'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    data = json.loads(DATA.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('JSON must be a list of schemes')
    docs = []
    for item in data:
        if isinstance(item, dict):
            docs.extend(mod.scheme_to_documents(item))
    chunks = mod.split_selectively(docs)
    if not chunks:
        raise ValueError('No usable chunks')
    return data, chunks


def manifest_digest(chunks):
    payload = [(digest(doc.page_content), doc.metadata) for doc in chunks]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode('utf-8')).hexdigest()


def digest(text):
    return hashlib.sha256((MODEL + '\0' + text).encode('utf-8')).hexdigest()


def estimate_tokens(text):
    try:
        import tiktoken
        return len(tiktoken.get_encoding('cl100k_base').encode(text))
    except ImportError:
        raise RuntimeError('Install tiktoken for accurate local token estimates: pip install tiktoken')


def preview():
    data, chunks = load_chunks()
    hashes = [digest(doc.page_content) for doc in chunks]
    distinct = dict((digest(doc.page_content), doc.page_content) for doc in chunks)
    with connect() as con:
        if distinct:
            available = set()
            keys = list(distinct)
            for i in range(0, len(keys), 400):
                page = keys[i:i+400]
                sql = 'SELECT hash FROM vectors WHERE hash IN (' + ','.join('?' for _ in page) + ')'
                available.update(x[0] for x in con.execute(sql, page))
        else:
            available = set()
    missing = {h: t for h, t in distinct.items() if h not in available}
    estimated = sum(estimate_tokens(t) for t in missing.values())
    index_present = all((INDEX / f).exists() for f in ('index.faiss', 'index.pkl'))
    current = (MANIFEST.exists() and MANIFEST.read_text(encoding='utf-8').strip() ==
               manifest_digest(chunks))
    return dict(schemes=len(data), chunks=len(chunks), unique_texts=len(distinct),
                new_vectors=len(missing), reused_vectors=len(distinct)-len(missing),
                estimated_tokens=estimated, index_present=index_present,
                unchanged_index=bool(current and index_present))


def build():
    """Recreate index from cached/new vectors; publish only after a complete successful build."""
    import faiss  # noqa: F401 (ensure installed)
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    from openai import OpenAI

    data, chunks = load_chunks()
    hashes = [digest(doc.page_content) for doc in chunks]
    manifest_hash = manifest_digest(chunks)
    before = preview()
    if before['unchanged_index']:
        return {'status': 'skipped', 'message': 'No changes; no embeddings requested.', **before}

    run_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with connect() as con:
        con.execute('INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)',
                    (run_id, now, 'running', len(data), len(chunks), before['new_vectors'],
                     before['reused_vectors'], before['estimated_tokens'], None, ''))
        con.commit()
    actual = 0
    try:
        unique = {digest(doc.page_content): doc.page_content for doc in chunks}
        with connect() as con:
            rows = con.execute('SELECT hash, embedding FROM vectors WHERE model = ?', (MODEL,)).fetchall()
            cache = {h: json.loads(e) for h, e in rows if h in unique}

        missing = [(h, text) for h, text in unique.items() if h not in cache]
        if missing:
            if not os.getenv('OPENAI_API_KEY'):
                raise RuntimeError('Set OPENAI_API_KEY in .env before embedding new chunks')
            client = OpenAI()
            for i in range(0, len(missing), BATCH_SIZE):
                batch = missing[i:i+BATCH_SIZE]
                # batch output indexed in response, not assumed input order
                response = client.embeddings.create(model=MODEL, input=[t for _, t in batch])
                actual += int(response.usage.total_tokens) if response.usage else 0
                embedded = {entry.index: entry.embedding for entry in response.data}
                if len(embedded) != len(batch):
                    raise RuntimeError('Incomplete embeddings response')
                with connect() as con:
                    for n, (h, content) in enumerate(batch):
                        vector = embedded[n]
                        cache[h] = vector
                        con.execute('INSERT OR REPLACE INTO vectors VALUES (?,?,?,?,?)',
                                    (h, MODEL, json.dumps(vector, separators=(',', ':')),
                                     estimate_tokens(content), now))
                    con.commit()

        # Rebuild FAISS index locally from all saved vectors (including reused ones).
        # This produces no additional embedding API calls.
        embeddings = OpenAIEmbeddings(model=MODEL)
        pairs = [(doc.page_content, cache[digest(doc.page_content)]) for doc in chunks]
        store = FAISS.from_embeddings(pairs, embeddings, metadatas=[d.metadata for d in chunks])
        staged = ROOT / f'.faiss_stage_{run_id}'
        staged.mkdir(exist_ok=False)
        store.save_local(str(staged))
        if not (staged / 'index.faiss').is_file() or not (staged / 'index.pkl').is_file():
            raise RuntimeError('Incomplete staged index')
        # Windows-friendly replace: close API first; rename old to backup, stage to active.
        backup = ROOT / f'.faiss_backup_{run_id}'
        if INDEX.exists():
            INDEX.rename(backup)
        try:
            staged.rename(INDEX)
        except Exception:
            if backup.exists() and not INDEX.exists():
                backup.rename(INDEX)
            raise
        MANIFEST.write_text(manifest_hash, encoding='utf-8')
        if backup.exists():
            shutil.rmtree(backup)
        with connect() as con:
            con.execute('UPDATE runs SET status=?, actual_tokens=?, message=? WHERE id=?',
                        ('success', actual, 'Index published. Restart FastAPI to load it.', run_id))
            con.commit()
        return {'status': 'success', 'run_id': run_id, 'actual_tokens': actual,
                'message': 'Index published; restart FastAPI to load it.', **before}
    except Exception as exc:
        with connect() as con:
            con.execute('UPDATE runs SET status=?, actual_tokens=?, message=? WHERE id=?',
                        ('failed', actual, str(exc)[:1000], run_id))
            con.commit()
        raise


def recent_runs(limit=25):
    with connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute('SELECT * FROM runs ORDER BY started_at DESC LIMIT ?', (limit,)).fetchall()
        return [dict(row) for row in rows]
