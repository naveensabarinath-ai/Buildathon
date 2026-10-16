"""Build and publish the local knowledge graph from scraped scheme records."""

import hashlib
import json
import os
import sqlite3
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from scripts import graph_retriever, scheme_documents

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA = ROOT / "data" / "all_tn_agriculture_schemes.json"
GRAPH = ROOT / "indexes" / "agriculture_knowledge_graph.json"
MANIFEST = ROOT / "indexes" / "graph_manifest.json"
DB = ROOT / "data" / "admin_metrics.db"
MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
EXTRACTION_WORKERS = 8
EXTRACTION_BATCH_SIZE = 32


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        """CREATE TABLE IF NOT EXISTS graph_runs (
        id TEXT PRIMARY KEY, started_at TEXT NOT NULL, status TEXT NOT NULL,
        schemes INTEGER NOT NULL, chunks INTEGER NOT NULL,
        extraction_calls INTEGER NOT NULL, prompt_tokens INTEGER NOT NULL,
        completion_tokens INTEGER NOT NULL, message TEXT)"""
    )
    con.commit()
    return con


@contextmanager
def database() -> Iterator[sqlite3.Connection]:
    con = connect()
    try:
        with con:
            yield con
    finally:
        con.close()


def load_chunks():
    if not DATA.is_file():
        raise FileNotFoundError(
            f"Missing {DATA}; run scripts/scrape_tn_agriculture_schemes.py first."
        )
    data = json.loads(DATA.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Scheme JSON must contain a top-level list.")
    documents = [
        document
        for item in data if isinstance(item, dict)
        for document in scheme_documents.scheme_to_documents(item)
    ]
    chunks = scheme_documents.split_selectively(documents)
    if not chunks:
        raise ValueError("No usable scheme text was found in the JSON.")
    return data, chunks


def source_digest(chunks: list[Any]) -> str:
    payload = [
        (document.page_content, document.metadata)
        for document in chunks
    ]
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, ensure_ascii=False, default=str
        ).encode("utf-8")
    ).hexdigest()


def preview() -> dict[str, Any]:
    data, chunks = load_chunks()
    digest = source_digest(chunks)
    graph_error = None
    graph_digest = None
    entities = relations = 0
    if GRAPH.is_file():
        try:
            graph = graph_retriever.load_graph(GRAPH)
            graph_digest = graph.graph.get("source_digest")
            entities = sum(
                1 for _, attributes in graph.nodes(data=True)
                if attributes.get("kind") == "entity"
            )
            relations = sum(
                1 for _, _, attributes in graph.edges(data=True)
                if attributes.get("relation") == "FACT"
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            graph_error = str(exc)
    current = (
        MANIFEST.is_file()
        and MANIFEST.read_text(encoding="utf-8").strip() == digest
        and graph_digest == digest
    )
    return {
        "schemes": len(data),
        "chunks": len(chunks),
        "entities": entities,
        "relations": relations,
        "graph_present": GRAPH.is_file(),
        "graph_current": current,
        "graph_error": graph_error,
        "unchanged_graph": current,
    }


def build(
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Extract graph facts from chunks and atomically publish the graph artifact."""
    data, chunks = load_chunks()
    digest = source_digest(chunks)
    before = preview()
    if before["unchanged_graph"]:
        return {
            "status": "skipped",
            "message": "Scheme content is unchanged; no graph extraction calls were made.",
            "extraction_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            **before,
        }
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is required for LLM entity/relation extraction.")

    run_id = uuid.uuid4().hex[:12]
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with database() as con:
        con.execute(
            "INSERT INTO graph_runs VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, started_at, "running", len(data), len(chunks), 0, 0, 0, ""),
        )

    llm = ChatOpenAI(model=MODEL, temperature=0, timeout=60, max_retries=2)
    extraction_calls = 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    staged_graph = ROOT / f".graph_stage_{run_id}.json"
    graph_backup = ROOT / f".graph_backup_{run_id}.json"
    manifest_backup = ROOT / f".graph_manifest_backup_{run_id}.json"
    staged_manifest = ROOT / f".graph_manifest_stage_{run_id}.txt"
    published_graph = False
    published_manifest = False
    try:
        extractable_chunks = [
            document
            for document in chunks
            if str(document.metadata.get("scheme_id", "")).strip()
        ]
        futures: dict[int, Future] = {}
        next_document = 0
        next_batch_start = 0

        def submit_next_batch(executor: ThreadPoolExecutor) -> None:
            nonlocal next_batch_start
            batch_end = min(
                next_batch_start + EXTRACTION_BATCH_SIZE, len(extractable_chunks)
            )
            for index in range(next_batch_start, batch_end):
                document = extractable_chunks[index]
                futures[id(document)] = executor.submit(
                    graph_retriever.extract_facts,
                    llm,
                    document.page_content,
                )
            next_batch_start = batch_end

        def facts_loader(document):
            nonlocal extraction_calls, next_document
            facts, item_usage = futures.pop(id(document)).result()
            next_document += 1
            extraction_calls += 1
            usage["prompt_tokens"] += item_usage["prompt_tokens"]
            usage["completion_tokens"] += item_usage["completion_tokens"]
            if (
                next_batch_start < len(extractable_chunks)
                and next_document == next_batch_start
            ):
                submit_next_batch(executor)
            return facts, item_usage

        with ThreadPoolExecutor(max_workers=EXTRACTION_WORKERS) as executor:
            submit_next_batch(executor)
            graph, _ = graph_retriever.build_graph(
                chunks,
                digest,
                facts_loader=facts_loader,
                progress_callback=progress_callback,
            )
        graph_retriever.save_graph(graph, staged_graph)
        staged_manifest.write_text(digest, encoding="utf-8")
        try:
            # Retain the previous valid pair unless both new files publish successfully.
            if GRAPH.exists():
                GRAPH.rename(graph_backup)
            if MANIFEST.exists():
                MANIFEST.rename(manifest_backup)
            os.replace(staged_graph, GRAPH)
            published_graph = True
            os.replace(staged_manifest, MANIFEST)
            published_manifest = True
        except Exception:
            if published_graph:
                GRAPH.unlink()
            if published_manifest:
                MANIFEST.unlink()
            if graph_backup.exists():
                graph_backup.rename(GRAPH)
            if manifest_backup.exists():
                manifest_backup.rename(MANIFEST)
            raise
        if graph_backup.exists():
            graph_backup.unlink()
        if manifest_backup.exists():
            manifest_backup.unlink()
        with database() as con:
            con.execute(
                """UPDATE graph_runs SET status=?, extraction_calls=?,
                prompt_tokens=?, completion_tokens=?, message=? WHERE id=?""",
                (
                    "success", extraction_calls, usage["prompt_tokens"],
                    usage["completion_tokens"],
                    "Knowledge graph published. Restart FastAPI to load it.", run_id,
                ),
            )
        return {
            "status": "success",
            "run_id": run_id,
            "extraction_calls": extraction_calls,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "message": "Knowledge graph published; restart FastAPI to load it.",
            **before,
        }
    except Exception as exc:
        with database() as con:
            con.execute(
                """UPDATE graph_runs SET status=?, extraction_calls=?,
                prompt_tokens=?, completion_tokens=?, message=? WHERE id=?""",
                (
                    "failed", extraction_calls, usage["prompt_tokens"],
                    usage["completion_tokens"], str(exc)[:1000], run_id,
                ),
            )
        raise
    finally:
        for path in (staged_graph, staged_manifest):
            if path.exists():
                path.unlink()


def recent_runs(limit: int = 25) -> list[dict[str, Any]]:
    with database() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            "SELECT * FROM graph_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
