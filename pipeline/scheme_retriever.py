"""Phase 2: retrieve scheme evidence from an existing, trusted local FAISS index.

No new indexing and no answer-generation LLM calls occur here.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DEFAULT_INDEX = ROOT / "indexes" / "faiss_tn_agriculture"
EMBEDDING_MODEL = "text-embedding-3-small"


def load_store(index_dir: Path = DEFAULT_INDEX) -> FAISS:
    """Load ONLY your own trusted index (index.pkl is pickle-serialized)."""
    if not (index_dir / "index.faiss").is_file() or not (index_dir / "index.pkl").is_file():
        raise FileNotFoundError(
            f"Missing index.faiss or index.pkl in {index_dir}. "
            "Build the local indexes/faiss_tn_agriculture folder or use --index-dir."
        )
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found. Set it in .env or environment.")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(
        str(index_dir), embeddings, allow_dangerous_deserialization=True
    )


def exact_documents(store: FAISS, *, scheme_id: str | None = None,
                    content_type: str | None = None) -> list[Document]:
    """Fetch every matching indexed chunk without embedding or top-k loss.

    The locally saved LangChain docstore must be trusted, just like index.pkl.
    """
    results = []
    for doc_id in store.index_to_docstore_id.values():
        document = store.docstore.search(doc_id)
        if not isinstance(document, Document):
            continue
        if scheme_id and str(document.metadata.get("scheme_id")) != str(scheme_id):
            continue
        if content_type and document.metadata.get("content_type") != content_type:
            continue
        results.append(document)
    return results


def fetch_documents(
    store: FAISS,
    question: str,
    *,
    k: int = 5,
    mode: str = "mmr",
    scheme_id: str | None = None,
    content_type: str | None = None,
    fetch_k: int = 40,
) -> list[Document]:
    """Retrieve semantic matches or exact metadata matches from trusted index."""
    if mode == "exact":
        if not scheme_id and not content_type:
            raise ValueError("Exact mode requires --scheme-id or --content-type")
        return exact_documents(store, scheme_id=scheme_id, content_type=content_type)
    if not question.strip():
        raise ValueError("Question cannot be blank")
    if k < 1 or fetch_k < k:
        raise ValueError("Require k >= 1 and fetch_k >= k")
    if mode not in {"mmr", "similarity"}:
        raise ValueError("mode must be 'mmr', 'similarity', or 'exact'")
    filters: dict[str, Any] = {}
    if scheme_id:
        filters["scheme_id"] = str(scheme_id)
    if content_type:
        filters["content_type"] = content_type
    kwargs: dict[str, Any] = {"k": k, "fetch_k": fetch_k}
    if filters:
        kwargs["filter"] = filters
    if mode == "mmr":
        kwargs["lambda_mult"] = 0.7
    retriever = store.as_retriever(search_type=mode, search_kwargs=kwargs)
    return retriever.invoke(question)


def search_distances(
    store: FAISS,
    question: str,
    *,
    k: int = 5,
    scheme_id: str | None = None,
    content_type: str | None = None,
    fetch_k: int = 40,
) -> list[tuple[Document, float]]:
    """Diagnostic similarity search: native FAISS L2 distance, lower is nearer.

    Distances are NOT calibrated relevance/confidence percentages.
    """
    if not question.strip():
        raise ValueError("Question cannot be blank")
    if k < 1 or fetch_k < k:
        raise ValueError("Require k >= 1 and fetch_k >= k")
    filters: dict[str, str] = {}
    if scheme_id:
        filters["scheme_id"] = scheme_id
    if content_type:
        filters["content_type"] = content_type
    return store.similarity_search_with_score(
        question, k=k, fetch_k=fetch_k, filter=filters or None
    )


def as_record(document: Document, rank: int, distance: float | None = None) -> dict:
    record: dict[str, Any] = {
        "rank": rank,
        "scheme_name": document.metadata.get("scheme_name", "Unknown"),
        "scheme_id": document.metadata.get("scheme_id", ""),
        "content_type": document.metadata.get("content_type", ""),
        "detail_row": document.metadata.get("detail_row"),
        "source": document.metadata.get("source", ""),
        "page_content": document.page_content,
    }
    if distance is not None:
        record["l2_distance"] = distance
    return record


def run_search(store: FAISS, question: str, args: argparse.Namespace) -> list[dict]:
    if args.mode == "distance":
        hits = search_distances(
            store, question, k=args.k, scheme_id=args.scheme_id,
            content_type=args.content_type, fetch_k=args.fetch_k
        )
        return [as_record(doc, i, distance=float(score)) for i, (doc, score) in enumerate(hits, 1)]
    docs = fetch_documents(
        store, question, k=args.k, mode=args.mode, scheme_id=args.scheme_id,
        content_type=args.content_type, fetch_k=args.fetch_k
    )
    return [as_record(doc, i) for i, doc in enumerate(docs, 1)]


def print_results(question: str, records: list[dict], json_output: bool) -> None:
    if json_output:
        print(json.dumps({"query": question, "results": records}, ensure_ascii=False, indent=2))
        return
    print(f"\nQuery: {question}\nRetrieved {len(records)} chunk(s)")
    if not records:
        print("No results. This does not prove that no government scheme exists.")
    for result in records:
        print(f"\n--- #{result['rank']}: {result['scheme_name']} ---")
        print(f"ID: {result['scheme_id']} | Type: {result['content_type']} | Detail row: {result['detail_row']}")
        if "l2_distance" in result:
            print(f"L2 distance (lower = closer): {result['l2_distance']:.4f}")
        print(result["page_content"])


def main() -> None:
    parser = argparse.ArgumentParser(description="TN Agriculture FAISS retrieval (Phase 2)")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--query", help="One question; without this, enter interactive mode")
    parser.add_argument("--mode", choices=("mmr", "similarity", "distance", "exact"), default="mmr")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fetch-k", type=int, default=40)
    parser.add_argument("--scheme-id", help="Exact scheme_id metadata filter")
    parser.add_argument("--content-type", choices=("scheme_listing", "scheme_description", "eligibility", "required_documents", "scheme_detail", "guidelines"))
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    args = parser.parse_args()

    if args.k < 1 or args.fetch_k < args.k:
        parser.error("--k must be positive and --fetch-k must be >= --k")
    if args.mode == "exact" and not (args.scheme_id or args.content_type):
        parser.error("--mode exact requires --scheme-id or --content-type")
    store = load_store(args.index_dir)
    if not args.json:
        print(f"Loaded FAISS index ({store.index.ntotal} vectors) from {args.index_dir}")
    if args.query is not None:
        try:
            print_results(args.query, run_search(store, args.query, args), args.json)
        except ValueError as exc:
            parser.error(str(exc))
        return

    print("Enter a question or type 'exit'.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        print_results(question, run_search(store, question, args), args.json)


if __name__ == "__main__":
    main()
