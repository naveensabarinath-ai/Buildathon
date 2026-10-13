"""
Build a local FAISS vector store from the scraped Tamil Nadu Agriculture scheme JSON.

Expected input:
    data/all_tn_agriculture_schemes.json

Output folder:
    indexes/faiss_tn_agriculture/

This indexes the text in the JSON listing and scheme-detail tables.
PDF URLs are retained as metadata/text references, but the PDF contents are NOT
downloaded or embedded by this script.
"""

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_JSON = BASE_DIR / "data" / "all_tn_agriculture_schemes.json"
VECTORSTORE_DIR = BASE_DIR / "indexes" / "faiss_tn_agriculture"

load_dotenv(BASE_DIR / ".env")

# OpenAI embedding model: good starting point for a small RAG project.
EMBEDDING_MODEL = "text-embedding-3-small"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def clean_value(value: Any) -> str:
    """Convert values to readable text without losing nested JSON content."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return " ".join(str(value).split())


def listing_to_text(listing: dict) -> str:
    """Convert the main scheme listing fields into text."""
    lines = []
    for key, value in listing.items():
        if key == "PDF Links":
            if value:
                lines.append("Government Orders and Guidelines / PDF links:")
                for item in value:
                    if isinstance(item, dict):
                        label = clean_value(item.get("link_text", "PDF"))
                        url = clean_value(item.get("url", ""))
                        lines.append(f"- {label}: {url}")
                    else:
                        lines.append(f"- {clean_value(item)}")
            continue

        value_text = clean_value(value)
        if value_text:
            lines.append(f"{key}: {value_text}")
    return "\n".join(lines)


def scheme_to_documents(item: dict) -> list[Document]:
    """Index each listing field individually so important values stay together."""
    scheme_name = clean_value(item.get("scheme_name", "Unknown scheme"))
    scheme_id = clean_value(item.get("scheme_id", ""))
    listing = item.get("listing") or {}
    details = item.get("details") or []
    base = {
        "scheme_name": scheme_name,
        "scheme_id": scheme_id,
        "source": "tnagriculture_scheme_portal",
        "detail_status": clean_value(item.get("detail_status", "")),
    }
    docs = []
    prefix = f"Scheme name: {scheme_name}\nScheme ID: {scheme_id}\n"
    field_types = {
        "Scheme Description": "scheme_description",
        "Eligibility Criteria": "eligibility",
        "Document's Required": "required_documents",
        "Documents Required": "required_documents",
        "GO & Guidelines": "guidelines",
    }
    if isinstance(listing, dict):
        # Keep full document checklists in one chunk, even if > CHUNK_SIZE.
        for field, value in listing.items():
            if field in {"Scheme", "S.No"}:
                continue
            if field == "PDF Links":
                field_text = listing_to_text({field: value})
            else:
                field_text = clean_value(value)
            if not field_text or field_text.lower() in {"null", "none"}:
                continue
            content_type = field_types.get(field, "scheme_listing")
            docs.append(Document(
                page_content=prefix + f"{field}: {field_text}",
                metadata={**base, "content_type": content_type, "listing_field": field},
            ))
    elif clean_value(listing):
        docs.append(Document(page_content=prefix + clean_value(listing),
                             metadata={**base, "content_type": "scheme_listing"}))

    for row_number, detail in enumerate(details, start=1):
        if isinstance(detail, dict):
            detail_text = "\n".join(
                f"{key}: {clean_value(value)}" for key, value in detail.items()
                if clean_value(value).lower() not in {"", "null", "none"}
            )
        else:
            detail_text = clean_value(detail)
        if detail_text:
            docs.append(Document(
                page_content=prefix + f"Scheme detail record {row_number}:\n{detail_text}",
                metadata={**base, "content_type": "scheme_detail", "detail_row": row_number},
            ))
    return docs


def split_selectively(documents: list[Document]) -> list[Document]:
    """Don't split application checklists; chunk other long records as needed."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in documents:
        if doc.metadata.get("content_type") in {"required_documents", "eligibility"}:
            chunks.append(doc)
        else:
            chunks.extend(splitter.split_documents([doc]))
    return chunks


def main():
    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_JSON.resolve()}\n"
            "Expected the scheme JSON at data/all_tn_agriculture_schemes.json."
        )

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. In PowerShell, run:\n"
            '$env:OPENAI_API_KEY = "your-openai-api-key"'
        )

    with INPUT_JSON.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            "Expected the scraper JSON to contain a top-level list of schemes."
        )

    documents = []
    for item in data:
        if isinstance(item, dict):
            documents.extend(scheme_to_documents(item))

    if not documents:
        raise ValueError(
            "No usable text was found in the JSON. Check that the scraper output "
            "contains listing fields or detail rows."
        )

    print(f"Schemes in JSON: {len(data)}")
    print(f"Documents before splitting: {len(documents)}")

    chunks = split_selectively(documents)

    print(f"Text chunks to embed: {len(chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("Creating embeddings and building FAISS index...")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print("\nVector store created successfully.")
    print(f"Saved to: {VECTORSTORE_DIR.resolve()}")
    print("Files normally include index.faiss and index.pkl.")

    # Quick retrieval smoke test.
    test_query = "Which agriculture schemes provide subsidy for farm equipment?"
    print(f"\nTest query: {test_query}")
    results = vectorstore.similarity_search(test_query, k=3)

    for number, doc in enumerate(results, start=1):
        print(f"\n--- Result {number} ---")
        print("Scheme:", doc.metadata.get("scheme_name", "Unknown"))
        print("Content type:", doc.metadata.get("content_type", "Unknown"))
        print(doc.page_content[:700])


if __name__ == "__main__":
    main()
