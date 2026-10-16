"""Convert scraped scheme records into LangChain documents and text chunks."""

import json
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    """Create field-aware documents from a scraped scheme record."""
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
        for field, value in listing.items():
            if field in {"Scheme", "S.No"}:
                continue
            field_text = (
                listing_to_text({field: value})
                if field == "PDF Links"
                else clean_value(value)
            )
            if not field_text or field_text.lower() in {"null", "none"}:
                continue
            docs.append(Document(
                page_content=prefix + f"{field}: {field_text}",
                metadata={
                    **base,
                    "content_type": field_types.get(field, "scheme_listing"),
                    "listing_field": field,
                },
            ))
    elif clean_value(listing):
        docs.append(Document(
            page_content=prefix + clean_value(listing),
            metadata={**base, "content_type": "scheme_listing"},
        ))

    for row_number, detail in enumerate(details, start=1):
        if isinstance(detail, dict):
            detail_lines = []
            for key, value in detail.items():
                value_text = clean_value(value)
                if value_text.lower() in {"", "null", "none"}:
                    if key.strip().casefold() == "pattern of subsidy":
                        detail_lines.append(
                            f"{key}: Not specified in source."
                        )
                    continue
                detail_lines.append(f"{key}: {value_text}")
            detail_text = "\n".join(detail_lines)
        else:
            detail_text = clean_value(detail)
        if detail_text:
            docs.append(Document(
                page_content=prefix + f"Scheme detail record {row_number}:\n{detail_text}",
                metadata={
                    **base,
                    "content_type": "scheme_detail",
                    "detail_row": row_number,
                },
            ))
    return docs


def split_selectively(documents: list[Document]) -> list[Document]:
    """Keep eligibility and document lists whole; split other long records."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        if document.metadata.get("content_type") in {"required_documents", "eligibility"}:
            chunks.append(document)
        else:
            chunks.extend(splitter.split_documents([document]))
    return chunks
