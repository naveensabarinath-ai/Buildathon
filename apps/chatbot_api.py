"""Phase 4 FastAPI backend. Reuses the Phase 3 RAG implementation."""
import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langsmith import traceable, tracing_context
from services import chat_usage

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / '.env')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('tn_agriculture_api')
LANGSMITH_TRACING = os.getenv('LANGSMITH_TRACING', '').strip().casefold() in {
    'true', '1', 'yes', 'on',
}
LANGSMITH_PROJECT = os.getenv('LANGSMITH_PROJECT', 'tn-agriculture-chatbot')
if LANGSMITH_TRACING and not os.getenv('LANGSMITH_API_KEY'):
    raise RuntimeError('LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not set.')


def load_phase3():
    spec = importlib.util.spec_from_file_location(
        'tn_agri_phase3', BASE_DIR / 'pipeline' / 'agriculture_rag.py'
    )
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot import Phase 3 script')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rag = load_phase3()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    scheme_id: str | None = Field(default=None, max_length=32)
    k: int = Field(default=5, ge=1, le=20)
    mode: Literal['mmr', 'similarity'] = 'mmr'


class SourceItem(BaseModel):
    label: str
    scheme_id: str
    scheme_name: str
    content_type: str
    detail_row: int | None = None
    excerpt: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    retrieval: dict
    usage: dict | None = None


app = FastAPI(title='TN Agriculture Schemes RAG API', version='1.0.0')
# Local development only; use explicit allowed origins if deploying on a network.
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:8501', 'http://127.0.0.1:8501'], allow_methods=['GET', 'POST'], allow_headers=['*'])


@lru_cache(maxsize=1)
def get_services():
    """Load the FAISS store, scheme catalog, and model once per API process."""
    store = rag.retrieval.load_store(BASE_DIR / 'indexes' / 'faiss_tn_agriculture')
    catalog = rag.scheme_catalog(store)
    llm = ChatOpenAI(model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4.1-mini'), temperature=0, timeout=60, max_retries=2)
    return store, catalog, llm


@traceable(name='chatbot_question', run_type='chain')
def traceable_answer(question, scheme_id, k, mode):
    store, catalog, llm = get_services()
    return rag.answer_question(
        store, llm, question, catalog, k=k, mode=mode, scheme_id=scheme_id
    )


@app.get('/health')
def health():
    """Basic health; does not trigger an embedding or LLM API call."""
    try:
        store, catalog, _ = get_services()
        return {'status': 'ok', 'vectors': int(store.index.ntotal), 'schemes': len(catalog)}
    except Exception:
        logger.exception('Health check failed')
        raise HTTPException(status_code=503, detail='Service not ready. Check API logs and local configuration.')


@app.get('/schemes')
def schemes():
    try:
        _, catalog, _ = get_services()
        return [{'scheme_id': sid, 'scheme_name': name} for sid, name in sorted(catalog.items(), key=lambda x: x[1].casefold())]
    except Exception:
        logger.exception('Could not list schemes')
        raise HTTPException(status_code=503, detail='Scheme catalog unavailable.')


@app.post('/ask', response_model=AskResponse)
def ask(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=422, detail='Question cannot be blank.')
    try:
        _, catalog, _ = get_services()
        if payload.scheme_id and payload.scheme_id not in catalog:
            raise HTTPException(status_code=404, detail='Scheme ID not found in the current index.')
        with tracing_context(
            enabled=LANGSMITH_TRACING,
            project_name=LANGSMITH_PROJECT,
            tags=['chatbot', payload.mode],
        ):
            result = traceable_answer(
                payload.question, payload.scheme_id, payload.k, payload.mode
            )
        usage = result.get('usage')
        if usage and usage.get('prompt_tokens') is not None and usage.get('completion_tokens') is not None:
            try:
                tracked = chat_usage.record(
                    model=usage['model'], prompt_tokens=usage['prompt_tokens'],
                    completion_tokens=usage['completion_tokens'],
                    embedding_estimate=chat_usage.estimate_embedding_tokens(payload.question),
                    scheme_id=result.get('retrieval', {}).get('scheme_id'),
                    source_count=len(result.get('sources', [])))
                result['usage'] = tracked
            except Exception:
                logger.exception('Usage persistence failed (answer still delivered)')
        return result
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception('Question processing failed')
        raise HTTPException(status_code=502, detail='Unable to generate answer. Check API logs, OpenAI access, and FAISS files.')
