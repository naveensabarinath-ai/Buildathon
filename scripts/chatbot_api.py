"""Phase 4 FastAPI backend. Reuses the Phase 3 RAG implementation."""
import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langsmith import traceable, tracing_context
from scripts import agriculture_rag as rag
from scripts import chat_usage
from scripts import graph_retriever as kg_retrieval

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


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    scheme_id: str | None = Field(default=None, max_length=32)
    k: int = Field(default=5, ge=1, le=20)


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
    """Load the graph, scheme catalog, and chat model."""
    manifest = BASE_DIR / 'indexes' / 'graph_manifest.json'
    if not manifest.is_file():
        raise FileNotFoundError('Graph manifest missing; build the knowledge graph.')
    expected_digest = manifest.read_text(encoding='utf-8').strip()
    graph = kg_retrieval.load_graph(
        BASE_DIR / 'indexes' / 'agriculture_knowledge_graph.json',
        expected_digest=expected_digest,
    )
    catalog = kg_retrieval.scheme_catalog(graph)
    llm = ChatOpenAI(model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4.1-mini'), temperature=0, timeout=60, max_retries=2)
    return graph, catalog, llm


@traceable(name='chatbot_question', run_type='chain')
def traceable_answer(question, scheme_id, k):
    graph, catalog, llm = get_services()
    return rag.answer_question(
        graph, llm, question, catalog, k=k, scheme_id=scheme_id,
    )


@app.get('/health')
def health():
    """Basic health; does not trigger an embedding or LLM API call."""
    try:
        graph, catalog, _ = get_services()
        return {
            'status': 'ok',
            'schemes': len(catalog),
            'graph_nodes': graph.number_of_nodes(),
        }
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
            tags=['chatbot', 'knowledge-graph'],
        ):
            result = traceable_answer(
                payload.question, payload.scheme_id, payload.k
            )
        usage = result.get('usage')
        if usage and usage.get('prompt_tokens') is not None and usage.get('completion_tokens') is not None:
            try:
                tracked = chat_usage.record(
                    model=usage['model'], prompt_tokens=usage['prompt_tokens'],
                    completion_tokens=usage['completion_tokens'],
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
        raise HTTPException(status_code=502, detail='Unable to generate answer. Check API logs, OpenAI access, and the knowledge-graph artifact.')
