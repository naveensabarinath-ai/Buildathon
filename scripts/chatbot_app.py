"""Phase 4 Streamlit frontend. Calls the local FastAPI backend."""
import base64
import os
from pathlib import Path

import httpx
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
API_URL = os.getenv('RAG_API_URL', 'http://127.0.0.1:8000').rstrip('/')
TIMEOUT = 120.0
AGRICULTURE_ART = base64.b64encode(
    (ROOT / 'assets' / 'agriculture_hero.svg').read_bytes()
).decode('ascii')

st.set_page_config(
    page_title='TamilNadu Agriculture : Assistant',
    page_icon='🌾',
    layout='wide',
)
st.markdown(
    """
    <style>
    .stApp {background: #f5f7f2;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1440px;}
    [data-testid="stSidebar"] {
        background: #edf3e9;
        border-right: 1px solid #dfe8d9;
    }
    [data-testid="stChatMessage"] {
        border: 1px solid #e5eadf;
        border-radius: 16px;
        padding: .9rem 1.1rem;
        background: #fff;
        box-shadow: 0 3px 12px rgba(34, 68, 42, .04);
    }
    [data-testid="stChatInput"] {
        background: #fff;
        border: 2px solid #6d9b62;
        border-radius: 18px;
        box-shadow: 0 5px 18px rgba(36, 85, 48, .14);
        transition: border-color .18s ease, box-shadow .18s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #28633b;
        box-shadow: 0 0 0 4px rgba(57, 115, 75, .18), 0 7px 22px rgba(36, 85, 48, .18);
    }
    [data-testid="stChatInput"] textarea {
        background: #fff;
        font-size: 1rem;
        min-height: 3.5rem;
        padding-top: .85rem;
        padding-bottom: .85rem;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #687a66;
        opacity: 1;
    }
    [data-testid="stMetric"] {
        background: #fff;
        border: 1px solid #e5eadf;
        padding: 12px 14px;
        border-radius: 14px;
    }
    .assistant-hero {
        min-height: 132px; padding: 18px 30px; box-sizing: border-box; border-radius: 20px;
        display: flex; flex-direction: column; justify-content: center;
        background-position: center 57%; background-size: cover;
        box-shadow: 0 12px 32px rgba(23, 61, 42, .16);
    }
    .assistant-hero h1 {font-size: 1.7rem; margin: 0; color: #fff;}
    .assistant-hero p {margin: 5px 0 0; color: #e2efd9; font-size: .92rem;}
    .sidebar-heading {color: #174b30; font-size: 1rem; font-weight: 750; margin: .3rem 0 .7rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="assistant-hero" style="background-image:linear-gradient(90deg,rgba(15,49,32,.84),rgba(15,49,32,.18)),url(data:image/svg+xml;base64,{AGRICULTURE_ART})">
        <h1>🌾 TamilNadu Agriculture : Assistant</h1>
        <p>Ask about Tamil Nadu agricultural schemes</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def api_get(path):
    try:
        with httpx.Client(timeout=15) as client:
            response = client.get(f'{API_URL}{path}')
            response.raise_for_status()
            return response.json()
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError):
        return None


status = api_get('/health')
if not status:
    st.error('Assistant service is offline. Start the local API, then refresh this page.')
    st.code('python -m uvicorn scripts.chatbot_api:app --host 127.0.0.1 --port 8000')
    st.stop()

st.sidebar.markdown('<div class="sidebar-heading">🌿 Assistant settings</div>', unsafe_allow_html=True)
st.sidebar.success(
    f"Connected · {status['schemes']} schemes · {status['graph_nodes']:,} graph nodes"
)
schemes = api_get('/schemes') or []
choices = {'Automatic detection': None}
for item in schemes:
    choices[f"{item['scheme_name']} (ID {item['scheme_id']})"] = item['scheme_id']
with st.sidebar:
    selected = st.selectbox('🎯 Scheme filter', list(choices), help='Choose a scheme or let the assistant detect one from your question.')
    top_k = st.slider('📚 Graph records to retrieve', 1, 15, 5)
    show_excerpts = st.checkbox('Show source excerpts', value=False)
    st.divider()
    if st.button('🗑️ Clear conversation', use_container_width=True):
        st.session_state['messages'] = []
        st.rerun()
    st.caption('Answers use entities, relationships, and source text in the local knowledge graph.')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('usage'):
            u = message['usage']
            st.caption(f"LLM tokens: {u.get('total_tokens', 0):,} (input {u.get('prompt_tokens', 0):,} / output {u.get('completion_tokens', 0):,}) | Est. USD ${u.get('estimated_cost_usd', 0):.6f}")
        if message.get('sources'):
            with st.expander(f"Sources ({len(message['sources'])})"):
                for source in message['sources']:
                    st.markdown(f"**[{source['label']}] {source['scheme_name']}** — ID {source['scheme_id']} · {source['content_type']}")
                    if show_excerpts:
                        st.text(source['excerpt'])

question = st.chat_input('✍️ Type your question here… e.g. Documents required for TNIAMP 2023-24?')
if question:
    st.session_state['messages'].append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        with st.spinner('Searching indexed schemes and generating an answer...'):
            try:
                with httpx.Client(timeout=TIMEOUT) as client:
                    response = client.post(f'{API_URL}/ask', json={'question': question, 'scheme_id': choices[selected], 'k': top_k})
                    response.raise_for_status()
                    result = response.json()
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text[:400]
                st.error(f'API request failed ({exc.response.status_code}): {detail}')
                result = None
            except (httpx.RequestError, ValueError) as exc:
                st.error(f'Cannot reach or parse API response: {exc}')
                result = None
        if result is not None:
            st.markdown(result['answer'])
            st.caption(f"Retrieval: {result['retrieval'].get('strategy', 'unknown')} · Scheme ID: {result['retrieval'].get('scheme_id') or 'not fixed'}")
            if result.get('sources'):
                with st.expander(f"Sources ({len(result['sources'])})"):
                    for source in result['sources']:
                        st.markdown(f"**[{source['label']}] {source['scheme_name']}** — ID {source['scheme_id']} · {source['content_type']}")
                        if show_excerpts:
                            st.text(source['excerpt'])
            if result.get('usage'):
                u = result['usage']
                st.caption(f"LLM tokens: {u['total_tokens']:,} (input {u['prompt_tokens']:,} / output {u['completion_tokens']:,}) · Est. cost: ${u['estimated_cost_usd']:.6f}")
            st.session_state['messages'].append({'role': 'assistant', 'content': result['answer'], 'sources': result.get('sources', []), 'usage': result.get('usage')})
