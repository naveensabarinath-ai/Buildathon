"""Local-only maintenance UI. Set ADMIN_PASSWORD in .env before launching."""
import base64
import hmac
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from services import index_manager as manager

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
st.set_page_config(page_title='TamilNadu Agriculture : Admin Dashboard', page_icon='🌾', layout='wide')
AGRICULTURE_ART = base64.b64encode(
    (ROOT / 'assets' / 'agriculture_hero.svg').read_bytes()
).decode('ascii')


def read_scheme_data():
    """Return usable scheme records and an actionable validation message."""
    if not manager.DATA.exists():
        return None, f'Scheme JSON not found: {manager.DATA.name}'
    try:
        data = json.loads(manager.DATA.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f'Cannot read scheme JSON: {exc}'
    if not isinstance(data, list):
        return None, 'Scheme JSON must contain a list.'
    records = [item for item in data if isinstance(item, dict)]
    if not records:
        return None, 'Scheme JSON does not contain any scheme records.'
    return records, None


@st.cache_data(ttl=60, show_spinner=False)
def indexed_scheme_ids(index_path):
    """Inspect trusted local FAISS metadata without making model API calls."""
    index_dir = Path(index_path)
    if not (index_dir / 'index.faiss').is_file() or not (index_dir / 'index.pkl').is_file():
        return []
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY is required to inspect the local FAISS index.')
    spec = importlib.util.spec_from_file_location(
        'admin_scheme_retrieval', ROOT / 'pipeline' / 'scheme_retriever.py'
    )
    if spec is None or spec.loader is None:
        raise RuntimeError('Could not load the FAISS retrieval module.')
    retrieval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(retrieval)
    store = retrieval.load_store(index_dir)
    return sorted({
        str(document.metadata.get('scheme_id', '')).strip()
        for document in retrieval.exact_documents(store)
        if str(document.metadata.get('scheme_id', '')).strip()
    })


st.markdown(
    """
    <style>
    .stApp {background: #f5f7f2;}
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1440px;}
    [data-testid="stSidebar"] {
        display: block;
        background: #edf3e9;
        border-right: 1px solid #dfe8d9;
    }
    [data-testid="stMetric"] {
        background: #fff;
        border: 1px solid #e5eadf;
        padding: 14px 16px;
        border-radius: 14px;
        box-shadow: 0 3px 12px rgba(34, 68, 42, .04);
    }
    [data-testid="stMetricLabel"] p {font-size: .82rem; color: #60705f;}
    [data-testid="stMetricValue"] {font-size: 1.55rem; color: #173d2a;}
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: .45rem;
        padding: .45rem;
        background: #e9efe5;
        border: 1px solid #dce6d8;
        border-radius: 14px;
        overflow-x: auto;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        min-height: 42px;
        padding: .55rem 1rem;
        border: 1px solid transparent;
        border-radius: 10px;
        color: #48604a;
        font-size: .92rem;
        font-weight: 650;
        white-space: nowrap;
        transition: background-color .18s ease, color .18s ease, box-shadow .18s ease;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        background: #f7faf5;
        color: #173d2a;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: #fff;
        border-color: #d9e5d5;
        color: #1b5e35;
        box-shadow: 0 2px 7px rgba(34, 68, 42, .09);
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {display: none;}
    div[data-testid="stTabs"] [data-testid="stTabContent"] {padding-top: 1.25rem;}
    .step-heading {
        display: flex; align-items: center; gap: .75rem;
        margin: 0 0 1rem; color: #174b30; font-size: 1.25rem; font-weight: 700;
    }
    .step-number {
        display: inline-flex; align-items: center; justify-content: center;
        width: 2rem; height: 2rem; border-radius: 10px;
        background: #e1efdf; color: #27683d; font-size: .9rem; font-weight: 800;
    }
    .hero {
        min-height: 132px; padding: 18px 30px; box-sizing: border-box; border-radius: 20px;
        display: flex; flex-direction: column; justify-content: center;
        color: white; background-position: center 57%; background-size: cover;
        box-shadow: 0 12px 32px rgba(23, 61, 42, .16);
    }
    .hero h1 {font-size: 1.7rem; margin: 0; color: white;}
    .hero p {margin: 5px 0 0; color: #e2efd9; font-size: .92rem;}
    div[data-testid="stForm"] label {color: #173d2a !important; font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)

password = os.getenv('ADMIN_PASSWORD', '')
if not password:
    st.error('Set ADMIN_PASSWORD in .env and restart this dashboard.')
    st.stop()

if not st.session_state.get('authenticated', False):
    st.markdown(
        f"""
        <div class="hero" style="background-image:linear-gradient(90deg,rgba(15,49,32,.82),rgba(15,49,32,.18)),url(data:image/svg+xml;base64,{AGRICULTURE_ART})">
            <h1>🌾 TamilNadu Agriculture : Admin Dashboard</h1>
            <p>Local operations dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, login_col, _ = st.columns([1, 1.1, 1])
    with login_col:
        st.subheader('Sign in')
        with st.form('admin_login_form', clear_on_submit=False):
            st.text_input('**UserName**', value='Admin', disabled=True)
            entered = st.text_input('**Password**', type='password', autocomplete='current-password')
            submitted = st.form_submit_button('🔐 Sign In', type='primary', use_container_width=True)
    if submitted:
        if hmac.compare_digest(entered, password):
            st.session_state['authenticated'] = True
            st.rerun()
        else:
            st.error('Invalid password')
    st.stop()

header_left, header_right = st.columns([8, 1], vertical_alignment='center')
with header_left:
    st.markdown(
        f"""
        <div class="hero" style="background-image:linear-gradient(90deg,rgba(15,49,32,.82),rgba(15,49,32,.18)),url(data:image/svg+xml;base64,{AGRICULTURE_ART})">
            <h1>🌾 TamilNadu Agriculture : Admin Dashboard</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.write('')
    if st.button('🚪 Sign out', use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
st.caption('ADMIN CONSOLE  /  OPERATIONS OVERVIEW')

try:
    stats = manager.preview()
except Exception as exc:
    stats = None
    preview_error = str(exc)
else:
    preview_error = None
scheme_records, scheme_data_error = read_scheme_data()

st.sidebar.subheader('📊 Overview')
sidebar_values = (
    len(scheme_records) if scheme_records else None,
    stats['chunks'] if stats else None,
    stats['new_vectors'] if stats else None,
    stats['estimated_tokens'] if stats else None,
)
for label, value in zip(
    ['📄 Schemes', '🧩 Chunks', '✨ New vectors', '🧠 Est. tokens'],
    sidebar_values,
):
    st.sidebar.metric(label, f'{value:,}' if value is not None else '—')

tab1, = st.tabs(['📥 Extract & load schemes'])

with tab1:
    with st.container(border=True):
        st.markdown(
            '<div class="step-heading"><span class="step-number">01</span>'
            '<span>Extract scheme details</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '[Open TN Agriculture portal](https://www.tnagrisnet.tn.gov.in/people_app/GoScheme) '
            'to collect scheme details and each scheme’s additional-detail information into JSON.'
        )
        confirm = st.checkbox('Enable to Extract and save Schemes to JSON')
        if st.button('🌾 Extract and save schemes', disabled=not confirm, type='primary'):
            with st.spinner('Opening the TN Agriculture portal and extracting scheme data…'):
                try:
                    result = subprocess.run(
                        [sys.executable, str(ROOT / 'pipeline' / 'scrape_tn_agriculture_schemes.py')],
                        cwd=ROOT,
                        input='\n',
                        capture_output=True,
                        text=True,
                        timeout=900,
                        errors='replace',
                    )
                    st.session_state['scheme_load_output'] = (
                        result.stdout + '\n' + result.stderr
                    )[-8000:]
                    loaded_records, validation_error = read_scheme_data()
                    if result.returncode == 0 and loaded_records is not None:
                        st.session_state['scheme_load_message'] = (
                            f'Extracted {len(loaded_records)} schemes and saved the JSON.'
                        )
                        st.session_state.pop('scheme_load_error', None)
                    else:
                        st.session_state['scheme_load_error'] = (
                            validation_error or f'Scraper exited with code {result.returncode}.'
                        )
                        st.session_state.pop('scheme_load_message', None)
                    st.rerun()
                except subprocess.TimeoutExpired:
                    st.session_state['scheme_load_error'] = 'Scraping timed out after 15 minutes.'
                except OSError as exc:
                    st.session_state['scheme_load_error'] = f'Could not start scraper: {exc}'

        if st.session_state.get('scheme_load_message'):
            st.success(st.session_state['scheme_load_message'])
        if st.session_state.get('scheme_load_error'):
            st.error(st.session_state['scheme_load_error'])
        if st.session_state.get('scheme_load_output'):
            with st.expander('View extraction log'):
                st.code(st.session_state['scheme_load_output'])

        if scheme_data_error:
            st.warning(f'Step 1 required: {scheme_data_error}')
        else:
            st.success(f'✅ JSON ready · {len(scheme_records)} schemes')
            st.caption(f'Saved in `{manager.DATA.name}`')
            try:
                loaded_ids = set(indexed_scheme_ids(str(manager.INDEX)))
                index_status_error = None
            except Exception as exc:
                loaded_ids = set()
                index_status_error = str(exc)
            if index_status_error:
                st.warning(f'Could not inspect existing vector data: {index_status_error}')
            with st.expander(f"View schemes ({len(scheme_records)})"):
                st.dataframe(
                    [
                        {
                            'scheme_id': str(x.get('scheme_id', '')),
                            'scheme_name': x.get('scheme_name', ''),
                            'scrape_status': x.get('detail_status', ''),
                            'vector_store': (
                                '✅ Already loaded'
                                if str(x.get('scheme_id', '')).strip() in loaded_ids
                                else '➕ Not loaded'
                            ) if not index_status_error else '❔ Unknown',
                        }
                        for x in scheme_records
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    with st.container(border=True):
        st.markdown(
            '<div class="step-heading"><span class="step-number">02</span>'
            '<span>Load scheme JSON to vector store</span></div>',
            unsafe_allow_html=True,
        )
        st.caption('Text splitter  →  embeddings  →  FAISS')
        if preview_error:
            st.error(f'Index preview unavailable: {preview_error}')
        elif stats:
            st.caption(
                f"Chunks: {stats['chunks']:,} · New embeddings: {stats['new_vectors']:,} · "
                f"Estimated tokens: {stats['estimated_tokens']:,}"
            )
            confirm_index = st.checkbox('Enable loading JSON to the vector store')
            if st.button(
                '🧠 Load JSON scheme to vector store',
                disabled=not bool(scheme_records) or not confirm_index,
                type='primary',
            ):
                with st.spinner('Splitting text, creating embeddings, and building FAISS…'):
                    try:
                        result = manager.build()
                        st.session_state['index_load_message'] = result['message']
                        st.session_state['index_load_tokens'] = result.get('actual_tokens', 0)
                        indexed_scheme_ids.clear()
                        st.rerun()
                    except Exception as exc:
                        st.error(f'Vector store load failed: {exc}')
        else:
            st.error('Cannot prepare vector-store preview.')
        if not scheme_records:
            st.caption('Complete Step 1 and load valid scheme JSON to enable this action.')
        if st.session_state.get('index_load_message'):
            st.success(st.session_state['index_load_message'])
            st.metric('Embedding tokens used in this run', f"{st.session_state.get('index_load_tokens', 0):,}")
