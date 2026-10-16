"""Local-only maintenance UI. Set ADMIN_PASSWORD in .env before launching."""
import base64
import hmac
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv
from scripts import index_manager as manager

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
    stats['entities'] if stats else None,
    stats['relations'] if stats else None,
)
for label, value in zip(
    ['📄 Schemes', '🧩 Chunks', '🔗 Entities', '🔁 Relations'],
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
                        [sys.executable, str(ROOT / 'scripts' / 'scrape_tn_agriculture_schemes.py')],
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
            graph_nodes = set()
            if manager.GRAPH.is_file():
                try:
                    graph = manager.graph_retriever.load_graph(manager.GRAPH)
                    graph_nodes = {
                        str(attributes.get('scheme_id'))
                        for _, attributes in graph.nodes(data=True)
                        if attributes.get('kind') == 'scheme'
                    }
                except Exception as exc:
                    st.warning(f'Could not inspect existing graph: {exc}')
            with st.expander(f"View schemes ({len(scheme_records)})"):
                st.dataframe(
                    [
                        {
                            'scheme_id': str(x.get('scheme_id', '')),
                            'scheme_name': x.get('scheme_name', ''),
                            'scrape_status': x.get('detail_status', ''),
                            'knowledge_graph': (
                                '✅ Loaded'
                                if str(x.get('scheme_id', '')).strip() in graph_nodes
                                else '➕ Not loaded'
                            ),
                        }
                        for x in scheme_records
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    with st.container(border=True):
        st.markdown(
            '<div class="step-heading"><span class="step-number">02</span>'
            '<span>Build knowledge graph</span></div>',
            unsafe_allow_html=True,
        )
        st.caption('Text splitter  →  entity/relation extraction  →  NetworkX graph')
        if preview_error:
            st.error(f'Index preview unavailable: {preview_error}')
        elif stats:
            st.caption(
                f"Chunks: {stats['chunks']:,} · Existing graph: "
                f"{'current' if stats['unchanged_graph'] else 'needs build/refresh'}"
            )
            if stats.get('graph_error'):
                st.warning(f"Existing knowledge graph is invalid and will be rebuilt: {stats['graph_error']}")
            elif stats.get('graph_present') and not stats.get('graph_current'):
                st.warning('Knowledge graph is missing current index content and will be rebuilt.')
            confirm_index = st.checkbox('Enable building the knowledge graph')
            if st.button(
                '🧠 Build knowledge graph from scheme JSON',
                disabled=not bool(scheme_records) or not confirm_index,
                type='primary',
            ):
                progress = st.progress(0, text='Preparing graph extraction…')
                progress_caption = st.empty()

                def show_progress(completed, total):
                    step = max(1, total // 200)
                    if completed == total or completed % step == 0:
                        progress.progress(
                            completed / total,
                            text=f'Extracting facts: {completed:,} / {total:,} chunks',
                        )
                        progress_caption.caption(
                            'LLM extraction is running concurrently; this may take several minutes.'
                        )

                try:
                    result = manager.build(progress_callback=show_progress)
                    st.session_state['index_load_message'] = result['message']
                    st.session_state['graph_extraction_calls'] = result.get('extraction_calls', 0)
                    st.rerun()
                except Exception as exc:
                    st.error(f'Knowledge graph build failed: {exc}')
        else:
            st.error('Cannot prepare the knowledge-graph build preview.')
        if not scheme_records:
            st.caption('Complete Step 1 and load valid scheme JSON to enable this action.')
        if st.session_state.get('index_load_message'):
            st.success(st.session_state['index_load_message'])
            st.metric(
                'Entity/relation extraction calls',
                f"{st.session_state.get('graph_extraction_calls', 0):,}",
            )
