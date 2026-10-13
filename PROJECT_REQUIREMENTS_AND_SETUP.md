# TamilNadu Agriculture Scheme Assistant

This guide describes the implemented project requirements, how the application
is assembled, the role of each file and folder, and how to run it on Windows.

## 1. Implemented requirements

The project provides a local scheme-question answering application with an
administrative data refresh workflow.

### Scheme extraction and indexing

1. The admin dashboard launches a visible Chromium browser through Playwright
   and opens the Tamil Nadu Agriculture scheme portal.
2. The scraper reads the scheme listing and calls the portal's own
   `getSchemeDet(id)` function to retrieve scheme-detail modal content,
   including paginated detail rows.
3. Scraped records and listing data are saved incrementally as JSON, with a
   text report, under `data\`.
4. The admin dashboard validates the JSON and shows the extracted schemes.
   When an existing FAISS index is present, it can show whether each scheme ID
   appears in its stored metadata.
5. The index manager converts JSON fields and detail rows into LangChain
   documents, selectively splits text, and embeds the resulting chunks.
6. Embeddings are cached locally in SQLite using a hash of the embedding model
   and text. Matching cached vectors can be reused.
7. A new FAISS index is built in a staging location and published only after
   the index files are verified. Stop the chatbot API before publishing an
   index, then restart it so its in-memory store reloads.

### Chatbot

1. The Streamlit chatbot UI gets the available scheme list and service status
   from the local FastAPI API.
2. The user can ask in Tamil or English, optionally select a scheme, choose a
   retrieval method, and set the number of semantic results.
3. The API retrieves evidence from the local FAISS index. The RAG layer selects
   relevant evidence and asks the configured OpenAI chat model to answer using
   that evidence.
4. Answers include source labels and retrieval information. The app answers
   from the indexed records; it does not verify current government information
   live against the portal.
5. The API records chatbot token and estimated-cost usage locally in
   `chat_usage.db`. These records are usage estimates/accounting, not an
   OpenAI balance or quota.

### Optional LangSmith tracing

LangSmith tracing is disabled unless `LANGSMITH_TRACING` is explicitly set to a
truthy value. When enabled, the trace can include the question, retrieved
scheme context, and generated answer. This content is sent to LangSmith. Set a
LangSmith API key and project name in the root `.env` only if that external
sharing is acceptable. Restart FastAPI after changing tracing settings.

## 2. Implementation plan: what was implemented and how

| Area | Implementation |
|---|---|
| Admin UI | Streamlit password gate using `ADMIN_PASSWORD`; scheme extraction and vector indexing are exposed as two steps in one tab. |
| Scraping | Playwright controls Chromium; BeautifulSoup parses listing and scheme-detail HTML; JSON and report are written after each processed scheme. |
| Text preparation | `pipeline\build_faiss_index.py` converts listing fields and detail rows into LangChain documents and selectively splits longer content. |
| Embedding reuse | `services\index_manager.py` caches vectors in `data\admin_metrics.db`, keyed by the selected embedding model and content hash. |
| Vector search | FAISS index files are stored under `indexes\faiss_tn_agriculture\`; `index_manifest.json` records the indexed-content digest. |
| Answer generation | The API invokes the RAG implementation, which retrieves local evidence and sends it to the configured chat model. |
| Chat UI | Streamlit calls the local API; it does not load FAISS or call the model directly. |
| Usage history | `services\chat_usage.py` records token metadata and estimated costs in root-level `chat_usage.db`. |

## 3. Files and folders

```text
Buildathon\
├── apps\
│   ├── __init__.py
│   ├── admin_dashboard.py                 # Admin: scrape and build index
│   ├── chatbot_api.py                     # FastAPI backend
│   └── chatbot_app.py                     # Streamlit chat interface
├── assets\
│   └── agriculture_hero.svg               # Shared local agriculture header art
├── data\
│   ├── all_tn_agriculture_schemes.json    # Scraper output; generated
│   └── admin_metrics.db                   # Embedding cache and index-job history
├── indexes\
│   ├── faiss_tn_agriculture\
│   │   ├── index.faiss                    # Generated FAISS vectors
│   │   └── index.pkl                      # Generated document store/metadata
│   └── index_manifest.json                # Digest for the published index
├── pipeline\
│   ├── __init__.py
│   ├── agriculture_rag.py                 # Evidence selection and answer generation
│   ├── build_faiss_index.py               # Document conversion and splitting
│   ├── scheme_retriever.py                # FAISS load and retrieval helpers
│   └── scrape_tn_agriculture_schemes.py   # Portal scraper
├── services\
│   ├── chat_usage.py                      # Chat usage persistence
│   └── index_manager.py                   # Preview, embedding cache, index build
├── .env                                   # Local secrets and configuration
├── requirements.txt                       # Python dependencies
├── README.md                              # Short project overview
└── PROJECT_REQUIREMENTS_AND_SETUP.md      # This guide
```

Other local/runtime items may include `venv_rag\`, Python `__pycache__\`
folders, and `chat_usage.db`. The virtual environment can be recreated from
`requirements.txt`; usage history is local data and is lost if its database is
deleted. Scraper output such as `data\scheme_scrape_report.txt` is generated
when extraction runs.

## 4. Prerequisites

- Windows with PowerShell.
- Python installed, or the existing project virtual environment at
  `venv_rag\`.
- Dependencies from `requirements.txt`.
- Playwright's Chromium browser for scheme extraction.
- An OpenAI API key with access to the configured chat and embedding models.
- A non-empty `ADMIN_PASSWORD` in the project-root `.env` to sign in to the
  admin dashboard.
- A valid scraped JSON file and FAISS index before the chatbot can answer
  questions.
- Optional: a LangSmith API key and project setting if tracing is enabled.

### Environment configuration

Keep `.env` at the project root. Never commit it or share its secret values.
The implementation reads these variable names:

| Variable | Required | Purpose |
|---|---:|---|
| `ADMIN_PASSWORD` | Yes for admin UI | Admin dashboard sign-in password. |
| `OPENAI_API_KEY` | Yes for embedding/indexing and chatbot model calls | Authenticates OpenAI requests. |
| `OPENAI_CHAT_MODEL` | No | Chat model; defaults to `gpt-4.1-mini`. |
| `OPENAI_EMBEDDING_MODEL` | No | Embedding model used by the incremental index manager; defaults to `text-embedding-3-small`. |
| `RAG_API_URL` | No | Chat UI's API base URL; defaults to `http://127.0.0.1:8000`. |
| `LANGSMITH_TRACING` | No | Enables tracing only when set to `true`, `1`, `yes`, or `on`. |
| `LANGSMITH_API_KEY` | Only when tracing is enabled | Authenticates LangSmith tracing. |
| `LANGSMITH_PROJECT` | No | Trace project name; defaults to `tn-agriculture-chatbot`. |

The standalone index-building script currently defines its embedding model in
code; the admin dashboard's incremental index manager uses
`OPENAI_EMBEDDING_MODEL`.

## 5. Install dependencies and browser

From the project root in PowerShell:

```powershell
cd E:\Projects\Buildathon
.\venv_rag\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m playwright install chromium
```

If you create a different virtual environment, activate that environment
instead. Run all three application commands from the project root so that the
project modules and root `.env` are resolved correctly.

## 6. Fresh setup and run order

### A. Start the admin dashboard

In Terminal 1:

```powershell
cd E:\Projects\Buildathon
.\venv_rag\Scripts\Activate.ps1
python -m streamlit run apps\admin_dashboard.py --server.address 127.0.0.1 --server.port 8502
```

Open `http://127.0.0.1:8502` and sign in using the configured admin password.

### B. Extract scheme data

In the dashboard's **Extract & Load Schemes** tab:

1. Enable **Enable to Extract and save Schemes to JSON**.
2. Click **Extract and save schemes**.
3. Wait for Chromium scraping to finish. The JSON is written to
   `data\all_tn_agriculture_schemes.json`.
4. Optionally open **View schemes** to inspect the extracted records.

### C. Build or refresh the FAISS index

1. Stop any running chatbot API process before publishing the index.
2. In Step 2, enable **Enable loading JSON to the vector store**.
3. Click **Load JSON scheme to vector store**.
4. Review the result and embedding tokens used for the run.

New or changed text may trigger OpenAI embedding requests and usage charges.
The SQLite embedding cache is retained between index rebuilds so matching
model/text vectors can be reused.

### D. Start the chatbot API

In Terminal 2, after indexing has completed:

```powershell
cd E:\Projects\Buildathon
.\venv_rag\Scripts\Activate.ps1
python -m uvicorn apps.chatbot_api:app --host 127.0.0.1 --port 8000
```

The API loads and caches the current FAISS store at startup.

### E. Start the chatbot UI

In Terminal 3:

```powershell
cd E:\Projects\Buildathon
.\venv_rag\Scripts\Activate.ps1
python -m streamlit run apps\chatbot_app.py
```

Open the Streamlit URL printed in the terminal (normally
`http://localhost:8501`). The chatbot UI requires the API from Terminal 2 to
remain running.

## 7. Important operational notes

- The admin dashboard and chatbot are intended for local use. Keep servers
  bound to `127.0.0.1` unless deployment security is deliberately configured.
- Stop FastAPI before rebuilding/publishing the FAISS index. Restart it after
  a successful build to avoid querying a stale in-memory index.
- The FAISS `index.pkl` file is pickle-serialized and must be treated as trusted
  local data only.
- Deleting `data\admin_metrics.db` removes the embedding cache. A later build
  may need to request embeddings again, potentially increasing API usage.
- Deleting `data\all_tn_agriculture_schemes.json` removes the extracted source
  data; the chatbot index alone is not enough to use the admin rebuild flow.
- Deleting `indexes\faiss_tn_agriculture\` removes the searchable index; rebuild
  it from the JSON before starting the chatbot API.
- `chat_usage.db` stores optional local chatbot usage history. It is not
  required to load the FAISS index or produce chatbot answers.
- Lines beginning with `#` in `.env` are comments and do not execute commands.
