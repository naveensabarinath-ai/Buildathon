# TamilNadu Agriculture Scheme Assistant

## Run locally on Windows

Open three terminals in the project root and activate the same Python environment
in each terminal if it is not already selected.

```powershell
python -m streamlit run apps\admin_dashboard.py --server.address 127.0.0.1 --server.port 8502
python -m uvicorn apps.chatbot_api:app --host 127.0.0.1 --port 8000
python -m streamlit run apps\chatbot_app.py
```

The admin dashboard and chatbot UI run in Streamlit. The chatbot UI requires the
FastAPI process to be running. Keep `.env` in the project root and configure
`ADMIN_PASSWORD` and `OPENAI_API_KEY` there. Do not share or commit `.env`.

## Optional LangSmith tracing

Chatbot question tracing is disabled by default. To enable it, configure these
values in the project-root `.env`, using a LangSmith API key from your LangSmith
account:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=tn-agriculture-chatbot
```

Tracing records the question, retrieved source context, and generated answer in
LangSmith. Only enable it if you are comfortable sending that chatbot content
to the LangSmith service. Restart FastAPI after changing these settings.

## Folder layout

```text
apps/       Streamlit dashboards and FastAPI entry point
assets/     Shared local visual assets
data/       Scraped scheme JSON and indexing database
indexes/    Published FAISS index and its manifest
pipeline/   Scraping, embedding/index-building, retrieval, and RAG code
services/   Shared index and chatbot usage services
```

The chatbot usage database remains at the project root as local state.

## Data workflow

1. In the admin dashboard, extract scheme data from the TN Agriculture portal.
2. Load the JSON into the vector store from Step 2 in the same tab.
3. Restart the FastAPI process after publishing a new FAISS index.
