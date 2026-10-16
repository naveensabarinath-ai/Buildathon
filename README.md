# Tamil Nadu Agriculture Scheme Assistant

This project answers questions about Tamil Nadu agriculture schemes using a
local **knowledge-graph-only retrieval** architecture:

- Playwright scrapes scheme listing and detail records from the Tamil Nadu
  Agriculture portal.
- Text is normalized into LangChain documents and selectively split into
  chunks.
- An LLM extracts typed entities and explicit subject-predicate-object facts
  from every chunk.
- NetworkX stores the source chunks, entities, relationships, scheme metadata,
  and provenance in one local graph artifact.
- For a question, an LLM extracts query entities/relations; graph traversal
  returns matching source chunks. A chat model generates a cited answer from
  those chunks.

**There is no vector store, embedding generation, or vector search in the
application path.** The graph itself stores the text excerpts used as answer
evidence.

## Architecture

```text
Tamil Nadu Agriculture portal
        |
        v
Playwright scraper
        |
        v
data/all_tn_agriculture_schemes.json
        |
        v
Document preparation and text splitting
        |
        v
LLM entity + relationship extraction (one extraction call per chunk)
        |
        v
NetworkX MultiGraph with chunk text and provenance
indexes/agriculture_knowledge_graph.json
        |
        +----------------------+
                               |
User question -> LLM query fact extraction
                               |
                               v
                 Entity/relation graph traversal
                               |
                               v
             Source excerpts -> answer LLM -> citations
```

The NetworkX graph is stored as JSON and loaded into memory by the API. It is
local to the application; no graph database server is required.

## Graph schema and implementation

The graph implementation is in
[`scripts/graph_retriever.py`](scripts/graph_retriever.py).
It uses `networkx.MultiGraph`, not a custom graph class. A `MultiGraph` allows
several relationships between the same entities while keeping each fact
linked to its source chunk.

### Nodes

| Node kind | Example ID | Important attributes |
|---|---|---|
| Scheme | `scheme:147` | `scheme_id`, `scheme_name` |
| Source chunk | `chunk:<sha256>` | Full `page_content`, original metadata, `scheme_id`, `content_type`, source fields |
| Extracted entity | `entity:benefit:subsidy` | Canonical `name`, `entity_type` |

The source chunk is intentionally stored in the graph. It is the evidence
store; the system does not need another document store to reconstruct answer
context.

### Edges

| Edge type | Connects | Attributes / purpose |
|---|---|---|
| `HAS_RECORD` | Scheme to source chunk | Associates an indexed record with its scheme |
| `MENTIONS` | Source chunk to entity | Records that an entity is supported/mentioned in this chunk |
| `FACT` | Subject entity to object entity | Stores a relation `predicate` and its `source_chunk_key` and `scheme_id` |

Example:

```text
scheme:147 --HAS_RECORD--> chunk:8a...
chunk:8a... --MENTIONS--> entity:benefit:subsidy
entity:Sub-Mission on Agricultural Mechanisation
    --FACT {predicate: "provides", source_chunk_key: "8a..."}-->
entity:benefit:subsidy
```

### Entity and relation extraction

`GraphEntity`, `GraphRelation`, and `ExtractedFacts` are Pydantic schemas used
as structured output for the extraction LLM. For each prepared text chunk,
`extract_facts()` asks the configured chat model to return:

- **Entities:** concise names and types, such as scheme, benefit, crop,
  beneficiary, document, eligibility condition, location, agency, amount, or
  application method.
- **Relations:** explicit subject, predicate, and object triples, for example
  `scheme --provides--> benefit`.

The extraction prompt instructs the LLM to use only facts supported by the
chunk, avoid outside knowledge, treat the input text as untrusted, and return
empty lists when no facts are explicit. If structured extraction fails, graph
building raises an error rather than silently creating an incomplete graph.

`entity_id()` normalizes an entity's name and type to form a stable node ID.
`chunk_key()` hashes chunk text together with sorted metadata so the source
chunk can be uniquely cited and linked from relation edges.

### Graph persistence

[`scripts/index_manager.py`](scripts/index_manager.py) performs graph
refresh:

1. `load_chunks()` loads the scraped JSON.
2. `scripts/scheme_documents.py` converts scheme records into LangChain
   documents and splits longer text. Eligibility and document-list chunks
   remain whole.
3. Each chunk is sent through structured LLM entity/relation extraction. Up to
   eight extraction requests run concurrently, submitted in batches of 32 to
   keep resource use bounded.
4. `build_graph()` assembles scheme, chunk, and entity nodes with provenance
   edges.
5. `save_graph()` writes graph metadata, nodes, and keyed edges to
   `indexes/agriculture_knowledge_graph.json`.
6. `indexes/graph_manifest.json` stores the digest of the source chunks.

The graph records its format and extraction-schema versions and the same
source digest. The API verifies the graph digest against the manifest at
startup. It refuses to serve requests if the graph is missing, incompatible,
or stale.

### Question-time graph retrieval

`retrieve_documents()` uses the same structured fact schema to extract
entities and relations from the user's question. It then:

1. Matches the extracted entity names and types to graph entity nodes (with a
   name-only fallback if the query extractor labels a type differently).
2. Traverses `MENTIONS` edges to find source chunks that mention a matched
   entity.
3. Traverses `FACT` edges and uses each fact's `source_chunk_key` to find the
   chunk that supports that relationship.
4. If the question contains a structured relation, gives a stronger match to
   graph facts whose subject, predicate, and object match that relation.
5. Applies a resolved scheme ID and detected field filter, ranks matching
   source chunks, and returns their stored text and metadata.
6. If a scheme is confidently resolved but query-extracted entities do not
   match graph entities, retrieves relevant chunks from that scheme rather than
   returning no evidence.

When the user selects a scheme and asks specifically for eligibility or
required documents, retrieval returns all graph chunks matching that scheme
and field without asking the query-extraction LLM. This preserves complete
checklists.

Broad discovery questions such as “Which schemes mention Chief Minister?”
search scheme names stored on graph scheme nodes and retrieve one descriptive
source chunk per matching scheme. These discovery requests are not restricted
by a scheme selected in the chatbot sidebar.

An empty `Pattern of Subsidy` value in a scraped detail row is rendered as
`Not specified in source` in its graph chunk. This distinguishes missing
source data from an extraction failure; it does not imply that the scheme has
no subsidy beyond what the source records.

The RAG layer in
[`scripts/agriculture_rag.py`](scripts/agriculture_rag.py) labels retrieved
chunks as `[S1]`, `[S2]`, and so forth. The answer model is instructed to use
only those excerpts, cite claims, and say when the graph has no evidence.

## LLM calls and cost considerations

Graph construction uses one structured extraction call per text chunk when a
rebuild is needed. Extraction runs with up to eight concurrent requests, and
the admin dashboard displays chunk-level progress. For example, the current
52-scheme input produces 1,647 chunks, so an initial build can still take
several minutes and incur usage for each chunk. Query processing normally uses one LLM call to extract
query facts and another to generate the answer. Exact scheme-and-field
retrieval skips query extraction but still uses the answer model if evidence
exists.

The admin preview detects when source chunks and graph manifest are unchanged;
in that case a build is skipped and no entity-extraction calls are made.
Changing the extraction schema version should trigger a rebuild so the graph
uses the new schema. Model usage returned by the API includes query extraction
and answer generation tokens. Admin build output reports extraction call and
token counts.

## Project structure

```text
Buildathon\
├── scripts\
│   ├── __init__.py                       # Shared Python package for all scripts
│   ├── admin_dashboard.py                 # Scraping and graph-build UI
│   ├── agriculture_rag.py                 # Graph evidence and answer generation
│   ├── chatbot_api.py                     # FastAPI backend
│   ├── chatbot_app.py                     # Streamlit chatbot
│   ├── chat_usage.py                      # Local model-token/cost estimates
│   ├── graph_retriever.py                 # Graph schema, extraction, traversal
│   ├── index_manager.py                   # Graph preview, build and publish
│   ├── scheme_documents.py                # Document conversion and splitting
│   ├── scrape_tn_agriculture_schemes.py   # Portal scraper
│   ├── test_chat_usage.py                 # Usage migration tests
│   └── test_knowledge_graph.py             # Graph extraction/retrieval tests
├── data\
│   └── all_tn_agriculture_schemes.json    # Scraper output (generated)
├── indexes\
│   ├── agriculture_knowledge_graph.json   # Graph plus source-chunk text
│   └── graph_manifest.json                # Source-content digest
├── requirements.txt
└── README.md
```

`.env`, scraped source data, the graph, and local databases may contain
application or usage data. Do not commit `.env` or share its secrets.

## Windows setup

Run commands from `E:\KG_RAG\Buildathon`. These commands use the project's
`venv` virtual environment.

Install the Python dependencies:

```powershell
Set-Location E:\KG_RAG\Buildathon
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install Playwright Chromium once if portal scraping is needed:

```powershell
.\venv\Scripts\python.exe -m playwright install chromium
```

Configure `.env` at the repository root:

| Variable | Required | Purpose |
|---|---:|---|
| `OPENAI_API_KEY` | Yes for graph extraction and chat | Authenticates chat-model calls |
| `ADMIN_PASSWORD` | Yes for admin UI | Protects admin login |
| `OPENAI_CHAT_MODEL` | No | Chat/extraction model; defaults to `gpt-4.1-mini` |
| `RAG_API_URL` | No | API base URL; defaults to `http://127.0.0.1:8000` |
| `LANGSMITH_TRACING` | No | Enable tracing with `true`, `1`, `yes`, or `on` |
| `LANGSMITH_API_KEY` | Only when tracing is enabled | LangSmith authentication |
| `LANGSMITH_PROJECT` | No | LangSmith project name |
| `CHAT_INPUT_USD_PER_M` | No | Optional prompt-token rate for local cost estimates |
| `CHAT_OUTPUT_USD_PER_M` | No | Optional completion-token rate for local cost estimates |

When LangSmith tracing is enabled, question and answer content may be sent to
LangSmith. Enable it only if that external sharing is acceptable.

## Run the application

### 1. Scrape scheme data

Start the admin dashboard:

```powershell
Set-Location E:\KG_RAG\Buildathon
.\venv\Scripts\python.exe -m streamlit run scripts\admin_dashboard.py --server.address 127.0.0.1 --server.port 8502
```

Open `http://127.0.0.1:8502`, sign in, and run the extraction step if
`data\all_tn_agriculture_schemes.json` is missing or out of date. The scraper
uses a visible Playwright Chromium browser.

### 2. Build or refresh the graph

In the admin dashboard, confirm the graph build and click **Build knowledge
graph from scheme JSON**. The admin preview shows source chunks and current
graph entity/relation counts. Stop the chatbot API before publishing a graph
refresh, then restart it after the build succeeds.

The first graph build performs an LLM extraction for each chunk and can incur
model usage. If the content digest and graph schema are unchanged, the build
is skipped. The graph itself is stored locally; no separate graph database
server is required.

### 3. Start the API

In a separate PowerShell terminal:

```powershell
Set-Location E:\KG_RAG\Buildathon
.\venv\Scripts\python.exe -m uvicorn scripts.chatbot_api:app --host 127.0.0.1 --port 8000
```

`GET http://127.0.0.1:8000/health` reports status, scheme count, and graph-node
count. A missing or stale graph causes a not-ready response.

### 4. Start the chatbot

In another terminal:

```powershell
Set-Location E:\KG_RAG\Buildathon
.\venv\Scripts\python.exe -m streamlit run scripts\chatbot_app.py
```

Open the Streamlit URL printed in the terminal, normally
`http://localhost:8501`. The UI calls the FastAPI API; the API performs graph
retrieval and answer generation.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Validate and summarize the loaded graph |
| `GET /schemes` | List schemes found in graph scheme nodes |
| `POST /ask` | Extract query facts, retrieve graph evidence, and generate an answer |

Example request:

```json
{
  "question": "What subsidy supports drip irrigation?",
  "scheme_id": null,
  "k": 5
}
```

The response includes `answer`, cited `sources`, `retrieval` diagnostics, and
model token usage when supplied by the model API.

## Tests

Run the focused test suite from the repository root:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Limitations and operational notes

- Graph retrieval depends on LLM-extracted entity and relation names. If the
  query extraction uses names that do not align with the graph's canonical
  names, a relevant record may not be found; inspect extraction quality and
  adjust the prompts or normalization if needed.
- The extractor is instructed to use explicit source facts only, but LLM
  extraction is not a formal guarantee. Every relation retains its source
  chunk so answers can cite and be checked against the original text.
- This implementation does not provide vector similarity, vector embeddings,
  or automatic semantic synonym search. Add synonyms or richer normalization
  explicitly if users need alternate terminology.
- The graph JSON stores source text as well as entities and relationships. It
  is the local source store, and it should be protected/backed up accordingly.
- Answers are based on locally scraped data and do not check live scheme
  validity at question time.
- Keep local servers bound to `127.0.0.1` unless deployment security is
  deliberately configured.


.\venv\Scripts\python.exe -m streamlit run scripts\admin_dashboard.py
.\venv\Scripts\python.exe -m uvicorn scripts.chatbot_api:app --host 127.0.0.1 --port 8000
.\venv\Scripts\python.exe -m streamlit run scripts\chatbot_app.py