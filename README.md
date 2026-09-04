# AI Assistant — RAG Chat System (Task 1 + Task 2)

A production-oriented AI assistant that satisfies both problem-set tracks:

- **Task 1 (Applied AI)** — LLM integration (OpenAI/Anthropic/Gemini), prompt engineering with tunable `temperature`/`top_p`, structured JSON responses, function/tool calling, a full RAG pipeline (chunking → embeddings → ChromaDB vector store → retrieval), and an option to serve an open-source model locally via **vLLM**.
- **Task 2 (Engineering AI Systems)** — a Streamlit web UI wired to the FastAPI backend, async/concurrent request handling, prompt/response caching, retries, rate limiting, automatic provider fallback, structured error handling, and Docker/Docker Compose deployment.

## 📁 Project structure

```
├── .github/workflows/ci.yml   # GitHub Actions: tests + Docker build
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entry point (/chat /ingest /health /sessions)
│   ├── config.py                # Centralized settings (env-driven)
│   ├── crud.py                  # CRUD operations
│   ├── database.py               # DB connection & session management
│   ├── logging_config.py         # Structured logging
│   ├── llm_client.py             # Multi-provider LLM client: retry, fallback, rate limit
│   ├── rag.py                    # Chunking, embeddings, ChromaDB retrieval ("ml-model" logic)
│   ├── tools.py                  # Function-calling tool definitions
│   ├── cache.py                  # TTL/LRU prompt-response cache
│   ├── models.py                 # ORM models
│   └── schemas.py                # Pydantic request/response schemas
├── data/
│   ├── app.db                   # SQLite database (created on first run)
│   └── sample_knowledge_base.txt
├── docs/
│   ├── index.md                 # MkDocs homepage
│   └── architecture.md          # Architecture diagram + design notes
├── env/                         # Reserved for env-specific overlays
├── models/                      # Reserved for local model configs/checkpoints
├── scripts/
│   └── ingest_documents.py      # Bulk document ingestion CLI
├── tests/
│   └── test_app.py              # Unit + API tests (mocked LLM calls)
├── ui/
│   └── streamlit_app.py         # Streamlit chat UI
├── Dockerfile
├── .env.example
├── .gitignore
├── docker-compose.yml
├── mkdocs.yml
└── requirements.txt
```

## 🚀 Quick start (local, no Docker)

### Option A — using `uv` (recommended, faster)

```bash
uv sync                             # creates .venv and installs from pyproject.toml
cp .env.example .env                # fill in at least one provider's API key

# Run the API
uv run uvicorn app.main:app --reload

# In a second terminal, run the UI
API_BASE_URL=http://localhost:8000 uv run streamlit run ui/streamlit_app.py

# Run tests
uv run pytest -v

# Bulk-ingest documents
uv run python scripts/ingest_documents.py --source ./data
```

`uv sync` reads `pyproject.toml` and creates a `.venv` + `uv.lock` automatically — no manual venv activation needed since `uv run` uses that environment for you. If you want the dev/test dependencies too, that's already covered (they're in the `dev` group in `pyproject.toml`); to skip them use `uv sync --no-dev`.

### Option B — using `pip`/`venv`

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # fill in at least one provider's API key

# Run the API
uvicorn app.main:app --reload

# In a second terminal, run the UI
API_BASE_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
```

Visit `http://localhost:8000/docs` for interactive API docs (Swagger), and `http://localhost:8501` for the chat UI.

## 🐳 Quick start (Docker Compose)

```bash
cp .env.example .env               # fill in your API keys
docker compose up --build
```

- API: `http://localhost:8000`
- UI: `http://localhost:8501`

To additionally run a **local open-source model via vLLM** (requires an NVIDIA GPU host):

```bash
docker compose --profile local-llm up --build
```

Then set `LLM_PROVIDER=local` (or `FALLBACK_LLM_PROVIDER=local`) in `.env` to route traffic to it.

## 🔑 Configuration

All configuration lives in environment variables — see `.env.example` for the full list. Key ones:

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | Primary provider: `openai`, `anthropic`, `gemini`, or `local` |
| `FALLBACK_LLM_PROVIDER` | Secondary provider used automatically on primary failure |
| `TEMPERATURE`, `TOP_P`, `MAX_TOKENS` | Generation parameters |
| `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` | RAG chunking/retrieval tuning |
| `RATE_LIMIT_PER_MINUTE` | Per-client request cap |
| `MAX_RETRIES`, `RETRY_BACKOFF_SECONDS` | Retry policy for LLM calls |
| `CACHE_TTL_SECONDS`, `CACHE_MAX_SIZE` | Response cache tuning |

## 📖 API reference

### `POST /chat`
```json
{
  "session_id": null,
  "message": "What does the assistant do?",
  "use_rag": true,
  "allow_tools": true
}
```
Returns a structured JSON response (`ChatResponse`) with the answer, which provider served it, whether fallback was used, retrieved sources, any tool calls made, latency, and cache status.

### `POST /ingest`
Multipart file upload (`.txt` / `.md`) — chunks and embeds the document into the vector store.

### `GET /health`
Liveness/readiness probe reporting vector store and LLM provider configuration status.

### `GET /sessions/{session_id}`
Returns the stored chat history for a session.

## 🧠 RAG pipeline

1. **Ingestion** — via `POST /ingest` or `scripts/ingest_documents.py` for bulk loading a directory.
2. **Chunking** — sliding-window character-based chunker (`app/rag.py::chunk_text`), tunable via `CHUNK_SIZE`/`CHUNK_OVERLAP`.
3. **Embedding** — `sentence-transformers` (`all-MiniLM-L6-v2` by default), runs locally, no external API needed.
4. **Vector store** — ChromaDB, persisted to disk (`VECTOR_DB_PATH`), cosine similarity search.
5. **Retrieval** — top-k chunks are injected into the system prompt as cited context (`[filename#chunk_id]`), and are also available as an explicit `search_knowledge_base` tool for models that prefer to call retrieval themselves.

Bulk-load a folder of documents:
```bash
python scripts/ingest_documents.py --source ./knowledge_base
```

## 🛠️ Tool calling

Three tools are registered out of the box (`app/tools.py`): `calculator`, `get_current_time`, and `search_knowledge_base`. Add new tools by writing a function + JSON schema and registering both in `TOOL_SCHEMAS` / `TOOL_REGISTRY`.

## 🧯 Reliability features

- **Retries**: exponential backoff via `tenacity` on every provider call.
- **Fallback**: automatic switch to `FALLBACK_LLM_PROVIDER` if the primary exhausts its retries.
- **Rate limiting**: an in-process token-bucket limiter plus a per-route `slowapi` limiter.
- **Caching**: identical `(message, use_rag, allow_tools)` requests are served from a TTL/LRU cache.
- **Graceful degradation**: provider failures return a `503` with a clear message rather than crashing; a global exception handler catches anything unexpected and returns structured JSON.

See `docs/architecture.md` for the full system diagram and request-flow walkthrough.

## ⚡ Model optimization notes (Task 2)

- **ONNX conversion**: not applicable to the hosted-provider path (OpenAI/Anthropic/Gemini APIs don't expose local weights to convert). For the **local vLLM path**, vLLM's PagedAttention + continuous batching already provides the throughput/latency benefits ONNX conversion targets, so it's used as the optimization layer instead of a manual ONNX export — this trade-off is documented here rather than forcing an unnecessary conversion step.
- **Concurrency**: FastAPI/Uvicorn serve requests asynchronously; the LLM SDK calls in `llm_client.py` are synchronous but I/O-bound, so they yield to the event loop under Uvicorn's worker model — for heavier concurrent load, run multiple Uvicorn workers (`--workers N`) or replicas behind a load balancer.
- **Caching** reduces both latency and provider cost for repeated queries (see above).

## ✅ Testing

```bash
pytest -v
```

Covers chunking, ingestion/retrieval roundtrip, the cache, tool execution (including a rejected unsafe-expression case), and the `/chat`, `/ingest`, `/health` endpoints (with the LLM call mocked so no API key or network access is required in CI).

## ☁️ Deployment (bonus)

The image is a standard container and can be deployed to any container platform:

- **Azure**: push to Azure Container Registry, deploy via Azure Container Apps or AKS; use Azure Database for PostgreSQL instead of SQLite for `DATABASE_URL`.
- **AWS**: push to ECR, deploy via ECS Fargate or EKS; use RDS for Postgres and EFS (or S3-backed sync) for the persisted ChromaDB volume.
- **GCP**: push to Artifact Registry, deploy via Cloud Run (API) — note Cloud Run's ephemeral filesystem means the vector store volume should be moved to a persistent disk or a managed vector DB for production use there.

In all cases: set the `.env` values as secrets/environment variables in the platform's configuration rather than baking them into the image.
