# AI Assistant

A production-oriented AI assistant combining:

- **Task 1 — Applied AI**: LLM integration, prompt engineering, structured output, tool calling, and a full RAG pipeline (chunking → embeddings → vector DB → retrieval), with an option to serve an open-source model locally via vLLM.
- **Task 2 — Engineering AI Systems**: a Streamlit UI on top of the FastAPI backend, concurrent/async request handling, latency optimization, prompt/response caching, retries, rate limiting, provider fallback, graceful error handling, and Docker/Docker Compose deployment.

See [Architecture](architecture.md) for the full system diagram and component breakdown, and the project `README.md` for setup and run instructions.

## Quick links

- API docs (Swagger UI): `http://localhost:8000/docs` once the app is running
- Streamlit UI: `http://localhost:8501`
- Health check: `GET /health`
