"""
FastAPI application entry point.

Endpoints:
  POST /chat            - RAG + tool-calling chat completion
  POST /ingest          - upload a text/markdown document into the vector store
  GET  /health          - liveness/readiness probe
  GET  /sessions/{id}   - fetch chat history for a session

Run locally:  uvicorn app.main:app --reload
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.logging_config import configure_logging, get_logger
from app.database import init_db, get_db
from app import crud, rag
from app.schemas import ChatRequest, ChatResponse, IngestResponse, HealthResponse, SourceChunk
from app.llm_client import call_llm, AllProvidersFailedError, RateLimitExceededError
from app.tools import execute_tool
from app.cache import response_cache, make_key

settings = get_settings()
configure_logging()
logger = get_logger(__name__)

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: initializing database and vector store...")
    init_db()
    logger.info("Startup complete. Vector store stats: %s", rag.collection_stats())
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Assistant API",
    description="RAG-powered AI assistant with tool calling, caching, retries, and fallback providers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


SYSTEM_PROMPT = (
    "You are a precise, helpful AI assistant with access to a knowledge base "
    "and a small set of tools. Use retrieved context when it is relevant, cite "
    "the source tag (e.g. [filename.txt#chunk_id]) when you rely on it, and "
    "say clearly when you don't know something rather than guessing. Keep "
    "answers concise and well-structured."
)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        stats = rag.collection_stats()
        vector_ok = True
    except Exception:  # noqa: BLE001
        vector_ok = False
    # We don't ping paid providers on every health check to avoid burning quota;
    # this reports configuration presence, not live connectivity.
    primary_ok = bool(settings.LLM_PROVIDER)
    fallback_ok = bool(settings.FALLBACK_LLM_PROVIDER)
    status = "ok" if vector_ok and primary_ok else "degraded"
    return HealthResponse(status=status, vector_store=vector_ok, primary_llm=primary_ok, fallback_llm=fallback_ok)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith((".txt", ".md")):
        raise HTTPException(400, "Only .txt and .md files are supported in this demo ingestion endpoint.")

    raw_bytes = await file.read()
    text = raw_bytes.decode("utf-8", errors="ignore")

    num_chunks = rag.ingest_document(file.filename, text)
    doc = crud.record_document(db, filename=file.filename, num_chunks=num_chunks)
    return IngestResponse(document_id=doc.id, filename=doc.filename, num_chunks=num_chunks)


@app.get("/sessions/{session_id}")
def get_session_history(session_id: str, db: Session = Depends(get_db)):
    history = crud.get_history(db, session_id)
    if not history:
        raise HTTPException(404, "Session not found or has no messages.")
    return [{"role": m.role, "content": m.content, "created_at": m.created_at.isoformat()} for m in history]


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
def chat(request: Request, payload: ChatRequest, db: Session = Depends(get_db)):
    start = time.perf_counter()
    session = crud.get_or_create_session(db, payload.session_id)
    crud.add_message(db, session.id, "user", payload.message)

    # --- Cache check -----------------------------------------------------
    cache_key = make_key(payload.message, str(payload.use_rag), str(payload.allow_tools))
    cached = response_cache.get(cache_key)
    if cached is not None:
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        return ChatResponse(session_id=session.id, latency_ms=latency_ms, cached=True, **cached)

    # --- Retrieval ---------------------------------------------------------
    sources = rag.retrieve(payload.message) if payload.use_rag else []
    context_block = rag.build_context_block(sources)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context_block:
        messages.append({"role": "system", "content": context_block})

    history = crud.get_history(db, session.id, limit=10)
    for m in history[:-1]:  # exclude the message we just added, added separately below
        if m.role in ("user", "assistant"):
            messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": payload.message})

    # --- LLM call (+ tool-calling loop) ------------------------------------
    tool_call_log = []
    try:
        result = call_llm(messages, allow_tools=payload.allow_tools)

        # Simple single-pass tool loop: if the model requested tools, execute
        # them and ask it to produce a final answer using the results.
        if result.tool_calls:
            messages.append({"role": "assistant", "content": result.text or ""})
            for tc in result.tool_calls:
                tool_result = execute_tool(tc["name"], tc["arguments"])
                tool_call_log.append({"name": tc["name"], "arguments": tc["arguments"], "result": tool_result})
                messages.append(
                    {"role": "user", "content": f"Tool '{tc['name']}' returned: {tool_result}"}
                )
            result = call_llm(messages, allow_tools=False)

    except RateLimitExceededError as exc:
        raise HTTPException(429, str(exc)) from exc
    except AllProvidersFailedError as exc:
        logger.error("All LLM providers failed: %s", exc)
        raise HTTPException(503, "AI service is temporarily unavailable. Please try again shortly.") from exc

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    crud.add_message(db, session.id, "assistant", result.text, provider_used=result.provider, latency_ms=latency_ms)

    response_payload = {
        "answer": result.text,
        "provider_used": result.provider,
        "used_fallback": result.provider == settings.FALLBACK_LLM_PROVIDER,
        "sources": [s.model_dump() for s in sources],
        "tool_calls": tool_call_log,
    }
    response_cache.set(cache_key, response_payload)

    return ChatResponse(session_id=session.id, latency_ms=latency_ms, cached=False, **response_payload)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal server error", "detail": str(exc)})
