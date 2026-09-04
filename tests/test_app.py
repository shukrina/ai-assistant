"""
Basic test suite covering the RAG pipeline, cache, tools, and API endpoints.

Run:  pytest -v
Note: /chat tests monkeypatch call_llm so no real API key/network is needed.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import rag, tools
from app.cache import TTLCache, make_key
from app.llm_client import LLMResult


@pytest.fixture
def client():
    return TestClient(app)


# --------------------------------------------------------------------------
# RAG pipeline
# --------------------------------------------------------------------------
def test_chunk_text_splits_long_text():
    text = "word " * 1000
    chunks = rag.chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_chunk_text_short_text_single_chunk():
    text = "This is a short sentence."
    chunks = rag.chunk_text(text, chunk_size=500, overlap=50)
    assert chunks == [text]


def test_ingest_and_retrieve_roundtrip():
    rag.ingest_document("test_doc.txt", "The Eiffel Tower is located in Paris, France. It was completed in 1889.")
    results = rag.retrieve("Where is the Eiffel Tower?", top_k=1)
    assert len(results) >= 1
    assert "Paris" in results[0].text or "Eiffel" in results[0].text


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def test_cache_set_and_get():
    cache = TTLCache(max_size=10, ttl_seconds=60)
    key = make_key("hello", "world")
    assert cache.get(key) is None
    cache.set(key, {"answer": "hi"})
    assert cache.get(key) == {"answer": "hi"}


def test_cache_evicts_oldest_when_full():
    cache = TTLCache(max_size=2, ttl_seconds=60)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------
def test_calculator_tool_valid_expression():
    result = tools.calculator("2 + 2 * 3")
    assert result["result"] == 8


def test_calculator_tool_rejects_unsafe_input():
    result = tools.calculator("__import__('os').system('echo hi')")
    assert "error" in result


def test_execute_tool_unknown_name():
    result = tools.execute_tool("not_a_real_tool", {})
    assert "error" in result


# --------------------------------------------------------------------------
# API endpoints
# --------------------------------------------------------------------------
def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("ok", "degraded")


def test_chat_endpoint_with_mocked_llm(client, monkeypatch):
    def fake_call_llm(messages, allow_tools=True):
        return LLMResult(text="This is a mocked answer.", provider="openai", tool_calls=[])

    monkeypatch.setattr("app.main.call_llm", fake_call_llm)

    resp = client.post("/chat", json={"message": "Hello, who are you?", "use_rag": False, "allow_tools": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "This is a mocked answer."
    assert data["provider_used"] == "openai"
    assert "session_id" in data


def test_ingest_endpoint(client):
    file_content = b"Sample knowledge base content about unit testing."
    resp = client.post("/ingest", files={"file": ("notes.txt", file_content)})
    assert resp.status_code == 200
    data = resp.json()
    assert data["filename"] == "notes.txt"
    assert data["num_chunks"] >= 1


def test_ingest_rejects_unsupported_extension(client):
    resp = client.post("/ingest", files={"file": ("image.png", b"binarydata")})
    assert resp.status_code == 400
