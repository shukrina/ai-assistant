"""
Pydantic schemas for request validation and structured (JSON) responses.
"""
from typing import Optional, List, Literal, Any, Dict
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Existing session id, or omit to start a new one")
    message: str = Field(..., min_length=1, max_length=4000)
    use_rag: bool = Field(True, description="Whether to retrieve context from the vector store")
    allow_tools: bool = Field(True, description="Whether the model may call tools")


class SourceChunk(BaseModel):
    document: str
    chunk_id: str
    text: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    provider_used: str
    used_fallback: bool = False
    sources: List[SourceChunk] = []
    tool_calls: List[Dict[str, Any]] = []
    latency_ms: float
    cached: bool = False


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    vector_store: bool
    primary_llm: bool
    fallback_llm: bool


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
