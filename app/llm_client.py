"""
Unified LLM client.

Supports OpenAI, Anthropic, Gemini, and a locally-served OpenAI-compatible
model (vLLM) behind one interface. Implements:
  - retries with exponential backoff (tenacity)
  - automatic fallback to a secondary provider on repeated failure
  - a simple token-bucket rate limiter
  - tool-calling normalized to a single (name, arguments) result shape
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.config import get_settings
from app.logging_config import get_logger
from app.tools import TOOL_SCHEMAS

settings = get_settings()
logger = get_logger(__name__)


class LLMError(Exception):
    """Raised when a provider call fails after all retries."""


class AllProvidersFailedError(Exception):
    """Raised when both primary and fallback providers fail."""


# --------------------------------------------------------------------------
# Rate limiter (token bucket, thread-safe, per-process)
# --------------------------------------------------------------------------
class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.capacity = requests_per_minute
        self.tokens = requests_per_minute
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


rate_limiter = RateLimiter(settings.RATE_LIMIT_PER_MINUTE)


class RateLimitExceededError(Exception):
    pass


@dataclass
class LLMResult:
    text: str
    provider: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw_tool_messages: List[Dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Provider adapters — each returns a raw (message) dict normalized below
# --------------------------------------------------------------------------
def _call_openai_compatible(
    base_url: Optional[str],
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    allow_tools: bool,
) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=settings.REQUEST_TIMEOUT_SECONDS)
    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=settings.TEMPERATURE,
        top_p=settings.TOP_P,
        max_tokens=settings.MAX_TOKENS,
    )
    if allow_tools:
        kwargs["tools"] = TOOL_SCHEMAS
        kwargs["tool_choice"] = "auto"

    completion = client.chat.completions.create(**kwargs)
    choice = completion.choices[0].message
    return {
        "content": choice.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments or "{}"),
            }
            for tc in (choice.tool_calls or [])
        ],
    }


def _call_anthropic(messages: List[Dict[str, Any]], allow_tools: bool) -> Dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY, timeout=settings.REQUEST_TIMEOUT_SECONDS)

    system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)
    convo = [m for m in messages if m["role"] != "system"]

    kwargs: Dict[str, Any] = dict(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=settings.MAX_TOKENS,
        temperature=settings.TEMPERATURE,
        top_p=settings.TOP_P,
        system=system_msg,
        messages=convo,
    )
    if allow_tools:
        kwargs["tools"] = [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "input_schema": t["function"]["parameters"],
            }
            for t in TOOL_SCHEMAS
        ]

    response = client.messages.create(**kwargs)

    text_parts = [b.text for b in response.content if b.type == "text"]
    tool_calls = [
        {"id": b.id, "name": b.name, "arguments": b.input}
        for b in response.content
        if b.type == "tool_use"
    ]
    return {"content": "\n".join(text_parts), "tool_calls": tool_calls}


_GEMINI_UNSUPPORTED_SCHEMA_KEYS = {"default", "additionalProperties", "$schema", "title", "examples"}


def _sanitize_schema_for_gemini(schema: Any) -> Any:
    """Recursively strip JSON-Schema fields that Gemini's function-declaration
    schema (a stricter OpenAPI subset) rejects, e.g. 'default'. Without this,
    Gemini raises 'Unknown field for Schema: default' and the whole call fails."""
    if isinstance(schema, dict):
        return {
            k: _sanitize_schema_for_gemini(v)
            for k, v in schema.items()
            if k not in _GEMINI_UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_sanitize_schema_for_gemini(item) for item in schema]
    return schema


def _tool_schemas_to_gemini(tool_schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert our OpenAI-style tool schemas into Gemini's `tools` format
    (a list with one entry containing `function_declarations`)."""
    declarations = []
    for t in tool_schemas:
        fn = t["function"]
        declarations.append(
            {
                "name": fn["name"],
                "description": fn["description"],
                "parameters": _sanitize_schema_for_gemini(fn["parameters"]),
            }
        )
    return [{"function_declarations": declarations}]


def _call_gemini(messages: List[Dict[str, Any]], allow_tools: bool) -> Dict[str, Any]:
    import google.generativeai as genai

    genai.configure(api_key=settings.GEMINI_API_KEY)
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), None)

    model_kwargs: Dict[str, Any] = {"system_instruction": system_msg}
    if allow_tools:
        model_kwargs["tools"] = _tool_schemas_to_gemini(TOOL_SCHEMAS)

    model = genai.GenerativeModel(settings.GEMINI_MODEL, **model_kwargs)

    history = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in messages
        if m["role"] in ("user", "assistant")
    ]
    convo = model.start_chat(history=history[:-1] if history else [])
    response = convo.send_message(
        history[-1]["parts"][0] if history else "",
        generation_config={
            "temperature": settings.TEMPERATURE,
            "top_p": settings.TOP_P,
            "max_output_tokens": settings.MAX_TOKENS,
        },
    )

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    candidate = response.candidates[0] if response.candidates else None
    parts = candidate.content.parts if candidate and candidate.content else []

    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc and getattr(fc, "name", None):
            tool_calls.append(
                {
                    # Gemini doesn't issue call ids; the name is unique enough
                    # for our single-pass execute-then-answer loop.
                    "id": fc.name,
                    "name": fc.name,
                    "arguments": dict(fc.args) if fc.args else {},
                }
            )
        elif getattr(part, "text", None):
            text_parts.append(part.text)

    return {"content": "\n".join(text_parts), "tool_calls": tool_calls}


def _dispatch(provider: str, messages: List[Dict[str, Any]], allow_tools: bool) -> Dict[str, Any]:
    if provider == "openai":
        return _call_openai_compatible(None, settings.OPENAI_API_KEY, settings.OPENAI_MODEL, messages, allow_tools)
    if provider == "anthropic":
        return _call_anthropic(messages, allow_tools)
    if provider == "gemini":
        return _call_gemini(messages, allow_tools)
    if provider == "local":
        return _call_openai_compatible(
            settings.LOCAL_LLM_BASE_URL, "sk-local", settings.LOCAL_LLM_MODEL, messages, allow_tools
        )
    raise LLMError(f"Unsupported provider: {provider}")


@retry(
    reraise=True,
    stop=stop_after_attempt(get_settings().MAX_RETRIES),
    wait=wait_exponential(multiplier=get_settings().RETRY_BACKOFF_SECONDS, min=1, max=10),
    retry=retry_if_exception_type(Exception),
)
def _call_with_retry(provider: str, messages: List[Dict[str, Any]], allow_tools: bool) -> Dict[str, Any]:
    logger.info("Calling LLM provider=%s", provider)
    return _dispatch(provider, messages, allow_tools)


def call_llm(messages: List[Dict[str, Any]], allow_tools: bool = True) -> LLMResult:
    """
    Call the primary provider; on failure (after retries) transparently fall
    back to the secondary provider. Raises AllProvidersFailedError only if
    both fail.
    """
    if not rate_limiter.allow():
        raise RateLimitExceededError("Rate limit exceeded, please slow down.")

    primary = settings.LLM_PROVIDER
    fallback = settings.FALLBACK_LLM_PROVIDER

    try:
        raw = _call_with_retry(primary, messages, allow_tools)
        return LLMResult(text=raw["content"], provider=primary, tool_calls=raw["tool_calls"])
    except Exception as primary_exc:  # noqa: BLE001
        logger.warning("Primary provider '%s' failed: %s. Falling back to '%s'.", primary, primary_exc, fallback)
        try:
            raw = _call_with_retry(fallback, messages, allow_tools)
            return LLMResult(text=raw["content"], provider=fallback, tool_calls=raw["tool_calls"])
        except Exception as fallback_exc:  # noqa: BLE001
            logger.error("Fallback provider '%s' also failed: %s", fallback, fallback_exc)
            raise AllProvidersFailedError(
                f"Both '{primary}' and '{fallback}' providers failed. "
                f"primary_error={primary_exc}; fallback_error={fallback_exc}"
            ) from fallback_exc
