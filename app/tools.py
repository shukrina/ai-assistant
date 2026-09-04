"""
Tool ("function calling") definitions available to the LLM.

Each tool has:
  - an OpenAI-style JSON schema (used for the API's `tools` parameter)
  - a Python implementation registered in TOOL_REGISTRY

Add new tools by writing a function + schema, then registering both below.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict

from app.logging_config import get_logger

logger = get_logger(__name__)


def get_current_time(timezone: str = "UTC") -> Dict[str, Any]:
    """Return the current date/time. Only UTC is truly accurate here; treat
    other timezone labels as illustrative unless you wire in `zoneinfo`."""
    now = dt.datetime.now(dt.timezone.utc)
    return {"timezone": timezone, "current_time_utc": now.isoformat()}


def calculator(expression: str) -> Dict[str, Any]:
    """Safely evaluate a basic arithmetic expression (no builtins, no names)."""
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return {"error": "Expression contains disallowed characters."}
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 (sandboxed)
        return {"expression": expression, "result": result}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not evaluate expression: {exc}"}


def search_knowledge_base(query: str, top_k: int = 4) -> Dict[str, Any]:
    """Explicit tool wrapper around the RAG retriever, for models that prefer
    to call retrieval as a tool rather than relying on injected context."""
    from app.rag import retrieve  # local import avoids circulars at module load

    sources = retrieve(query, top_k=top_k)
    return {"results": [s.model_dump() for s in sources]}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "description": "IANA timezone name, e.g. 'UTC'"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression, e.g. '(12 + 8) / 4'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the ingested document knowledge base for relevant passages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 4},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_REGISTRY = {
    "get_current_time": get_current_time,
    "calculator": calculator,
    "search_knowledge_base": search_knowledge_base,
}


def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return fn(**arguments)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Tool '%s' raised an exception", name)
        return {"error": str(exc)}
