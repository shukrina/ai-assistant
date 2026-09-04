"""
CRUD helper functions isolating all direct DB access from route handlers.
"""
from typing import Optional, List
from sqlalchemy.orm import Session
from app import models


def get_or_create_session(db: Session, session_id: Optional[str]) -> models.ChatSession:
    if session_id:
        existing = db.get(models.ChatSession, session_id)
        if existing:
            return existing
    new_session = models.ChatSession()
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return new_session


def add_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    provider_used: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> models.ChatMessage:
    msg = models.ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        provider_used=provider_used,
        latency_ms=latency_ms,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def get_history(db: Session, session_id: str, limit: int = 20) -> List[models.ChatMessage]:
    return (
        db.query(models.ChatMessage)
        .filter(models.ChatMessage.session_id == session_id)
        .order_by(models.ChatMessage.created_at.asc())
        .limit(limit)
        .all()
    )


def record_document(db: Session, filename: str, num_chunks: int) -> models.Document:
    doc = models.Document(filename=filename, num_chunks=num_chunks)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc
