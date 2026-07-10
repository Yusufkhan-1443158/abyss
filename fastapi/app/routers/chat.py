"""LLM assistant router — basic Claude-backed chat (POST /api/chat).

Phase 1 (now): a plain conversational assistant powered by the Anthropic API.
Phase 2 (later): ground responses on the platform's own data — surveys, depths,
reports — so the assistant answers from the DB.

Auth-guarded like every other data route. The Anthropic key is operator-supplied
(config.anthropic_api_key: env var → /run/secrets/anthropic_api_key). When it's
absent the endpoint returns a friendly "not configured" reply instead of
erroring, so the dormant→live flip is graceful. The endpoint never 500s the chat
UI — model/SDK errors come back as a readable message.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..config import get_settings
from ..middleware.auth_middleware import get_current_user
from ..models import User

router = APIRouter()
settings = get_settings()
_log = logging.getLogger("abyss.chat")

SYSTEM_PROMPT = (
    "You are the Abyss assistant, an AI helper inside Abyss by Orbion — a "
    "bathymetric-intelligence platform where analysts run satellite-derived depth "
    "analyses over ocean areas (the Studio), store the imagery they ingest "
    "(Collections), and generate reports on those areas (Reports). Be concise, "
    "helpful, and professional. You can answer general questions and explain how "
    "the platform works. You do not yet have live access to the user's surveys, "
    "depths, or reports — if asked for specific data values, say data-grounded "
    "answers are coming soon and point them to the Studio, Collections, or "
    "Reports apps."
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    reply: str


@lru_cache(maxsize=1)
def _client(api_key: str):
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Single-turn (optionally history-aware) chat with the Abyss assistant."""
    key = settings.anthropic_api_key
    if not key:
        return ChatResponse(reply=(
            "The Abyss assistant isn’t configured yet — an administrator needs to "
            "add an Anthropic API key. It’s coming soon."
        ))

    messages = [{"role": m.role, "content": m.content} for m in body.history[-20:]]
    messages.append({"role": "user", "content": body.message})

    try:
        resp = _client(key).messages.create(
            model=settings.CHAT_MODEL,
            max_tokens=settings.CHAT_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        text = "".join(
            block.text for block in resp.content
            if getattr(block, "type", None) == "text"
        )
        return ChatResponse(reply=text.strip() or "(no response)")
    except Exception as exc:  # noqa: BLE001 — never surface a 500 to the chat UI
        _log.warning("chat error: %s", exc)
        return ChatResponse(reply="The assistant is unavailable right now. Please try again.")
