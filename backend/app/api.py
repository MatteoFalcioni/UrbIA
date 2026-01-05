from __future__ import annotations

from typing import Optional, Any
from datetime import datetime, timezone
from uuid import UUID

import logging
import uuid
import json
import time
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, case
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.session import get_session
from backend.db.models import Thread, Message, Config, UserAPIKeys
from backend.utils.encryption import encrypt_api_key, decrypt_api_key, mask_api_key


# API router for thread- and message-related endpoints, mounted under /api
router = APIRouter()


async def get_user_api_keys_for_llm(user_id: str, session: AsyncSession) -> dict | None:
    """
    Get user's API keys for LLM usage (raw, unencrypted).
    Returns None if no keys are found.
    """
    try:
        result = await session.execute(
            select(UserAPIKeys).where(UserAPIKeys.user_id == user_id)
        )
        user_keys = result.scalar_one_or_none()

        if not user_keys:
            return None

        keys = {}
        if user_keys.openai_key:
            keys["openai_key"] = decrypt_api_key(user_keys.openai_key)
        if user_keys.anthropic_key:
            keys["anthropic_key"] = decrypt_api_key(user_keys.anthropic_key)

        return keys if keys else None
    except Exception:
        return None


# Request schema for creating a new thread
class ThreadCreate(BaseModel):
    # Caller-provided user id. In production, this would come from auth.
    user_id: str
    # Optional title; can be auto-titled later from first assistant reply.
    title: str = "New chat"


# Response schema returned for thread resources
class ThreadOut(BaseModel):
    id: UUID
    user_id: str
    title: Optional[str]
    archived_at: Optional[datetime] = None

    class Config:
        # Allow constructing from ORM models
        from_attributes = True


def to_jsonable(value: Any) -> Any:
    """
    Best-effort conversion of arbitrary objects (LangChain/LangGraph types, Pydantic models, etc.)
    into JSON-serializable primitives for DB storage and SSE.
    """
    try:
        json.dumps(value)
        return value
    except Exception:
        pass

    # Handle ToolRuntime - don't serialize it, return a placeholder
    try:
        if hasattr(value, "__class__") and "ToolRuntime" in str(value.__class__):
            return {"type": "ToolRuntime", "serialized": False}
    except Exception:
        pass

    # Handle dict with ToolRuntime values
    if isinstance(value, dict):
        filtered_dict = {}
        for k, v in value.items():
            if hasattr(v, "__class__") and "ToolRuntime" in str(v.__class__):
                filtered_dict[k] = {"type": "ToolRuntime", "serialized": False}
            else:
                filtered_dict[k] = to_jsonable(v)
        return filtered_dict

    # Pydantic v2
    try:
        if hasattr(value, "model_dump"):
            return value.model_dump()
    except Exception:
        pass

    # Pydantic v1 / dataclass-like
    try:
        if hasattr(value, "dict"):
            return value.dict()
    except Exception:
        pass

    # LangChain message chunks often have .content
    try:
        if hasattr(value, "content"):
            return getattr(value, "content")
    except Exception:
        pass

    # Bytes → utf-8 string
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="ignore")

    # Fallback string representation
    try:
        return str(value)
    except Exception:
        return None


def extract_text_from_content(content: Any) -> str:
    """
    Extract text from message content, handling both dict and list formats.
    - Dict format: {'text': '...'} → returns the text
    - List format: [{'text': '...', 'type': 'text'}] → returns text from all text blocks (ignores tool_use)
    - String: returns as-is
    - Other: returns string representation
    """
    if not content:
        return ""

    # Dict format (OpenAI old style or normalized)
    if isinstance(content, dict):
        return content.get("text", str(content))

    # List format (Claude/new models)
    if isinstance(content, list) and len(content) > 0:
        # Filter and extract only text blocks, ignore tool_use blocks
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # Only process text blocks, skip tool_use and other types
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(item["text"])
                # Legacy format without explicit type
                elif "text" in item and "type" not in item:
                    text_parts.append(item["text"])

        if text_parts:
            return "".join(text_parts)

        # Fallback: if no text blocks found, return empty string (don't show tool use JSON)
        return ""

    # String or other
    return str(content)


@router.post("/threads", response_model=ThreadOut)
async def create_thread(
    payload: ThreadCreate, session: AsyncSession = Depends(get_session)
) -> ThreadOut:
    """
    Create a new thread row for the given user.
    Returns the created thread in a stable response shape for the frontend.
    """
    thread = Thread(user_id=payload.user_id, title=payload.title)
    session.add(thread)
    await session.commit()
    # Refresh to ensure we return DB-generated values (e.g., UUID)
    await session.refresh(thread)
    return ThreadOut.model_validate(thread)


@router.get("/threads", response_model=list[ThreadOut])
async def list_threads(
    user_id: str = Query(..., description="Scope threads by user"),
    limit: int = Query(20, ge=1, le=100),
    include_archived: bool = Query(False, description="Include archived threads"),
    session: AsyncSession = Depends(get_session),
) -> list[ThreadOut]:
    """
    List recent threads for a user, ordered by last update.
    Limit is capped to avoid excessive payloads.
    """
    conditions = [Thread.user_id == user_id]
    if not include_archived:
        conditions.append(Thread.archived_at.is_(None))

    stmt = (
        select(Thread)
        .where(*conditions)
        .order_by(Thread.updated_at.desc())
        .limit(limit)
    )
    res = await session.execute(stmt)
    rows = res.scalars().all()
    return [ThreadOut.model_validate(r) for r in rows]


@router.get("/threads/{thread_id}", response_model=ThreadOut)
async def get_thread(
    thread_id: str,
    session: AsyncSession = Depends(get_session),
) -> ThreadOut:
    """
    Get thread metadata by ID.
    """
    t = await session.get(Thread, thread_id)
    if not t:
        raise HTTPException(status_code=404, detail="Thread not found")
    return ThreadOut.model_validate(t)


@router.post("/threads/{thread_id}/archive", response_model=ThreadOut)
async def archive_thread(
    thread_id: str,
    session: AsyncSession = Depends(get_session),
) -> ThreadOut:
    """
    Soft-delete (archive) a thread. Archived threads are hidden from list by default.
    """
    t = await session.get(Thread, thread_id)
    if not t:
        raise HTTPException(status_code=404, detail="Thread not found")
    if t.archived_at is None:
        t.archived_at = datetime.now(timezone.utc)
        await session.commit()
        await session.refresh(t)
    return ThreadOut.model_validate(t)


@router.post("/threads/{thread_id}/unarchive", response_model=ThreadOut)
async def unarchive_thread(
    thread_id: str,
    session: AsyncSession = Depends(get_session),
) -> ThreadOut:
    """
    Un-archive a thread, making it visible in the default list again.
    """
    t = await session.get(Thread, thread_id)
    if not t:
        raise HTTPException(status_code=404, detail="Thread not found")
    if t.archived_at is not None:
        t.archived_at = None
        await session.commit()
        await session.refresh(t)
    return ThreadOut.model_validate(t)


@router.delete("/threads/{thread_id}", status_code=204, response_class=Response)
async def delete_thread(
    thread_id: str,
    session: AsyncSession = Depends(get_session),
) -> Response:
    """
    Hard-delete a thread and all related rows (messages, configs, artifacts) via cascades.
    Best-effort: acquire per-thread lock to avoid concurrent runs.
    """
    from backend.main import get_thread_lock

    lock = get_thread_lock(str(thread_id))
    async with lock:
        t = await session.get(Thread, thread_id)
        if not t:
            # Idempotent: deleting a non-existent thread returns 204
            return Response(status_code=204)

        await session.delete(t)
        await session.commit()

        # Best-effort: we could also remove LangGraph checkpoints here if API allows.
        # Skipped to keep implementation simple.

    return Response(status_code=204)


class ThreadTitleUpdate(BaseModel):
    title: str


@router.patch("/threads/{thread_id}/title", response_model=ThreadOut)
async def update_thread_title(
    thread_id: str,
    payload: ThreadTitleUpdate,
    session: AsyncSession = Depends(get_session),
) -> ThreadOut:
    """
    Update thread title manually.
    """
    t = await session.get(Thread, thread_id)
    if not t:
        raise HTTPException(status_code=404, detail="Thread not found")
    t.title = payload.title
    await session.commit()
    await session.refresh(t)
    return ThreadOut.model_validate(t)


@router.post("/threads/{thread_id}/title/auto", response_model=ThreadOut)
async def llm_update_thread_title(
    thread_id: str,
    session: AsyncSession = Depends(get_session),
) -> ThreadOut:
    """
    Auto-generate thread title using LLM based on conversation so far.
    """
    t = await session.get(Thread, thread_id)
    if not t:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Fetch recent messages (max last 6 for context)
    stmt = (
        select(Message)
        .where(Message.thread_id == t.id)
        .order_by(Message.created_at.asc())
        .limit(6)
    )
    res = await session.execute(stmt)
    messages = res.scalars().all()

    # Build thread text from messages
    thread_text = "\n".join(
        [
            f"{m.role}: {m.content.get('text', str(m.content)) if m.content else ''}"
            for m in messages
        ]
    )

    if not thread_text.strip():
        raise HTTPException(
            status_code=400, detail="No messages to generate title from"
        )

    # Generate title with LLM
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from pydantic import SecretStr
    import os

    # Get user API keys
    user_api_keys = await get_user_api_keys_for_llm(t.user_id, session)

    # Configure LLM with user's API key if available
    llm_kwargs = {"model": "gpt-4o-mini", "temperature": 0}
    if user_api_keys and user_api_keys.get("openai_key"):
        llm_kwargs["api_key"] = SecretStr(user_api_keys["openai_key"])
    elif os.getenv("OPENAI_API_KEY"):
        llm_kwargs["api_key"] = SecretStr(os.getenv("OPENAI_API_KEY"))

    llm = ChatOpenAI(**llm_kwargs)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Return a concise, engaging title. No quotes, <= 8 words."),
            ("user", "Text:\n{body}\n\nTitle:"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    t.title = chain.invoke({"body": thread_text})
    await session.commit()
    await session.refresh(t)
    return ThreadOut.model_validate(t)


# Config schemas
class ConfigOut(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    context_window: Optional[int] = None
    settings: Optional[dict] = None

    class Config:
        from_attributes = True


class ConfigUpdate(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    context_window: Optional[int] = None
    settings: Optional[dict] = None


@router.get("/config/defaults", response_model=ConfigOut)
async def get_default_config() -> ConfigOut:
    """
    Get default config from environment variables (no thread required).
    """
    from backend.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE, CONTEXT_WINDOW

    return ConfigOut(
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        context_window=CONTEXT_WINDOW,
        settings=None,
    )


@router.get("/threads/{thread_id}/config", response_model=ConfigOut)
async def get_thread_config(
    thread_id: str,
    session: AsyncSession = Depends(get_session),
) -> ConfigOut:
    """
    Get thread-specific config (model, temperature, system_prompt).
    Returns defaults if no config exists.
    """
    from backend.config import DEFAULT_MODEL, DEFAULT_TEMPERATURE
    from sqlalchemy import select

    try:
        thread_uuid = UUID(thread_id)
    except ValueError:
        logging.warning(f"Invalid thread_id format in get_thread_config: {thread_id}")
        raise HTTPException(status_code=400, detail="Invalid thread_id format")

    cfg_result = await session.execute(
        select(Config).where(Config.thread_id == thread_uuid)
    )
    cfg = cfg_result.scalar_one_or_none()
    if not cfg:
        # Return env-based defaults
        from backend.config import CONTEXT_WINDOW

        return ConfigOut(
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            context_window=CONTEXT_WINDOW,
            settings=None,
        )
    return ConfigOut.model_validate(cfg)


@router.post("/threads/{thread_id}/config", response_model=ConfigOut)
async def update_thread_config(
    thread_id: str,
    payload: ConfigUpdate,
    session: AsyncSession = Depends(get_session),
) -> ConfigOut:
    """
    Update thread config (upsert). Frontend can set model, temperature, system_prompt per thread.
    """
    from sqlalchemy import select

    try:
        thread_uuid = UUID(thread_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid thread_id format")

    result = await session.execute(select(Thread).where(Thread.id == thread_uuid))
    t = result.scalar_one_or_none()
    if not t:
        raise HTTPException(status_code=404, detail="Thread not found")

    cfg_result = await session.execute(
        select(Config).where(Config.thread_id == thread_uuid)
    )
    cfg = cfg_result.scalar_one_or_none()
    if not cfg:
        cfg = Config(thread_id=t.id)
        session.add(cfg)

    # Update fields if provided
    if payload.model is not None:
        cfg.model = payload.model
        logging.info(f"Updated thread {thread_id} model to: {payload.model}")
    if payload.temperature is not None:
        cfg.temperature = payload.temperature
    if payload.system_prompt is not None:
        cfg.system_prompt = payload.system_prompt
    if payload.context_window is not None:
        cfg.context_window = payload.context_window
    if payload.settings is not None:
        cfg.settings = payload.settings

    await session.commit()
    await session.refresh(cfg)
    logging.info(
        f"Thread {thread_id} config saved - model: {cfg.model}, temperature: {cfg.temperature}"
    )
    return ConfigOut.model_validate(cfg)


# Response schema for messages
class ArtifactOut(BaseModel):
    id: UUID
    name: str
    mime: str
    size: int
    url: str

    class Config:
        from_attributes = True


class MessageOut(BaseModel):
    id: UUID
    thread_id: UUID
    message_id: Optional[str] = (
        None  # Client-supplied idempotency key, also used to link segments to user messages
    )
    role: str
    content: Optional[dict] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[dict] = None
    meta: Optional[dict] = None  # For storing agent name in subagent messages
    artifacts: list[ArtifactOut] = []
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


@router.get("/threads/{thread_id}/state")
async def get_thread_state(
    thread_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Get the current LangGraph state for a thread, including token count.
    This allows the frontend to display context usage without sending a message.
    """
    from backend.graph.graph import make_graph
    from backend.main import _checkpointer_cm
    from backend.config import CONTEXT_WINDOW as DEFAULT_CONTEXT_WINDOW

    # Ensure thread exists
    t = await session.get(Thread, thread_id)
    if not t:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Read thread config to get context_window
    cfg = await session.get(Config, thread_id)

    if not _checkpointer_cm:
        raise HTTPException(status_code=500, detail="Checkpointer not initialized")

    # Get user API keys for LLM usage
    user_api_keys = await get_user_api_keys_for_llm(t.user_id, session)

    # Create graph with thread-specific config and user API keys
    graph = make_graph(
        model_name=cfg.model if cfg else None,
        temperature=cfg.temperature if cfg else None,
        system_prompt=cfg.system_prompt if cfg else None,
        context_window=cfg.context_window if cfg else None,
        checkpointer=_checkpointer_cm[0],
        user_api_keys=user_api_keys,
    )

    config = {"configurable": {"thread_id": str(thread_id)}}

    try:
        state_snapshot = await graph.aget_state(config)
        token_count = (
            state_snapshot.values.get("token_count", 0) if state_snapshot.values else 0
        )
        todos = state_snapshot.values.get("todos", []) if state_snapshot.values else []
        reports = (
            state_snapshot.values.get("reports", {}) if state_snapshot.values else {}
        )
        last_report_title = (
            state_snapshot.values.get("last_report_title", "")
            if state_snapshot.values
            else ""
        )
        code_logs = (
            state_snapshot.values.get("code_logs", []) if state_snapshot.values else []
        )
        final_score = (
            state_snapshot.values.get("final_score") if state_snapshot.values else None
        )
        analysis_status = (
            state_snapshot.values.get("analysis_status") if state_snapshot.values else None
        )
        context_window = (
            cfg.context_window
            if (cfg and cfg.context_window)
            else DEFAULT_CONTEXT_WINDOW
        )

        # Get the current report content if available
        current_report_content = (
            reports.get(last_report_title, "") if last_report_title else ""
        )

        return {
            "token_count": token_count,
            "context_window": context_window,
            "todos": todos,
            "report_title": last_report_title,
            "report_content": current_report_content,
            "reports": reports,
            "code_logs": code_logs,
            "final_score": final_score,
            "analysis_status": analysis_status,
        }
    except Exception as e:
        # If state doesn't exist yet (new thread), return defaults
        logging.debug(
            f"Thread state not found for {thread_id}, returning defaults: {e}"
        )
        context_window = (
            cfg.context_window
            if (cfg and cfg.context_window)
            else DEFAULT_CONTEXT_WINDOW
        )
        return {
            "token_count": 0,
            "context_window": context_window,
            "todos": [],
            "report_title": "",
            "report_content": "",
            "reports": {},
            "code_logs": [],
            "final_score": None,
            "analysis_status": None,
        }


@router.get("/threads/{thread_id}/messages", response_model=list[MessageOut])
async def list_messages(
    thread_id: str,
    limit: int = Query(50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
) -> list[MessageOut]:
    """
    List recent messages for a thread (reverse chronological by created_at).
    Only finalized messages are stored and returned (no partial tokens).
    Includes artifacts associated with each message via tool_call_id.
    """
    from backend.db.models import Artifact
    from backend.artifacts.storage import generate_artifact_download_url

    stmt = (
        select(Message)
        .where(Message.thread_id == thread_id)
        .order_by(
            Message.created_at.desc(),
            # Custom ordering: tool messages first, then subagent segments, then supervisor
            # This ensures correct chronological order even when timestamps are identical
            case(
                (Message.message_id.like("tool:%"), 0),
                (Message.message_id.like("subagent:%"), 1),
                (Message.message_id.like("assistant:%"), 2),
                else_=3,
            ).asc(),
            Message.id.desc(),  # Final tie-breaker for deterministic ordering
        )
        .limit(limit)
    )
    res = await session.execute(stmt)
    messages = res.scalars().all()

    # Build message outputs with artifacts
    message_outputs = []
    for msg in messages:
        # Normalize content: Claude/newer models return list, but frontend expects dict
        content = msg.content
        if isinstance(content, list) and len(content) > 0:
            # Extract text from list format: [{'text': '...', 'type': 'text'}]
            if isinstance(content[0], dict) and "text" in content[0]:
                content = {"text": content[0]["text"]}

        msg_dict = {
            "id": msg.id,
            "thread_id": msg.thread_id,
            "message_id": msg.message_id,  # Include message_id for frontend to link segments to user messages
            "role": msg.role,
            "content": content,
            "tool_name": msg.tool_name,
            "tool_input": msg.tool_input,
            "tool_output": msg.tool_output,
            "meta": msg.meta,  # Include meta field (contains agent name for subagent messages)
            "artifacts": [],
        }

        # For assistant messages, look for artifacts from tool calls in the same conversational turn
        # A "turn" is defined as all tool executions that happened immediately before this assistant response
        if msg.role == "assistant":
            # Find the most recent USER message before this assistant message
            # This gives us the start of the current conversational turn
            prev_user_stmt = (
                select(Message.created_at)
                .where(Message.thread_id == thread_id)
                .where(Message.created_at < msg.created_at)
                .where(Message.role == "user")
                .order_by(Message.created_at.desc())
                .limit(1)
            )
            prev_user_res = await session.execute(prev_user_stmt)
            turn_start_time = prev_user_res.scalar_one_or_none()

            # Query for tool messages in this conversational turn (after the user message, up to and including this assistant message)
            tool_msgs_stmt = (
                select(Message)
                .where(Message.thread_id == thread_id)
                .where(Message.role == "tool")
                .where(Message.created_at <= msg.created_at)
            )

            if turn_start_time:
                # Only get tool messages after the user message that started this turn
                tool_msgs_stmt = tool_msgs_stmt.where(
                    Message.created_at > turn_start_time
                )

            tool_msgs_stmt = tool_msgs_stmt.order_by(Message.created_at.desc())
            tool_msgs_res = await session.execute(tool_msgs_stmt)
            tool_messages = tool_msgs_res.scalars().all()

            # Collect artifacts from all tool calls in this turn
            for tool_msg in tool_messages:
                # Extract tool_call_id from tool_input (where it's actually stored)
                tool_call_id = None
                if isinstance(tool_msg.tool_input, dict):
                    tool_call_id = tool_msg.tool_input.get("tool_call_id")

                # Fallback: check tool_output
                if not tool_call_id and isinstance(tool_msg.tool_output, dict):
                    tool_call_id = tool_msg.tool_output.get("tool_call_id")

                # Also check meta field
                if not tool_call_id and tool_msg.meta:
                    tool_call_id = tool_msg.meta.get("tool_call_id")

                if tool_call_id:
                    # Query artifacts for this tool call
                    artifact_stmt = select(Artifact).where(
                        Artifact.tool_call_id == tool_call_id
                    )
                    artifact_res = await session.execute(artifact_stmt)
                    artifacts = artifact_res.scalars().all()

                    # Build artifact outputs with download URLs (S3 presigned)
                    for artifact in artifacts:
                        try:
                            url = generate_artifact_download_url(
                                artifact, expiry_seconds=86400
                            )
                            msg_dict["artifacts"].append(
                                {
                                    "id": artifact.id,
                                    "name": artifact.filename,
                                    "mime": artifact.mime,
                                    "size": artifact.size,
                                    "url": url,
                                }
                            )
                        except Exception as e:
                            logging.warning(
                                f"Failed to generate URL for artifact {artifact.id}: {e}"
                            )
                            pass  # Skip artifacts that fail URL generation

        message_outputs.append(MessageOut.model_validate(msg_dict))

    return message_outputs


class PostMessageIn(BaseModel):
    # Client idempotency key (required)
    message_id: str
    # Message body (text or structured blocks); keep minimal for stub
    content: dict
    # Role must be 'user' for this stub
    role: str = "user"


@router.post("/threads/{thread_id}/messages")
async def post_message_stream(
    request: Request,
    thread_id: str,
    payload: PostMessageIn,
    session: AsyncSession = Depends(get_session),
):
    """
    Accept user message, run LangGraph agent, stream tokens via SSE,
    and persist finalized assistant message at end.
    Enforces idempotency via unique message_id and per-thread locking.
    """
    if payload.role != "user":
        raise HTTPException(status_code=400, detail="Only user role allowed")

    # Ensure thread exists
    t = await session.get(Thread, thread_id)
    if not t:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Read thread config (model, temperature, system_prompt, context_window)
    cfg = await session.get(Config, thread_id)

    # Insert user message; unique constraint on message_id enforces idempotency
    msg = Message(
        thread_id=t.id,
        message_id=payload.message_id,
        role="user",
        content=payload.content,
    )
    session.add(msg)
    try:
        await session.commit()
    except Exception:
        await session.rollback()
        raise HTTPException(status_code=409, detail="Duplicate message_id")

    # Auto-title after first user message (async, best-effort)
    # This runs in background and won't block the response stream
    async def auto_title_from_first_message():
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from pydantic import SecretStr
            import os
            from backend.db.session import ASYNC_SESSION_MAKER

            async with ASYNC_SESSION_MAKER() as title_sess:
                thread_check = await title_sess.get(Thread, t.id)
                if not thread_check or thread_check.title != "New chat":
                    return  # Already titled or thread not found

                # Check if this is the first user message
                stmt = (
                    select(Message)
                    .where(Message.thread_id == thread_check.id)
                    .where(Message.role == "user")
                )
                res = await title_sess.execute(stmt)
                user_messages = res.scalars().all()

                if len(user_messages) != 1:
                    return  # Not the first message

                logging.info(f"Auto-titling thread {t.id} based on first user message")

                # Get the first user message content
                first_msg = user_messages[0]
                user_text = extract_text_from_content(first_msg.content)

                if not user_text.strip():
                    logging.warning("No content to generate title from")
                    return

                # Get user API keys
                user_api_keys = await get_user_api_keys_for_llm(t.user_id, title_sess)

                # Configure LLM with user's API key if available
                llm_kwargs = {"model": "gpt-4o-mini", "temperature": 0}
                if user_api_keys and user_api_keys.get("openai_key"):
                    llm_kwargs["api_key"] = SecretStr(user_api_keys["openai_key"])
                    logging.info("Using user's OpenAI API key for titling")
                elif os.getenv("OPENAI_API_KEY"):
                    llm_kwargs["api_key"] = SecretStr(os.getenv("OPENAI_API_KEY"))
                    logging.info("Using environment OpenAI API key for titling")
                else:
                    logging.warning("No OpenAI API key available for titling")
                    return

                llm = ChatOpenAI(**llm_kwargs)
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "Return a concise, engaging title based on the user's request. No quotes, <= 8 words.",
                        ),
                        ("user", "User message:\n{body}\n\nTitle:"),
                    ]
                )
                chain = prompt | llm | StrOutputParser()
                new_title = chain.invoke({"body": user_text})
                logging.info(f"Generated title: {new_title}")
                thread_check.title = new_title
                await title_sess.commit()

                return new_title
        except Exception as e:
            logging.error(f"Auto-title failed for thread {t.id}: {e}", exc_info=True)
            return None

    # Start auto-titling in background (don't await it)
    import asyncio

    title_task = asyncio.create_task(auto_title_from_first_message())

    # Stream LangGraph agent response via SSE
    async def event_stream():
        from backend.main import get_thread_lock
        from backend.graph.graph import make_graph
        from backend.main import _checkpointer_cm
        from backend.db.session import ASYNC_SESSION_MAKER
        from backend.graph.context import (
            set_db_session,
            set_thread_id,
            clear_db_session,
            clear_thread_id,
        )
        import uuid as uuid_module

        if not _checkpointer_cm:
            yield f"data: {json.dumps({'error': 'Checkpointer not initialized'})}\n\n"
            return

        # Wait for auto-titling to complete and emit event if successful
        try:
            new_title = await title_task
            if new_title:
                yield f"data: {json.dumps({'type': 'title_updated', 'title': new_title})}\n\n"
        except Exception as e:
            logging.error(f"Failed to get title from background task: {e}")

        # Get user API keys for LLM usage
        user_api_keys = await get_user_api_keys_for_llm(t.user_id, session)

        # Create graph with thread-specific config and user API keys
        graph = make_graph(
            model_name=cfg.model if cfg else None,
            temperature=cfg.temperature if cfg else None,
            system_prompt=cfg.system_prompt if cfg else None,
            context_window=(
                cfg.context_window if cfg else None
            ),  # Use thread config or env default
            checkpointer=_checkpointer_cm[0],  # Reuse global checkpointer
            user_api_keys=user_api_keys,
            plot_graph=False,
        )

        # Log model being used for this message
        from backend.config import DEFAULT_MODEL

        effective_model = cfg.model if cfg and cfg.model else DEFAULT_MODEL
        print(f"[MESSAGE START] Thread {thread_id} - Model: {effective_model}")

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        tool_calls = []  # Track tool calls for persistence

        try:
            lock = get_thread_lock(str(thread_id))
            async with lock:
                # Set context variables for tools to access database session and thread_id
                set_db_session(session)
                set_thread_id(uuid_module.UUID(str(thread_id)))

                # Extract text from content dict; LangChain messages expect string content
                user_text = payload.content.get("text", str(payload.content))
                state = {"messages": [{"role": "user", "content": user_text}]}
                config = {
                    "configurable": {"thread_id": str(thread_id)},
                    "recursion_limit": 150,
                }

                # Context update will be emitted after agent finishes processing

                # Stream events from LangGraph
                # We follow docs here: https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html?_gl=1*15ktatf*_gcl_au*MTc4MTgwMzA1Ny4xNzU4ODA2Mjcy*_ga*MTUzOTQwNjk3NS4xNzUwODY1MDM0*_ga_47WX3HKKY2*czE3NTk4MjY0Mzkkbzk5JGcxJHQxNzU5ODI2NTg0JGoxMyRsMCRoMA..#langchain_core.language_models.chat_models.BaseChatModel.astream_events

                # Variables to track content per agent
                current_agent_node = None  # Track which agent we're currently executing (supervisor, data_analyst, etc.)
                # Track streamed content for database storage - separate for each agent
                supervisor_content = ""  # Supervisor messages (main response)
                subagent_content = (
                    {}
                )  # Subagent messages: {"data_analyst": "", "report_writer": "", "reviewer": ""}
                subagent_order = (
                    []
                )  # Track execution order: ["data_analyst", "reviewer", ...]
                # Track segments for each agent (saved when tools start)
                subagent_segments = (
                    {}
                )  # {"data_analyst": [{"content": "...", "segment_index": 0}, ...]}
                subagent_segment_counters = {}  # Track segment index per agent

                # Track the last known agent node to handle cases where node becomes "model"/"tools"
                last_known_agent_node = None

                async for event in graph.astream_events(state, config, version="v2"):
                    event_type = event.get("event")
                    event_name = event.get("name", "")
                    event_meta = event.get("metadata", {})
                    node = event_meta.get("langgraph_node")
                    checkpoint_ns = event_meta.get("langgraph_checkpoint_ns", "")
                    event_meta.get("langgraph_step")

                    # Track which agent node we're currently in
                    # Important: node becomes "model"/"tools" during execution, so we need to track it
                    # when we see chain_start or chat_model_start events, OR extract from checkpoint_ns
                    if node and node not in ["model", "tools"]:
                        current_agent_node = node
                        last_known_agent_node = (
                            node  # Keep track of last known agent node
                        )
                        logging.debug(f"Agent node updated: {current_agent_node}")
                    elif node in ["model", "tools"] and checkpoint_ns:
                        # Extract agent name from checkpoint_ns (format: "data_analyst:uuid|model:uuid")
                        # The agent name is the first part before the first colon
                        checkpoint_parts = checkpoint_ns.split(":")
                        if checkpoint_parts and checkpoint_parts[0] in [
                            "supervisor",
                            "data_analyst",
                            "report_writer",
                            "reviewer",
                        ]:
                            current_agent_node = checkpoint_parts[0]
                            last_known_agent_node = checkpoint_parts[0]
                            logging.debug(
                                f"Agent node extracted from checkpoint_ns: {current_agent_node}"
                            )
                        elif last_known_agent_node:
                            # Fallback to last known agent node
                            current_agent_node = last_known_agent_node

                    # Also track agent node from chain_start events (when agent node starts)
                    if (
                        event_type == "on_chain_start"
                        and node
                        and node not in ["model", "tools"]
                    ):
                        current_agent_node = node
                        last_known_agent_node = node
                        logging.debug(
                            f"Agent node from chain_start: {current_agent_node}"
                        )

                    # Check for graph interrupts (for human-in-the-loop)
                    data = event.get("data", {})
                    chunk = data.get("chunk")

                    # Interrupt comes as a dict chunk with '__interrupt__' key
                    if chunk and isinstance(chunk, dict) and "__interrupt__" in chunk:
                        logging.info(
                            f"🛑 INTERRUPT DETECTED - chunk keys: {chunk.keys()}"
                        )
                        # Graph has been interrupted - send interrupt event to frontend
                        interrupt_tuple = chunk[
                            "__interrupt__"
                        ]  # It's a tuple: (Interrupt(...),)
                        interrupt_obj = interrupt_tuple[
                            0
                        ]  # Get the Interrupt object from tuple
                        interrupt_value = (
                            interrupt_obj.value
                        )  # Get the actual value dict
                        logging.info(f"Interrupt value: {interrupt_value}")
                        yield f"data: {json.dumps({'type': 'interrupt', 'value': to_jsonable(interrupt_value)})}\n\n"
                        # Stream will end naturally after interrupt, no break needed

                    # Stream token chunks from the LLM
                    if event_type == "on_chat_model_stream":
                        # Log for debugging
                        if not current_agent_node:
                            logging.warning(
                                f"on_chat_model_stream with no current_agent_node, node={node}, checkpoint_ns={checkpoint_ns}"
                            )

                        chunk = event.get("data", {}).get("chunk")
                        if chunk:
                            # Extract text from chunk content
                            if hasattr(chunk, "content") and chunk.content:
                                chunk_text = extract_text_from_content(chunk.content)
                                if chunk_text:
                                    # Stream based on which agent is active
                                    if current_agent_node == "supervisor":
                                        # Supervisor: normal streaming
                                        supervisor_content += chunk_text
                                        yield f"data: {json.dumps({'type': 'token', 'content': chunk_text})}\n\n"
                                    elif current_agent_node in [
                                        "data_analyst",
                                        "report_writer",
                                        "reviewer",
                                    ]:
                                        # Subagents: translucido streaming with dropdown
                                        agent_name = current_agent_node
                                        if agent_name not in subagent_content:
                                            subagent_content[agent_name] = ""
                                            # Track execution order: add agent to order list when first token arrives
                                            if agent_name not in subagent_order:
                                                subagent_order.append(agent_name)
                                        subagent_content[agent_name] += chunk_text
                                        logging.debug(
                                            f"Streaming subagent token: {agent_name}, content length: {len(chunk_text)}"
                                        )
                                        yield f"data: {json.dumps({'type': 'subagent_token', 'agent': agent_name, 'content': chunk_text})}\n\n"
                                    else:
                                        # Fallback: if we don't know the agent, log it
                                        # Skip SummarizationMiddleware - it's not an agent, just internal summarization
                                        if (
                                            current_agent_node
                                            != "SummarizationMiddleware.before_model"
                                        ):
                                            logging.warning(
                                                f"Unknown agent node during streaming: {current_agent_node}, node={node}"
                                            )

                    # Track agent node from chat_model_start events too (before streaming begins)
                    elif event_type == "on_chat_model_start":
                        # When chat_model_start fires, node might still be the agent node
                        # or we can infer it from the context
                        if node and node not in ["model", "tools"]:
                            current_agent_node = node
                            logging.debug(
                                f"Agent node from chat_model_start: {current_agent_node}"
                            )

                        if current_agent_node == "reviewer":
                            yield f"data: {json.dumps({'type': 'reviewing', 'status': 'start'})}\n\n"

                    # Detect reviewer end
                    elif (
                        event_type == "on_chat_model_end"
                        and current_agent_node == "reviewer"
                    ):
                        yield f"data: {json.dumps({'type': 'reviewing', 'status': 'done'})}\n\n"

                        # Get the final_score from state after reviewer completes
                        try:
                            state_snapshot = await graph.aget_state(config)
                            current_state = state_snapshot.values
                            final_score = current_state.get("final_score")
                            if final_score is not None:
                                yield f"data: {json.dumps({'type': 'score_updated', 'score': final_score})}\n\n"
                        except Exception as e:
                            logging.warning(
                                f"Failed to get final_score from state: {e}"
                            )

                    # Stream tool execution start
                    elif event_type == "on_tool_start":
                        tool_input = event.get("data", {}).get("input")
                        yield f"data: {json.dumps({'type': 'tool_start', 'name': event_name, 'input': to_jsonable(tool_input)})}\n\n"

                        # Save current segment for the active agent before tool starts
                        if current_agent_node and current_agent_node in [
                            "data_analyst",
                            "report_writer",
                            "reviewer",
                        ]:
                            agent_name = current_agent_node
                            if (
                                agent_name in subagent_content
                                and subagent_content[agent_name].strip()
                            ):
                                # Initialize segments list and counter if needed
                                if agent_name not in subagent_segments:
                                    subagent_segments[agent_name] = []
                                    subagent_segment_counters[agent_name] = 0

                                # Save current content as a segment
                                segment_content = subagent_content[agent_name]
                                segment_index = subagent_segment_counters[agent_name]
                                subagent_segments[agent_name].append(
                                    {
                                        "content": segment_content,
                                        "segment_index": segment_index,
                                    }
                                )
                                subagent_segment_counters[agent_name] += 1

                                # Reset content for next segment
                                subagent_content[agent_name] = ""
                                logging.debug(
                                    f"Saved segment {segment_index} for {agent_name}, content length: {len(segment_content)}"
                                )

                    # Stream tool execution end and capture for persistence
                    elif event_type == "on_tool_end":
                        raw_input = event.get("data", {}).get("input")
                        raw_output = event.get("data", {}).get("output")

                        # Emit objectives_updated event if set_analysis_objectives_tool was called
                        if event_name == "set_analysis_objectives_tool":
                            if hasattr(raw_output, "update") and isinstance(
                                raw_output.update, dict
                            ):
                                objectives = raw_output.update.get(
                                    "analysis_objectives", []
                                )
                                if objectives:
                                    yield f"data: {json.dumps({'type': 'objectives_updated', 'objectives': objectives})}\n\n"

                        # Emit report_written event if write_report_tool was called
                        if event_name == "write_report_tool":
                            if hasattr(raw_output, "update") and isinstance(
                                raw_output.update, dict
                            ):
                                reports = raw_output.update.get("reports", {})
                                report_title = raw_output.update.get(
                                    "last_report_title", ""
                                )
                                if reports and report_title and report_title in reports:
                                    report_content = reports[report_title]
                                    yield f"data: {json.dumps({'type': 'report_written', 'title': report_title, 'content': report_content})}\n\n"

                        # Emit todos_updated event if write_todos was called
                        if event_name == "write_todos":
                            # Output can be Command(update={'todos': ...})
                            if hasattr(raw_output, "update") and isinstance(
                                raw_output.update, dict
                            ):
                                todos = raw_output.update.get("todos", [])
                                if todos:
                                    yield f"data: {json.dumps({'type': 'todos_updated', 'todos': todos})}\n\n"

                        # Extract artifacts and content from Command -> ToolMessage if present
                        artifacts = None
                        tool_content = None

                        # Case 1: Output is a Command object (from code_sandbox tool)
                        if hasattr(raw_output, "update") and isinstance(
                            raw_output.update, dict
                        ):
                            messages = raw_output.update.get("messages", [])
                            if messages and len(messages) > 0:
                                tool_msg = messages[0]
                                # Extract artifacts
                                if hasattr(tool_msg, "artifact") and tool_msg.artifact:
                                    artifacts = tool_msg.artifact
                                # Extract content for database persistence
                                if hasattr(tool_msg, "content"):
                                    tool_content = tool_msg.content
                        # Case 2: Output is a ToolMessage directly
                        elif hasattr(raw_output, "artifact"):
                            artifacts = raw_output.artifact
                            if hasattr(raw_output, "content"):
                                tool_content = raw_output.content

                        # For database: store the tool content as dict (not the whole Command object)
                        if tool_content:
                            # If content is a string, wrap it in a dict
                            tool_output_for_db = (
                                {"content": tool_content}
                                if isinstance(tool_content, str)
                                else to_jsonable(tool_content)
                            )
                        else:
                            # Fallback to jsonable representation
                            tool_output_for_db = to_jsonable(raw_output)

                        # Extract tool_call_id from raw_output (Command object)
                        tool_call_id = None
                        if hasattr(raw_output, "update") and isinstance(
                            raw_output.update, dict
                        ):
                            messages = raw_output.update.get("messages", [])
                            if messages and len(messages) > 0:
                                tool_msg = messages[0]
                                if hasattr(tool_msg, "tool_call_id"):
                                    tool_call_id = tool_msg.tool_call_id

                        tool_calls.append(
                            {
                                "name": event_name,
                                "input": to_jsonable(raw_input),
                                "output": tool_output_for_db,
                                "tool_call_id": tool_call_id,
                            }
                        )

                        # Include artifacts in SSE event for frontend
                        event_data = {
                            "type": "tool_end",
                            "name": event_name,
                            "output": tool_output_for_db,
                        }
                        if artifacts:
                            # Convert S3 keys to presigned HTTP URLs and save to database
                            from backend.artifacts.storage import (
                                generate_presigned_url_from_s3_key,
                            )
                            from backend.artifacts.ingest import (
                                ingest_artifact_metadata,
                            )

                            # Save artifacts to DB immediately with separate session (commit before SSE send)
                            saved_artifact_dicts = []
                            async with ASYNC_SESSION_MAKER() as artifact_sess:
                                for art in artifacts:
                                    if "s3_key" in art and tool_call_id:
                                        try:
                                            artifact_dict = (
                                                await ingest_artifact_metadata(
                                                    session=artifact_sess,
                                                    thread_id=t.id,
                                                    s3_key=art["s3_key"],
                                                    sha256=art.get("sha256", ""),
                                                    filename=art.get("name", "unknown"),
                                                    mime=art.get(
                                                        "mime",
                                                        "application/octet-stream",
                                                    ),
                                                    size=art.get("size", 0),
                                                    session_id=str(
                                                        t.id
                                                    ),  # Modal sandbox session ID
                                                    tool_call_id=tool_call_id,
                                                )
                                            )
                                            saved_artifact_dicts.append(artifact_dict)
                                        except Exception as e:
                                            logging.warning(
                                                f"Failed to ingest artifact {art.get('name')}: {e}"
                                            )
                                # Commit immediately so artifacts are available for frontend refetch
                                await artifact_sess.commit()

                            # Now prepare artifacts for SSE with DB UUIDs (not temp SHA IDs)
                            converted_artifacts = []
                            for art_dict in saved_artifact_dicts:
                                try:
                                    # art_dict already has UUID from DB
                                    converted = {
                                        "id": art_dict[
                                            "id"
                                        ],  # Use DB UUID instead of SHA[:16]
                                        "name": art_dict["name"],
                                        "mime": art_dict["mime"],
                                        "size": art_dict["size"],
                                    }
                                    # Generate presigned URL - need to get s3_key from original artifacts list
                                    matching_art = next(
                                        (
                                            a
                                            for a in artifacts
                                            if a.get("sha256") == art_dict["sha256"]
                                        ),
                                        None,
                                    )
                                    if matching_art and "s3_key" in matching_art:
                                        converted["url"] = (
                                            generate_presigned_url_from_s3_key(
                                                matching_art["s3_key"]
                                            )
                                        )
                                    converted_artifacts.append(converted)
                                except Exception as e:
                                    logging.warning(
                                        f"Failed to prepare artifact for SSE: {e}"
                                    )
                                    continue

                            event_data["artifacts"] = converted_artifacts

                        yield f"data: {json.dumps(event_data)}\n\n"
                # Persist using a short-lived session to avoid holding an open connection during SSE
                a_msg_id = None
                async with ASYNC_SESSION_MAKER() as write_sess:
                    # Tool messages first
                    for idx, tool_call in enumerate(tool_calls):
                        # Extract tool_call_id if available
                        tool_call_id = tool_call.get("id") or tool_call.get(
                            "tool_call_id"
                        )

                        # Add tool_call_id to tool_input for easier retrieval
                        tool_input = tool_call.get("input", {})
                        if isinstance(tool_input, dict) and tool_call_id:
                            tool_input = {**tool_input, "tool_call_id": tool_call_id}

                        tool_msg = Message(
                            thread_id=t.id,
                            message_id=f"tool:{payload.message_id}:{idx}",
                            role="tool",
                            tool_name=tool_call["name"],
                            tool_input=tool_input,
                            tool_output=tool_call.get("output"),
                            content=None,
                            meta=(
                                {"tool_call_id": tool_call_id} if tool_call_id else None
                            ),
                        )
                        write_sess.add(tool_msg)

                    # Save supervisor message (main assistant response) FIRST
                    # This ensures it has an earlier created_at timestamp
                    if supervisor_content:
                        a_msg = Message(
                            thread_id=t.id,
                            message_id=f"assistant:{payload.message_id}",
                            role="assistant",
                            content=(
                                {"text": supervisor_content}
                                if isinstance(supervisor_content, str)
                                else supervisor_content
                            ),
                        )
                        write_sess.add(a_msg)
                        await write_sess.flush()  # Flush to get the ID and commit timestamp
                        a_msg_id = str(a_msg.id)

                    # Small delay to ensure supervisor message has earlier timestamp
                    import asyncio

                    await asyncio.sleep(0.01)  # 10ms delay

                    # Save subagent segments AFTER supervisor
                    # Order subagents by their actual execution order (tracked during streaming)
                    # This ensures messages appear in the order they were executed, not a hardcoded order
                    for agent_name in subagent_order:
                        # Save all segments for this agent
                        if agent_name in subagent_segments:
                            for segment in subagent_segments[agent_name]:
                                segment_content = segment["content"]
                                segment_index = segment["segment_index"]
                                subagent_msg = Message(
                                    thread_id=t.id,
                                    message_id=f"subagent:{payload.message_id}:{agent_name}:segment:{segment_index}",
                                    role="assistant",
                                    content=(
                                        {"text": segment_content}
                                        if isinstance(segment_content, str)
                                        else segment_content
                                    ),
                                    meta={
                                        "agent": agent_name,
                                        "segment_index": segment_index,
                                    },
                                )
                                write_sess.add(subagent_msg)
                                await asyncio.sleep(
                                    0.01
                                )  # Small delay between each segment

                        # Save any remaining content as final segment (if there's content after last tool)
                        if (
                            agent_name in subagent_content
                            and subagent_content[agent_name].strip()
                        ):
                            remaining_content = subagent_content[agent_name]
                            # Get the next segment index
                            segment_index = subagent_segment_counters.get(agent_name, 0)
                            subagent_msg = Message(
                                thread_id=t.id,
                                message_id=f"subagent:{payload.message_id}:{agent_name}:segment:{segment_index}",
                                role="assistant",
                                content=(
                                    {"text": remaining_content}
                                    if isinstance(remaining_content, str)
                                    else remaining_content
                                ),
                                meta={
                                    "agent": agent_name,
                                    "segment_index": segment_index,
                                },
                            )
                            write_sess.add(subagent_msg)
                            await asyncio.sleep(0.01)  # Small delay

                    # Handle any subagents not in the tracked order (shouldn't happen, but safety check)
                    # Save their segments if they exist
                    for agent_name in subagent_segments:
                        if agent_name not in subagent_order:
                            for segment in subagent_segments[agent_name]:
                                segment_content = segment["content"]
                                segment_index = segment["segment_index"]
                                subagent_msg = Message(
                                    thread_id=t.id,
                                    message_id=f"subagent:{payload.message_id}:{agent_name}:segment:{segment_index}",
                                    role="assistant",
                                    content=(
                                        {"text": segment_content}
                                        if isinstance(segment_content, str)
                                        else segment_content
                                    ),
                                    meta={
                                        "agent": agent_name,
                                        "segment_index": segment_index,
                                    },
                                )
                                write_sess.add(subagent_msg)

                    # Also handle any remaining content for agents not in order
                    for agent_name, agent_content in subagent_content.items():
                        if agent_name not in subagent_order and agent_content.strip():
                            # If no segments exist, save as single message (backward compatibility)
                            if agent_name not in subagent_segments:
                                subagent_msg = Message(
                                    thread_id=t.id,
                                    message_id=f"subagent:{payload.message_id}:{agent_name}",
                                    role="assistant",
                                    content=(
                                        {"text": agent_content}
                                        if isinstance(agent_content, str)
                                        else agent_content
                                    ),
                                    meta={"agent": agent_name},
                                )
                                write_sess.add(subagent_msg)

                    await write_sess.commit()
                    # If no supervisor message was saved, use None for message_id
                    if not a_msg_id and supervisor_content:
                        # This shouldn't happen, but just in case
                        logging.warning(
                            f"No supervisor message ID after commit for thread {thread_id}"
                        )

                yield f"data: {json.dumps({'type': 'done', 'message_id': a_msg_id})}\n\n"

        except Exception as e:
            logging.exception(
                "agent_stream_failed",
                extra={
                    "request_id": request_id,
                    "thread_id": str(thread_id),
                    "message_id": payload.message_id,
                },
            )
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            # Always clear context variables to prevent connection leaks
            clear_db_session()
            clear_thread_id()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# Resume endpoint for handling interrupts (human-in-the-loop)
class ResumeRequest(BaseModel):
    resume_value: (
        dict
    )  # dictionary of values 


@router.post("/threads/{thread_id}/continue")
async def continue_thread(
    thread_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Continue execution from the last checkpoint after user stopped execution.

    This is different from /resume which handles explicit interrupts with user decisions.
    This endpoint simply continues from where the graph left off when the SSE stream was cancelled.

    The checkpointer automatically saves state after each node execution, so we can
    resume by calling the graph with None (empty state update) and the same thread_id.
    """
    from backend.graph.graph import make_graph
    from backend.main import get_thread_lock, _checkpointer_cm
    from backend.graph.context import (
        set_db_session,
        set_thread_id,
        clear_db_session,
        clear_thread_id,
    )
    import uuid as uuid_module

    # Verify checkpointer is initialized
    if not _checkpointer_cm:
        raise HTTPException(status_code=500, detail="Checkpointer not initialized")

    # Get thread to verify it exists
    result = await session.execute(select(Thread).where(Thread.id == UUID(thread_id)))
    t = result.scalar_one_or_none()
    if not t:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Get thread config
    cfg_result = await session.execute(
        select(Config).where(Config.thread_id == UUID(thread_id))
    )
    cfg = cfg_result.scalar_one_or_none()

    # Get user API keys
    user_api_keys = await get_user_api_keys_for_llm(t.user_id, session)

    # Create graph with SAME config as original (checkpointer will restore state)
    graph = make_graph(
        model_name=cfg.model if cfg else None,
        temperature=cfg.temperature if cfg else None,
        system_prompt=cfg.system_prompt if cfg else None,
        context_window=cfg.context_window if cfg else None,
        checkpointer=_checkpointer_cm[0],  # Reuse global singleton checkpointer
        user_api_keys=user_api_keys,
    )

    # SAME config with SAME thread_id - checkpointer will restore state
    config = {"configurable": {"thread_id": str(thread_id)}, "recursion_limit": 150}

    # First, check if there's actually something to continue
    try:
        state_snapshot = await graph.aget_state(config)
        if not state_snapshot.next:
            # No pending nodes - execution was already complete
            raise HTTPException(
                status_code=400,
                detail="No execution to continue - thread is already complete",
            )
        logging.info(
            f"Continuing thread {thread_id} from node(s): {state_snapshot.next}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.warning(f"Could not check thread state: {e}")
        # Continue anyway - let the graph handle it

    # Stream response using SSE
    async def stream_continue():
        from backend.db.session import ASYNC_SESSION_MAKER

        try:
            lock = get_thread_lock(str(thread_id))
            async with lock:
                # Set context variables for tools
                set_db_session(session)
                set_thread_id(uuid_module.UUID(str(thread_id)))

                # Variables to track content per agent (same as POST /messages)
                current_agent_node = None
                supervisor_content = ""
                subagent_content = {}
                subagent_order = []
                subagent_segments = {}
                subagent_segment_counters = {}
                tool_calls = []
                last_known_agent_node = None

                # Generate a unique message_id for this continue operation
                continue_message_id = str(uuid_module.uuid4())

                # Key: Continue from checkpoint by calling astream_events with None
                # LangGraph will automatically resume from last saved state
                async for event in graph.astream_events(None, config, version="v2"):
                    event_type = event.get("event")
                    event_name = event.get("name", "")
                    event_meta = event.get("metadata", {})
                    node = event_meta.get("langgraph_node")
                    checkpoint_ns = event_meta.get("langgraph_checkpoint_ns", "")
                    event_meta.get("langgraph_step")

                    # Track which agent node we're currently in
                    if node and node not in ["model", "tools"]:
                        current_agent_node = node
                        last_known_agent_node = node
                        logging.debug(
                            f"Agent node updated (continue): {current_agent_node}"
                        )
                    elif node in ["model", "tools"] and checkpoint_ns:
                        checkpoint_parts = checkpoint_ns.split(":")
                        if checkpoint_parts and checkpoint_parts[0] in [
                            "supervisor",
                            "data_analyst",
                            "report_writer",
                            "reviewer",
                        ]:
                            current_agent_node = checkpoint_parts[0]
                            last_known_agent_node = checkpoint_parts[0]
                            logging.debug(
                                f"Agent node extracted from checkpoint_ns (continue): {current_agent_node}"
                            )
                        elif last_known_agent_node:
                            current_agent_node = last_known_agent_node

                    if (
                        event_type == "on_chain_start"
                        and node
                        and node not in ["model", "tools"]
                    ):
                        current_agent_node = node
                        last_known_agent_node = node
                        logging.debug(
                            f"Agent node from chain_start (continue): {current_agent_node}"
                        )

                    # Check for interrupts (graph might hit another interrupt)
                    data = event.get("data", {})
                    chunk = data.get("chunk")

                    if chunk and isinstance(chunk, dict) and "__interrupt__" in chunk:
                        logging.info(
                            f"🛑 INTERRUPT DETECTED (continue) - chunk keys: {chunk.keys()}"
                        )
                        interrupt_tuple = chunk["__interrupt__"]
                        interrupt_obj = interrupt_tuple[0]
                        interrupt_value = interrupt_obj.value
                        logging.info(f"Interrupt value (continue): {interrupt_value}")
                        yield f"data: {json.dumps({'type': 'interrupt', 'value': to_jsonable(interrupt_value)})}\n\n"

                    # Stream token chunks
                    if event_type == "on_chat_model_stream":
                        if not current_agent_node:
                            logging.warning(
                                f"on_chat_model_stream (continue) with no current_agent_node, node={node}, checkpoint_ns={checkpoint_ns}"
                            )

                        chunk = event.get("data", {}).get("chunk")
                        if chunk:
                            if hasattr(chunk, "content") and chunk.content:
                                chunk_text = extract_text_from_content(chunk.content)
                                if chunk_text:
                                    if current_agent_node == "supervisor":
                                        supervisor_content += chunk_text
                                        yield f"data: {json.dumps({'type': 'token', 'content': chunk_text})}\n\n"
                                    elif current_agent_node in [
                                        "data_analyst",
                                        "report_writer",
                                        "reviewer",
                                    ]:
                                        agent_name = current_agent_node
                                        if agent_name not in subagent_content:
                                            subagent_content[agent_name] = ""
                                            if agent_name not in subagent_order:
                                                subagent_order.append(agent_name)
                                        subagent_content[agent_name] += chunk_text
                                        logging.debug(
                                            f"Streaming subagent token (continue): {agent_name}, content length: {len(chunk_text)}"
                                        )
                                        yield f"data: {json.dumps({'type': 'subagent_token', 'agent': agent_name, 'content': chunk_text})}\n\n"
                                    else:
                                        if (
                                            current_agent_node
                                            != "SummarizationMiddleware.before_model"
                                        ):
                                            logging.warning(
                                                f"Unknown agent node during streaming (continue): {current_agent_node}, node={node}"
                                            )

                    elif event_type == "on_chat_model_start":
                        if node and node not in ["model", "tools"]:
                            current_agent_node = node
                            logging.debug(
                                f"Agent node from chat_model_start (continue): {current_agent_node}"
                            )

                        if current_agent_node == "reviewer":
                            yield f"data: {json.dumps({'type': 'reviewing', 'status': 'start'})}\n\n"

                    elif (
                        event_type == "on_chat_model_end"
                        and current_agent_node == "reviewer"
                    ):
                        yield f"data: {json.dumps({'type': 'reviewing', 'status': 'done'})}\n\n"

                        # Get the final_score from state after reviewer completes
                        try:
                            state_snapshot = await graph.aget_state(config)
                            current_state = state_snapshot.values
                            final_score = current_state.get("final_score")
                            if final_score is not None:
                                yield f"data: {json.dumps({'type': 'score_updated', 'score': final_score})}\n\n"
                        except Exception as e:
                            logging.warning(
                                f"Failed to get final_score from state (continue): {e}"
                            )

                    elif event_type == "on_tool_start":
                        tool_input = event.get("data", {}).get("input")
                        yield f"data: {json.dumps({'type': 'tool_start', 'name': event_name, 'input': to_jsonable(tool_input)})}\n\n"

                        # Save current segment for the active agent before tool starts
                        if current_agent_node and current_agent_node in [
                            "data_analyst",
                            "report_writer",
                            "reviewer",
                        ]:
                            agent_name = current_agent_node
                            if (
                                agent_name in subagent_content
                                and subagent_content[agent_name].strip()
                            ):
                                if agent_name not in subagent_segments:
                                    subagent_segments[agent_name] = []
                                    subagent_segment_counters[agent_name] = 0

                                segment_content = subagent_content[agent_name]
                                segment_index = subagent_segment_counters[agent_name]
                                subagent_segments[agent_name].append(
                                    {
                                        "content": segment_content,
                                        "segment_index": segment_index,
                                    }
                                )
                                subagent_segment_counters[agent_name] += 1
                                subagent_content[agent_name] = ""
                                logging.debug(
                                    f"Saved segment {segment_index} for {agent_name} (continue), content length: {len(segment_content)}"
                                )

                    elif event_type == "on_tool_end":
                        raw_output = event.get("data", {}).get("output")
                        raw_input = event.get("data", {}).get("input")

                        # Emit objectives_updated event
                        if event_name == "set_analysis_objectives_tool":
                            if hasattr(raw_output, "update") and isinstance(
                                raw_output.update, dict
                            ):
                                objectives = raw_output.update.get(
                                    "analysis_objectives", []
                                )
                                if objectives:
                                    yield f"data: {json.dumps({'type': 'objectives_updated', 'objectives': objectives})}\n\n"

                        # Emit report_written event
                        if event_name == "write_report_tool":
                            if hasattr(raw_output, "update") and isinstance(
                                raw_output.update, dict
                            ):
                                reports = raw_output.update.get("reports", {})
                                report_title = raw_output.update.get(
                                    "last_report_title", ""
                                )
                                if reports and report_title and report_title in reports:
                                    report_content = reports[report_title]
                                    yield f"data: {json.dumps({'type': 'report_written', 'title': report_title, 'content': report_content})}\n\n"

                        # Emit todos_updated event
                        if event_name == "write_todos":
                            if hasattr(raw_output, "update") and isinstance(
                                raw_output.update, dict
                            ):
                                todos = raw_output.update.get("todos", [])
                                if todos:
                                    yield f"data: {json.dumps({'type': 'todos_updated', 'todos': todos})}\n\n"

                        artifacts = None
                        tool_content = None

                        if hasattr(raw_output, "update") and isinstance(
                            raw_output.update, dict
                        ):
                            messages = raw_output.update.get("messages", [])
                            if messages and len(messages) > 0:
                                tool_msg = messages[0]
                                if hasattr(tool_msg, "artifact") and tool_msg.artifact:
                                    artifacts = tool_msg.artifact
                                if hasattr(tool_msg, "content"):
                                    tool_content = tool_msg.content
                        elif hasattr(raw_output, "artifact"):
                            artifacts = raw_output.artifact
                            if hasattr(raw_output, "content"):
                                tool_content = raw_output.content

                        tool_output_for_db = (
                            {"content": tool_content}
                            if isinstance(tool_content, str)
                            else (
                                to_jsonable(tool_content)
                                if tool_content
                                else to_jsonable(raw_output)
                            )
                        )

                        # Extract tool_call_id
                        tool_call_id = None
                        if hasattr(raw_output, "update") and isinstance(
                            raw_output.update, dict
                        ):
                            messages = raw_output.update.get("messages", [])
                            if messages and len(messages) > 0:
                                tool_msg = messages[0]
                                if hasattr(tool_msg, "tool_call_id"):
                                    tool_call_id = tool_msg.tool_call_id

                        tool_calls.append(
                            {
                                "name": event_name,
                                "input": to_jsonable(raw_input),
                                "output": tool_output_for_db,
                                "tool_call_id": tool_call_id,
                            }
                        )

                        event_data = {
                            "type": "tool_end",
                            "name": event_name,
                            "output": tool_output_for_db,
                        }
                        if artifacts:
                            from backend.artifacts.storage import (
                                generate_presigned_url_from_s3_key,
                            )
                            from backend.artifacts.ingest import (
                                ingest_artifact_metadata,
                            )

                            saved_artifact_dicts = []
                            async with ASYNC_SESSION_MAKER() as artifact_sess:
                                for art in artifacts:
                                    if "s3_key" in art and tool_call_id:
                                        try:
                                            artifact_dict = (
                                                await ingest_artifact_metadata(
                                                    session=artifact_sess,
                                                    thread_id=t.id,
                                                    s3_key=art["s3_key"],
                                                    sha256=art.get("sha256", ""),
                                                    filename=art.get("name", "unknown"),
                                                    mime=art.get(
                                                        "mime",
                                                        "application/octet-stream",
                                                    ),
                                                    size=art.get("size", 0),
                                                    session_id=str(t.id),
                                                    tool_call_id=tool_call_id,
                                                )
                                            )
                                            saved_artifact_dicts.append(artifact_dict)
                                        except Exception as e:
                                            logging.warning(
                                                f"Failed to ingest artifact {art.get('name')}: {e}"
                                            )
                                await artifact_sess.commit()

                            converted_artifacts = []
                            for art_dict in saved_artifact_dicts:
                                try:
                                    converted = {
                                        "id": art_dict["id"],
                                        "name": art_dict["name"],
                                        "mime": art_dict["mime"],
                                        "size": art_dict["size"],
                                    }
                                    matching_art = next(
                                        (
                                            a
                                            for a in artifacts
                                            if a.get("sha256") == art_dict["sha256"]
                                        ),
                                        None,
                                    )
                                    if matching_art and "s3_key" in matching_art:
                                        converted["url"] = (
                                            generate_presigned_url_from_s3_key(
                                                matching_art["s3_key"]
                                            )
                                        )
                                    converted_artifacts.append(converted)
                                except Exception as e:
                                    logging.warning(
                                        f"Failed to prepare artifact for SSE: {e}"
                                    )
                                    continue
                            event_data["artifacts"] = converted_artifacts

                        yield f"data: {json.dumps(event_data)}\n\n"

                # Persist messages to database
                a_msg_id = None
                async with ASYNC_SESSION_MAKER() as write_sess:
                    # Tool messages first
                    for idx, tool_call in enumerate(tool_calls):
                        tool_call_id = tool_call.get("id") or tool_call.get(
                            "tool_call_id"
                        )
                        tool_input = tool_call.get("input", {})
                        if isinstance(tool_input, dict) and tool_call_id:
                            tool_input = {**tool_input, "tool_call_id": tool_call_id}

                        tool_msg = Message(
                            thread_id=t.id,
                            message_id=f"tool:{continue_message_id}:{idx}",
                            role="tool",
                            tool_name=tool_call["name"],
                            tool_input=tool_input,
                            tool_output=tool_call.get("output"),
                            content=None,
                            meta=(
                                {"tool_call_id": tool_call_id} if tool_call_id else None
                            ),
                        )
                        write_sess.add(tool_msg)

                    # Save supervisor message FIRST
                    if supervisor_content:
                        a_msg = Message(
                            thread_id=t.id,
                            message_id=f"assistant:{continue_message_id}",
                            role="assistant",
                            content=(
                                {"text": supervisor_content}
                                if isinstance(supervisor_content, str)
                                else supervisor_content
                            ),
                        )
                        write_sess.add(a_msg)
                        await write_sess.flush()
                        a_msg_id = str(a_msg.id)

                    import asyncio

                    await asyncio.sleep(0.01)

                    # Save subagent segments AFTER supervisor
                    for agent_name in subagent_order:
                        if agent_name in subagent_segments:
                            for segment in subagent_segments[agent_name]:
                                segment_content = segment["content"]
                                segment_index = segment["segment_index"]
                                subagent_msg = Message(
                                    thread_id=t.id,
                                    message_id=f"subagent:{continue_message_id}:{agent_name}:segment:{segment_index}",
                                    role="assistant",
                                    content=(
                                        {"text": segment_content}
                                        if isinstance(segment_content, str)
                                        else segment_content
                                    ),
                                    meta={
                                        "agent": agent_name,
                                        "segment_index": segment_index,
                                    },
                                )
                                write_sess.add(subagent_msg)
                                await asyncio.sleep(0.01)

                        if (
                            agent_name in subagent_content
                            and subagent_content[agent_name].strip()
                        ):
                            remaining_content = subagent_content[agent_name]
                            segment_index = subagent_segment_counters.get(agent_name, 0)
                            subagent_msg = Message(
                                thread_id=t.id,
                                message_id=f"subagent:{continue_message_id}:{agent_name}:segment:{segment_index}",
                                role="assistant",
                                content=(
                                    {"text": remaining_content}
                                    if isinstance(remaining_content, str)
                                    else remaining_content
                                ),
                                meta={
                                    "agent": agent_name,
                                    "segment_index": segment_index,
                                },
                            )
                            write_sess.add(subagent_msg)
                            await asyncio.sleep(0.01)

                    # Handle any subagents not in the tracked order
                    for agent_name in subagent_segments:
                        if agent_name not in subagent_order:
                            for segment in subagent_segments[agent_name]:
                                segment_content = segment["content"]
                                segment_index = segment["segment_index"]
                                subagent_msg = Message(
                                    thread_id=t.id,
                                    message_id=f"subagent:{continue_message_id}:{agent_name}:segment:{segment_index}",
                                    role="assistant",
                                    content=(
                                        {"text": segment_content}
                                        if isinstance(segment_content, str)
                                        else segment_content
                                    ),
                                    meta={
                                        "agent": agent_name,
                                        "segment_index": segment_index,
                                    },
                                )
                                write_sess.add(subagent_msg)

                    for agent_name, agent_content in subagent_content.items():
                        if agent_name not in subagent_order and agent_content.strip():
                            if agent_name not in subagent_segments:
                                subagent_msg = Message(
                                    thread_id=t.id,
                                    message_id=f"subagent:{continue_message_id}:{agent_name}",
                                    role="assistant",
                                    content=(
                                        {"text": agent_content}
                                        if isinstance(agent_content, str)
                                        else agent_content
                                    ),
                                    meta={"agent": agent_name},
                                )
                                write_sess.add(subagent_msg)

                    await write_sess.commit()
                    if not a_msg_id and supervisor_content:
                        logging.warning(
                            f"No supervisor message ID after commit for continue thread {thread_id}"
                        )

                yield f"data: {json.dumps({'type': 'done', 'message_id': a_msg_id})}\n\n"

        except Exception as e:
            logging.error(f"Continue failed for thread {thread_id}: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            clear_db_session()
            clear_thread_id()

    return StreamingResponse(stream_continue(), media_type="text/event-stream")


@router.post("/threads/{thread_id}/resume")
async def resume_thread(
    thread_id: str,
    payload: ResumeRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Resume a thread after an interrupt.

    Uses the same graph instance (via singleton checkpointer) and config to resume execution.

    Resume value format is a dict for all interrupts. Currently (23/12/2025) interrupt is only used in assigning to report writer.
    """
    from backend.graph.graph import make_graph
    from backend.main import get_thread_lock, _checkpointer_cm
    from backend.graph.context import (
        set_db_session,
        set_thread_id,
        clear_db_session,
        clear_thread_id,
    )
    from langgraph.types import Command
    import uuid as uuid_module

    # Verify checkpointer is initialized
    if not _checkpointer_cm:
        raise HTTPException(status_code=500, detail="Checkpointer not initialized")

    # Get thread to verify it exists
    result = await session.execute(select(Thread).where(Thread.id == UUID(thread_id)))
    t = result.scalar_one_or_none()
    if not t:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Get thread config
    cfg_result = await session.execute(
        select(Config).where(Config.thread_id == UUID(thread_id))
    )
    cfg = cfg_result.scalar_one_or_none()

    # Get user API keys
    user_api_keys = await get_user_api_keys_for_llm(t.user_id, session)

    # Create graph with SAME config as original (checkpointer will restore state)
    graph = make_graph(
        model_name=cfg.model if cfg else None,
        temperature=cfg.temperature if cfg else None,
        system_prompt=cfg.system_prompt if cfg else None,
        context_window=cfg.context_window if cfg else None,
        checkpointer=_checkpointer_cm[0],  # Reuse global singleton checkpointer
        user_api_keys=user_api_keys,
    )

    # SAME config with SAME thread_id - checkpointer will restore state
    config = {"configurable": {"thread_id": str(thread_id)}, "recursion_limit": 150}

    # Stream response using SSE
    async def stream_resume():
        try:
            lock = get_thread_lock(str(thread_id))
            async with lock:
                # Set context variables for tools
                set_db_session(session)
                set_thread_id(uuid_module.UUID(str(thread_id)))

                # Variables to track content per agent (same as POST /messages)
                current_agent_node = None  # Track which agent we're currently executing
                # Track streamed content for database storage - separate for each agent
                supervisor_content = ""  # Supervisor messages (main response)
                subagent_content = (
                    {}
                )  # Subagent messages: {"data_analyst": "", "report_writer": "", "reviewer": ""}
                subagent_order = (
                    []
                )  # Track execution order: ["data_analyst", "reviewer", ...]
                # Track segments for each agent (saved when tools start)
                subagent_segments = (
                    {}
                )  # {"data_analyst": [{"content": "...", "segment_index": 0}, ...]}
                subagent_segment_counters = {}  # Track segment index per agent
                tool_calls = (
                    []
                )  # Track tool calls for persistence (same as POST /messages)

                # Track the last known agent node to handle cases where node becomes "model"/"tools"
                last_known_agent_node = None

                # Generate a unique message_id for this resume operation
                resume_message_id = str(uuid_module.uuid4())

                # Log resume value for debugging
                logging.info(f"[RESUME] Thread {thread_id} - Resume value type: {type(payload.resume_value)}, value: {payload.resume_value}")

                # Resume with Command(resume=resume_value) using SAME graph and config
                async for event in graph.astream_events(
                    Command(resume=payload.resume_value), config, version="v2"
                ):
                    event_type = event.get("event")
                    event_name = event.get("name", "")
                    event_meta = event.get("metadata", {})
                    node = event_meta.get("langgraph_node")
                    checkpoint_ns = event_meta.get("langgraph_checkpoint_ns", "")
                    event_meta.get("langgraph_step")

                    # Track which agent node we're currently in
                    # Important: node becomes "model"/"tools" during execution, so we need to track it
                    # when we see chain_start or chat_model_start events, OR extract from checkpoint_ns
                    if node and node not in ["model", "tools"]:
                        current_agent_node = node
                        last_known_agent_node = (
                            node  # Keep track of last known agent node
                        )
                        logging.debug(
                            f"Agent node updated (resume): {current_agent_node}"
                        )
                    elif node in ["model", "tools"] and checkpoint_ns:
                        # Extract agent name from checkpoint_ns (format: "data_analyst:uuid|model:uuid")
                        # The agent name is the first part before the first colon
                        checkpoint_parts = checkpoint_ns.split(":")
                        if checkpoint_parts and checkpoint_parts[0] in [
                            "supervisor",
                            "data_analyst",
                            "report_writer",
                            "reviewer",
                        ]:
                            current_agent_node = checkpoint_parts[0]
                            last_known_agent_node = checkpoint_parts[0]
                            logging.debug(
                                f"Agent node extracted from checkpoint_ns (resume): {current_agent_node}"
                            )
                        elif last_known_agent_node:
                            # Fallback to last known agent node
                            current_agent_node = last_known_agent_node

                    # Also track agent node from chain_start events (when agent node starts)
                    if (
                        event_type == "on_chain_start"
                        and node
                        and node not in ["model", "tools"]
                    ):
                        current_agent_node = node
                        last_known_agent_node = node
                        logging.debug(
                            f"Agent node from chain_start (resume): {current_agent_node}"
                        )

                    # Check for another interrupt (same logic as POST /messages)
                    data = event.get("data", {})
                    chunk = data.get("chunk")

                    if chunk and isinstance(chunk, dict) and "__interrupt__" in chunk:
                        logging.info(
                            f"🛑 INTERRUPT DETECTED (resume) - chunk keys: {chunk.keys()}"
                        )
                        interrupt_tuple = chunk[
                            "__interrupt__"
                        ]  # It's a tuple: (Interrupt(...),)
                        interrupt_obj = interrupt_tuple[
                            0
                        ]  # Get the Interrupt object from tuple
                        interrupt_value = (
                            interrupt_obj.value
                        )  # Get the actual value dict
                        logging.info(f"Interrupt value (resume): {interrupt_value}")
                        yield f"data: {json.dumps({'type': 'interrupt', 'value': to_jsonable(interrupt_value)})}\n\n"
                        # Stream will end naturally after interrupt

                    # Stream token chunks (same as POST /messages)
                    if event_type == "on_chat_model_stream":
                        # Log for debugging
                        if not current_agent_node:
                            logging.warning(
                                f"on_chat_model_stream (resume) with no current_agent_node, node={node}, checkpoint_ns={checkpoint_ns}"
                            )

                        chunk = event.get("data", {}).get("chunk")
                        if chunk:
                            # Extract text from chunk content
                            if hasattr(chunk, "content") and chunk.content:
                                chunk_text = extract_text_from_content(chunk.content)
                                if chunk_text:
                                    # Stream based on which agent is active
                                    if current_agent_node == "supervisor":
                                        # Supervisor: normal streaming
                                        supervisor_content += chunk_text
                                        yield f"data: {json.dumps({'type': 'token', 'content': chunk_text})}\n\n"
                                    elif current_agent_node in [
                                        "data_analyst",
                                        "report_writer",
                                        "reviewer",
                                    ]:
                                        # Subagents: translucido streaming with dropdown
                                        agent_name = current_agent_node
                                        if agent_name not in subagent_content:
                                            subagent_content[agent_name] = ""
                                            # Track execution order: add agent to order list when first token arrives
                                            if agent_name not in subagent_order:
                                                subagent_order.append(agent_name)
                                        subagent_content[agent_name] += chunk_text
                                        logging.debug(
                                            f"Streaming subagent token (resume): {agent_name}, content length: {len(chunk_text)}"
                                        )
                                        yield f"data: {json.dumps({'type': 'subagent_token', 'agent': agent_name, 'content': chunk_text})}\n\n"
                                    else:
                                        # Fallback: if we don't know the agent, log it
                                        # Skip SummarizationMiddleware - it's not an agent, just internal summarization
                                        if (
                                            current_agent_node
                                            != "SummarizationMiddleware.before_model"
                                        ):
                                            logging.warning(
                                                f"Unknown agent node during streaming (resume): {current_agent_node}, node={node}"
                                            )

                    # Track agent node from chat_model_start events too (before streaming begins)
                    elif event_type == "on_chat_model_start":
                        # When chat_model_start fires, node might still be the agent node
                        # or we can infer it from the context
                        if node and node not in ["model", "tools"]:
                            current_agent_node = node
                            logging.debug(
                                f"Agent node from chat_model_start (resume): {current_agent_node}"
                            )

                        # Detect reviewer start
                        if current_agent_node == "reviewer":
                            yield f"data: {json.dumps({'type': 'reviewing', 'status': 'start'})}\n\n"

                    # Detect reviewer end
                    elif (
                        event_type == "on_chat_model_end"
                        and current_agent_node == "reviewer"
                    ):
                        yield f"data: {json.dumps({'type': 'reviewing', 'status': 'done'})}\n\n"

                        # Get the final_score from state after reviewer completes
                        try:
                            state_snapshot = await graph.aget_state(config)
                            current_state = state_snapshot.values
                            final_score = current_state.get("final_score")
                            if final_score is not None:
                                yield f"data: {json.dumps({'type': 'score_updated', 'score': final_score})}\n\n"
                        except Exception as e:
                            logging.warning(
                                f"Failed to get final_score from state (resume): {e}"
                            )

                    # Tool events (same as POST /messages)
                    elif event_type == "on_tool_start":
                        tool_input = event.get("data", {}).get("input")
                        yield f"data: {json.dumps({'type': 'tool_start', 'name': event_name, 'input': to_jsonable(tool_input)})}\n\n"

                        # Save current segment for the active agent before tool starts
                        if current_agent_node and current_agent_node in [
                            "data_analyst",
                            "report_writer",
                            "reviewer",
                        ]:
                            agent_name = current_agent_node
                            if (
                                agent_name in subagent_content
                                and subagent_content[agent_name].strip()
                            ):
                                # Initialize segments list and counter if needed
                                if agent_name not in subagent_segments:
                                    subagent_segments[agent_name] = []
                                    subagent_segment_counters[agent_name] = 0

                                # Save current content as a segment
                                segment_content = subagent_content[agent_name]
                                segment_index = subagent_segment_counters[agent_name]
                                subagent_segments[agent_name].append(
                                    {
                                        "content": segment_content,
                                        "segment_index": segment_index,
                                    }
                                )
                                subagent_segment_counters[agent_name] += 1

                                # Reset content for next segment
                                subagent_content[agent_name] = ""
                                logging.debug(
                                    f"Saved segment {segment_index} for {agent_name} (resume), content length: {len(segment_content)}"
                                )

                    elif event_type == "on_tool_end":
                        raw_output = event.get("data", {}).get("output")
                        raw_input = event.get("data", {}).get("input")

                        # Emit objectives_updated event if set_analysis_objectives_tool was called
                        if event_name == "set_analysis_objectives_tool":
                            if hasattr(raw_output, "update") and isinstance(
                                raw_output.update, dict
                            ):
                                objectives = raw_output.update.get(
                                    "analysis_objectives", []
                                )
                                if objectives:
                                    yield f"data: {json.dumps({'type': 'objectives_updated', 'objectives': objectives})}\n\n"

                        # Emit report_written event if write_report_tool was called
                        if event_name == "write_report_tool":
                            if hasattr(raw_output, "update") and isinstance(
                                raw_output.update, dict
                            ):
                                reports = raw_output.update.get("reports", {})
                                report_title = raw_output.update.get(
                                    "last_report_title", ""
                                )
                                if reports and report_title and report_title in reports:
                                    report_content = reports[report_title]
                                    yield f"data: {json.dumps({'type': 'report_written', 'title': report_title, 'content': report_content})}\n\n"

                        # Emit todos_updated event if write_todos was called
                        if event_name == "write_todos":
                            # Output can be Command(update={'todos': ...})
                            if hasattr(raw_output, "update") and isinstance(
                                raw_output.update, dict
                            ):
                                todos = raw_output.update.get("todos", [])
                                if todos:
                                    yield f"data: {json.dumps({'type': 'todos_updated', 'todos': todos})}\n\n"

                        artifacts = None
                        tool_content = None

                        if hasattr(raw_output, "update") and isinstance(
                            raw_output.update, dict
                        ):
                            messages = raw_output.update.get("messages", [])
                            if messages and len(messages) > 0:
                                tool_msg = messages[0]
                                if hasattr(tool_msg, "artifact") and tool_msg.artifact:
                                    artifacts = tool_msg.artifact
                                if hasattr(tool_msg, "content"):
                                    tool_content = tool_msg.content
                        elif hasattr(raw_output, "artifact"):
                            artifacts = raw_output.artifact
                            if hasattr(raw_output, "content"):
                                tool_content = raw_output.content

                        tool_output_for_sse = (
                            {"content": tool_content}
                            if isinstance(tool_content, str)
                            else (
                                to_jsonable(tool_content)
                                if tool_content
                                else to_jsonable(raw_output)
                            )
                        )
                        tool_output_for_db = tool_output_for_sse  # Same format for DB

                        # Extract tool_call_id and collect tool_calls for persistence (same as POST /messages)
                        tool_call_id = None
                        if hasattr(raw_output, "update") and isinstance(
                            raw_output.update, dict
                        ):
                            messages = raw_output.update.get("messages", [])
                            if messages and len(messages) > 0:
                                tool_msg = messages[0]
                                if hasattr(tool_msg, "tool_call_id"):
                                    tool_call_id = tool_msg.tool_call_id

                        # Collect tool_call for persistence
                        tool_calls.append(
                            {
                                "name": event_name,
                                "input": to_jsonable(raw_input),
                                "output": tool_output_for_db,
                                "tool_call_id": tool_call_id,
                            }
                        )

                        event_data = {
                            "type": "tool_end",
                            "name": event_name,
                            "output": tool_output_for_sse,
                        }
                        if artifacts:
                            # Handle artifacts same as POST /messages
                            from backend.artifacts.storage import (
                                generate_presigned_url_from_s3_key,
                            )
                            from backend.artifacts.ingest import (
                                ingest_artifact_metadata,
                            )
                            from backend.db.session import ASYNC_SESSION_MAKER

                            # tool_call_id already extracted above (lines 1361-1368)
                            # Save artifacts to DB and collect descriptors with UUIDs
                            saved_artifact_dicts = []
                            async with ASYNC_SESSION_MAKER() as artifact_sess:
                                for art in artifacts:
                                    if "s3_key" in art and tool_call_id:
                                        try:
                                            artifact_dict = (
                                                await ingest_artifact_metadata(
                                                    session=artifact_sess,
                                                    thread_id=t.id,
                                                    s3_key=art["s3_key"],
                                                    sha256=art.get("sha256", ""),
                                                    filename=art.get("name", "unknown"),
                                                    mime=art.get(
                                                        "mime",
                                                        "application/octet-stream",
                                                    ),
                                                    size=art.get("size", 0),
                                                    session_id=str(t.id),
                                                    tool_call_id=tool_call_id,
                                                )
                                            )
                                            saved_artifact_dicts.append(artifact_dict)
                                        except Exception as e:
                                            logging.warning(
                                                f"Failed to ingest artifact {art.get('name')}: {e}"
                                            )
                                await artifact_sess.commit()

                            # Convert for SSE with DB UUIDs (not temp SHA IDs)
                            converted_artifacts = []
                            for art_dict in saved_artifact_dicts:
                                try:
                                    converted = {
                                        "id": art_dict[
                                            "id"
                                        ],  # Use DB UUID instead of SHA[:16]
                                        "name": art_dict["name"],
                                        "mime": art_dict["mime"],
                                        "size": art_dict["size"],
                                    }
                                    # Generate presigned URL - need to get s3_key from original artifacts list
                                    matching_art = next(
                                        (
                                            a
                                            for a in artifacts
                                            if a.get("sha256") == art_dict["sha256"]
                                        ),
                                        None,
                                    )
                                    if matching_art and "s3_key" in matching_art:
                                        converted["url"] = (
                                            generate_presigned_url_from_s3_key(
                                                matching_art["s3_key"]
                                            )
                                        )
                                    converted_artifacts.append(converted)
                                except Exception as e:
                                    logging.warning(
                                        f"Failed to prepare artifact for SSE: {e}"
                                    )
                                    continue
                            event_data["artifacts"] = converted_artifacts

                        yield f"data: {json.dumps(event_data)}\n\n"

                # Persist messages to database (same as POST /messages)
                a_msg_id = None
                from backend.db.session import ASYNC_SESSION_MAKER

                async with ASYNC_SESSION_MAKER() as write_sess:
                    # Tool messages first
                    for idx, tool_call in enumerate(tool_calls):
                        # Extract tool_call_id if available
                        tool_call_id = tool_call.get("id") or tool_call.get(
                            "tool_call_id"
                        )

                        # Add tool_call_id to tool_input for easier retrieval
                        tool_input = tool_call.get("input", {})
                        if isinstance(tool_input, dict) and tool_call_id:
                            tool_input = {**tool_input, "tool_call_id": tool_call_id}

                        tool_msg = Message(
                            thread_id=t.id,
                            message_id=f"tool:{resume_message_id}:{idx}",
                            role="tool",
                            tool_name=tool_call["name"],
                            tool_input=tool_input,
                            tool_output=tool_call.get("output"),
                            content=None,
                            meta=(
                                {"tool_call_id": tool_call_id} if tool_call_id else None
                            ),
                        )
                        write_sess.add(tool_msg)

                    # Save supervisor message (main assistant response) FIRST
                    # This ensures it has an earlier created_at timestamp
                    if supervisor_content:
                        a_msg = Message(
                            thread_id=t.id,
                            message_id=f"assistant:{resume_message_id}",
                            role="assistant",
                            content=(
                                {"text": supervisor_content}
                                if isinstance(supervisor_content, str)
                                else supervisor_content
                            ),
                        )
                        write_sess.add(a_msg)
                        await write_sess.flush()  # Flush to get the ID and commit timestamp
                        a_msg_id = str(a_msg.id)

                    # Small delay to ensure supervisor message has earlier timestamp
                    import asyncio

                    await asyncio.sleep(0.01)  # 10ms delay

                    # Save subagent segments AFTER supervisor
                    # Order subagents by their actual execution order (tracked during streaming)
                    # This ensures messages appear in the order they were executed, not a hardcoded order
                    for agent_name in subagent_order:
                        # Save all segments for this agent
                        if agent_name in subagent_segments:
                            for segment in subagent_segments[agent_name]:
                                segment_content = segment["content"]
                                segment_index = segment["segment_index"]
                                subagent_msg = Message(
                                    thread_id=t.id,
                                    message_id=f"subagent:{resume_message_id}:{agent_name}:segment:{segment_index}",
                                    role="assistant",
                                    content=(
                                        {"text": segment_content}
                                        if isinstance(segment_content, str)
                                        else segment_content
                                    ),
                                    meta={
                                        "agent": agent_name,
                                        "segment_index": segment_index,
                                    },
                                )
                                write_sess.add(subagent_msg)
                                await asyncio.sleep(
                                    0.01
                                )  # Small delay between each segment

                        # Save any remaining content as final segment (if there's content after last tool)
                        if (
                            agent_name in subagent_content
                            and subagent_content[agent_name].strip()
                        ):
                            remaining_content = subagent_content[agent_name]
                            # Get the next segment index
                            segment_index = subagent_segment_counters.get(agent_name, 0)
                            subagent_msg = Message(
                                thread_id=t.id,
                                message_id=f"subagent:{resume_message_id}:{agent_name}:segment:{segment_index}",
                                role="assistant",
                                content=(
                                    {"text": remaining_content}
                                    if isinstance(remaining_content, str)
                                    else remaining_content
                                ),
                                meta={
                                    "agent": agent_name,
                                    "segment_index": segment_index,
                                },
                            )
                            write_sess.add(subagent_msg)
                            await asyncio.sleep(0.01)  # Small delay

                    # Handle any subagents not in the tracked order (shouldn't happen, but safety check)
                    # Save their segments if they exist
                    for agent_name in subagent_segments:
                        if agent_name not in subagent_order:
                            for segment in subagent_segments[agent_name]:
                                segment_content = segment["content"]
                                segment_index = segment["segment_index"]
                                subagent_msg = Message(
                                    thread_id=t.id,
                                    message_id=f"subagent:{resume_message_id}:{agent_name}:segment:{segment_index}",
                                    role="assistant",
                                    content=(
                                        {"text": segment_content}
                                        if isinstance(segment_content, str)
                                        else segment_content
                                    ),
                                    meta={
                                        "agent": agent_name,
                                        "segment_index": segment_index,
                                    },
                                )
                                write_sess.add(subagent_msg)

                    # Also handle any remaining content for agents not in order
                    for agent_name, agent_content in subagent_content.items():
                        if agent_name not in subagent_order and agent_content.strip():
                            # If no segments exist, save as single message (backward compatibility)
                            if agent_name not in subagent_segments:
                                subagent_msg = Message(
                                    thread_id=t.id,
                                    message_id=f"subagent:{resume_message_id}:{agent_name}",
                                    role="assistant",
                                    content=(
                                        {"text": agent_content}
                                        if isinstance(agent_content, str)
                                        else agent_content
                                    ),
                                    meta={"agent": agent_name},
                                )
                                write_sess.add(subagent_msg)

                    await write_sess.commit()
                    # If no supervisor message was saved, use None for message_id
                    if not a_msg_id and supervisor_content:
                        logging.warning(
                            f"No supervisor message ID after commit for resume thread {thread_id}"
                        )

                yield f"data: {json.dumps({'type': 'done', 'message_id': a_msg_id})}\n\n"

        except Exception as e:
            logging.error(f"Resume failed for thread {thread_id}: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            # Always clear context variables to prevent connection leaks
            clear_db_session()
            clear_thread_id()

    return StreamingResponse(stream_resume(), media_type="text/event-stream")


# API Key Management Endpoints


class APIKeysRequest(BaseModel):
    openai_key: Optional[str] = None
    anthropic_key: Optional[str] = None


class APIKeysResponse(BaseModel):
    openai_key: Optional[str] = None  # Masked version
    anthropic_key: Optional[str] = None  # Masked version


@router.get("/users/{user_id}/api-keys", response_model=APIKeysResponse)
async def get_user_api_keys(user_id: str, session: AsyncSession = Depends(get_session)):
    """Get user's API keys (masked for security)."""
    start_time = time.time()
    logging.info(f"[API-KEYS] Starting fetch for user {user_id}")
    
    try:
        result = await session.execute(
            select(UserAPIKeys).where(UserAPIKeys.user_id == user_id)
        )
        query_time = time.time() - start_time
        logging.info(f"[API-KEYS] Query completed in {query_time:.3f}s for user {user_id}")
        
        user_keys = result.scalar_one_or_none()

        if not user_keys:
            total_time = time.time() - start_time
            logging.info(f"[API-KEYS] No keys found. Total time: {total_time:.3f}s for user {user_id}")
            return APIKeysResponse()

        response = APIKeysResponse(
            openai_key=(
                mask_api_key(decrypt_api_key(user_keys.openai_key))
                if user_keys.openai_key
                else None
            ),
            anthropic_key=(
                mask_api_key(decrypt_api_key(user_keys.anthropic_key))
            if user_keys.anthropic_key
                else None
            ),
        )
        
        total_time = time.time() - start_time
        logging.info(f"[API-KEYS] Success. Total time: {total_time:.3f}s for user {user_id}")
        return response
    except Exception as e:
        error_time = time.time() - start_time
        logging.error(f"[API-KEYS] Error after {error_time:.3f}s for user {user_id}: {str(e)}")
        raise


@router.post("/users/{user_id}/api-keys", response_model=APIKeysResponse)
async def save_user_api_keys(
    user_id: str, keys: APIKeysRequest, session: AsyncSession = Depends(get_session)
):
    """Save or update user's API keys."""
    # Get existing keys
    result = await session.execute(
        select(UserAPIKeys).where(UserAPIKeys.user_id == user_id)
    )
    user_keys = result.scalar_one_or_none()

    if not user_keys:
        # Create new record
        user_keys = UserAPIKeys(user_id=user_id)
        session.add(user_keys)

    # Update keys (encrypt before storing)
    # Handle both empty strings and None - treat both as "clear the key"
    if "openai_key" in keys.model_fields_set:
        user_keys.openai_key = (
            encrypt_api_key(keys.openai_key) if keys.openai_key else None
        )

    if "anthropic_key" in keys.model_fields_set:
        user_keys.anthropic_key = (
            encrypt_api_key(keys.anthropic_key) if keys.anthropic_key else None
        )

    await session.commit()
    await session.refresh(user_keys)

    # Return masked versions
    return APIKeysResponse(
        openai_key=(
            mask_api_key(decrypt_api_key(user_keys.openai_key))
            if user_keys.openai_key
            else None
        ),
        anthropic_key=(
            mask_api_key(decrypt_api_key(user_keys.anthropic_key))
            if user_keys.anthropic_key
            else None
        ),
    )


@router.get("/users/{user_id}/api-keys/raw")
async def get_user_api_keys_raw(
    user_id: str, session: AsyncSession = Depends(get_session)
):
    """Get user's API keys in raw format (for internal use by LLM services)."""
    start_time = time.time()
    logging.info(f"[API-KEYS-RAW] Starting fetch for user {user_id}")
    
    try:
        result = await session.execute(
            select(UserAPIKeys).where(UserAPIKeys.user_id == user_id)
        )
        query_time = time.time() - start_time
        logging.info(f"[API-KEYS-RAW] Query completed in {query_time:.3f}s for user {user_id}")
        
        user_keys = result.scalar_one_or_none()

        if not user_keys:
            total_time = time.time() - start_time
            logging.info(f"[API-KEYS-RAW] No keys found. Total time: {total_time:.3f}s for user {user_id}")
            return {"openai_key": None, "anthropic_key": None}

        response = {
            "openai_key": (
                decrypt_api_key(user_keys.openai_key) if user_keys.openai_key else None
            ),
            "anthropic_key": (
                decrypt_api_key(user_keys.anthropic_key)
                if user_keys.anthropic_key
                else None
            ),
        }
        
        total_time = time.time() - start_time
        logging.info(f"[API-KEYS-RAW] Success. Total time: {total_time:.3f}s for user {user_id}")
        return response
    except Exception as e:
        error_time = time.time() - start_time
        logging.error(f"[API-KEYS-RAW] Error after {error_time:.3f}s for user {user_id}: {str(e)}")
        raise
