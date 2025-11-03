from __future__ import annotations

from typing import Optional, Any
from datetime import datetime, timezone
from uuid import UUID

import logging
import uuid
import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
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
            keys['openai_key'] = decrypt_api_key(user_keys.openai_key)
        if user_keys.anthropic_key:
            keys['anthropic_key'] = decrypt_api_key(user_keys.anthropic_key)
        
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
        return ''
    
    # Dict format (OpenAI old style or normalized)
    if isinstance(content, dict):
        return content.get('text', str(content))
    
    # List format (Claude/new models)
    if isinstance(content, list) and len(content) > 0:
        # Filter and extract only text blocks, ignore tool_use blocks
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # Only process text blocks, skip tool_use and other types
                if item.get('type') == 'text' and 'text' in item:
                    text_parts.append(item['text'])
                # Legacy format without explicit type
                elif 'text' in item and 'type' not in item:
                    text_parts.append(item['text'])
        
        if text_parts:
            return ''.join(text_parts)
        
        # Fallback: if no text blocks found, return empty string (don't show tool use JSON)
        return ''
    
    # String or other
    return str(content)


@router.post("/threads", response_model=ThreadOut)
async def create_thread(payload: ThreadCreate, session: AsyncSession = Depends(get_session)) -> ThreadOut:
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
    thread_text = "\n".join([
        f"{m.role}: {m.content.get('text', str(m.content)) if m.content else ''}"
        for m in messages
    ])
    
    if not thread_text.strip():
        raise HTTPException(status_code=400, detail="No messages to generate title from")

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
    if user_api_keys and user_api_keys.get('openai_key'):
        llm_kwargs['api_key'] = SecretStr(user_api_keys['openai_key'])
    elif os.getenv('OPENAI_API_KEY'):
        llm_kwargs['api_key'] = SecretStr(os.getenv('OPENAI_API_KEY'))
    
    llm = ChatOpenAI(**llm_kwargs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Return a concise, engaging title. No quotes, <= 8 words."),
        ("user", "Text:\n{body}\n\nTitle:")
    ])
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
        settings=None
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
    
    cfg = await session.get(Config, thread_id)
    if not cfg:
        # Return env-based defaults
        from backend.config import CONTEXT_WINDOW
        return ConfigOut(
            model=DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE,
            context_window=CONTEXT_WINDOW,
            settings=None
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
    t = await session.get(Thread, thread_id)
    if not t:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    cfg = await session.get(Config, thread_id)
    if not cfg:
        cfg = Config(thread_id=t.id)
        session.add(cfg)
    
    # Update fields if provided
    if payload.model is not None:
        cfg.model = payload.model
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
    role: str
    content: Optional[dict] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[dict] = None
    artifacts: list[ArtifactOut] = []

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
        token_count = state_snapshot.values.get("token_count", 0) if state_snapshot.values else 0
        context_window = cfg.context_window if (cfg and cfg.context_window) else DEFAULT_CONTEXT_WINDOW
        
        return {
            "token_count": token_count,
            "context_window": context_window
        }
    except Exception as e:
        # If state doesn't exist yet (new thread), return 0
        context_window = cfg.context_window if (cfg and cfg.context_window) else DEFAULT_CONTEXT_WINDOW
        return {
            "token_count": 0,
            "context_window": context_window
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
        .order_by(Message.created_at.desc())
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
            if isinstance(content[0], dict) and 'text' in content[0]:
                content = {"text": content[0]['text']}
        
        msg_dict = {
            "id": msg.id,
            "thread_id": msg.thread_id,
            "role": msg.role,
            "content": content,
            "tool_name": msg.tool_name,
            "tool_input": msg.tool_input,
            "tool_output": msg.tool_output,
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
                tool_msgs_stmt = tool_msgs_stmt.where(Message.created_at > turn_start_time)
            
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
                    artifact_stmt = select(Artifact).where(Artifact.tool_call_id == tool_call_id)
                    artifact_res = await session.execute(artifact_stmt)
                    artifacts = artifact_res.scalars().all()
                    
                    # Build artifact outputs with download URLs (S3 presigned)
                    for artifact in artifacts:
                        try:
                            url = generate_artifact_download_url(artifact, expiry_seconds=86400)
                            msg_dict["artifacts"].append({
                                "id": artifact.id,
                                "name": artifact.filename,
                                "mime": artifact.mime,
                                "size": artifact.size,
                                "url": url,
                            })
                        except Exception:
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
                if user_api_keys and user_api_keys.get('openai_key'):
                    llm_kwargs['api_key'] = SecretStr(user_api_keys['openai_key'])
                    logging.info("Using user's OpenAI API key for titling")
                elif os.getenv('OPENAI_API_KEY'):
                    llm_kwargs['api_key'] = SecretStr(os.getenv('OPENAI_API_KEY'))
                    logging.info("Using environment OpenAI API key for titling")
                else:
                    logging.warning("No OpenAI API key available for titling")
                    return
                
                llm = ChatOpenAI(**llm_kwargs)
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Return a concise, engaging title based on the user's request. No quotes, <= 8 words."),
                    ("user", "User message:\n{body}\n\nTitle:")
                ])
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
        from backend.graph.context import set_db_session, set_thread_id
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
            context_window=cfg.context_window if cfg else None,  # Use thread config or env default
            checkpointer=_checkpointer_cm[0],  # Reuse global checkpointer
            user_api_keys=user_api_keys,
        )

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        assistant_content = None
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
                config = {"configurable": {"thread_id": str(thread_id)}, "recursion_limit": 40}
                
                # Context update will be emitted after agent finishes processing
                
                # Stream events from LangGraph
                # We follow docs here: https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html?_gl=1*15ktatf*_gcl_au*MTc4MTgwMzA1Ny4xNzU4ODA2Mjcy*_ga*MTUzOTQwNjk3NS4xNzUwODY1MDM0*_ga_47WX3HKKY2*czE3NTk4MjY0Mzkkbzk5JGcxJHQxNzU5ODI2NTg0JGoxMyRsMCRoMA..#langchain_core.language_models.chat_models.BaseChatModel.astream_events
                
                # Variables to track thinking vs final response
                current_step_content = ""  # Accumulate ALL content in current step
                current_step_has_tools = False
                current_langgraph_step = None
                # Track all streamed content for database storage (only final response)
                all_streamed_content = ""
                
                async for event in graph.astream_events(state, config, version="v2"):
                    event_type = event.get("event")
                    event_name = event.get("name", "")
                    event_meta = event.get("metadata", {})
                    node = event_meta.get("langgraph_node")
                    checkpoint_ns = event_meta.get("langgraph_checkpoint_ns", "")
                    langgraph_step = event_meta.get("langgraph_step")
                    
                    # Detect step change - decide if previous step was thinking or final response
                    if langgraph_step != current_langgraph_step and current_langgraph_step is not None:
                        if current_step_content:
                            if current_step_has_tools:
                                # Previous step had tool calls - it was thinking (Claude pattern)
                                yield f"data: {json.dumps({'type': 'thinking', 'content': current_step_content})}\n\n"
                            else:
                                # Previous step had no tool calls - it was final response, stream it now
                                all_streamed_content += current_step_content
                                yield f"data: {json.dumps({'type': 'token', 'content': current_step_content})}\n\n"
                        
                        # Reset for new step
                        current_step_content = ""
                        current_step_has_tools = False
                    
                    current_langgraph_step = langgraph_step
                    
                    # Stream token chunks from the LLM (but not from summarizer or its sub-calls)
                    if event_type == "on_chat_model_stream":
                        # Skip if we're inside summarization context (agent called by summarizer)
                        if checkpoint_ns.startswith("summarize_conversation:"):
                            continue
                        
                        chunk = event.get("data", {}).get("chunk")
                        if chunk:
                            # Check if this chunk has tool calls (indicates thinking/reasoning phase)
                            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                                current_step_has_tools = True
                            
                            # Extract text from chunk content - always accumulate, decide later
                            if hasattr(chunk, "content") and chunk.content:
                                chunk_text = extract_text_from_content(chunk.content)
                                if chunk_text:
                                    current_step_content += chunk_text
                    
                    # Detect summarization start
                    elif event_type == "on_chat_model_start" and checkpoint_ns.startswith("summarize_conversation:"):
                        yield f"data: {json.dumps({'type': 'summarizing', 'status': 'start'})}\n\n"
                    
                    # Detect summarization end
                    elif event_type == "on_chain_end" and node == "agent":
                        # Emit context update after agent finishes processing
                        try:
                            from backend.config import CONTEXT_WINDOW as DEFAULT_CONTEXT_WINDOW
                            state_snapshot = await graph.aget_state(config)
                            token_count = state_snapshot.values.get("token_count", 0) if state_snapshot.values else 0
                            max_tokens = cfg.context_window if cfg and cfg.context_window else DEFAULT_CONTEXT_WINDOW
                            yield f"data: {json.dumps({'type': 'context_update', 'tokens_used': token_count, 'max_tokens': max_tokens})}\n\n"
                        except Exception as e:
                            logging.warning(f"Failed to get state for context update: {e}")
                    
                    
                    elif event_type == "on_chat_model_end" and checkpoint_ns.startswith("summarize_conversation:"):
                        yield f"data: {json.dumps({'type': 'summarizing', 'status': 'done'})}\n\n"
                        # Emit context reset immediately after summarization (token_count is now 0)
                        from backend.config import CONTEXT_WINDOW as DEFAULT_CONTEXT_WINDOW
                        max_tokens = cfg.context_window if cfg and cfg.context_window else DEFAULT_CONTEXT_WINDOW
                        yield f"data: {json.dumps({'type': 'context_update', 'tokens_used': 0, 'max_tokens': max_tokens})}\n\n"
                
                    # Stream tool execution start
                    elif event_type == "on_tool_start":
                        tool_input = event.get("data", {}).get("input")
                        yield f"data: {json.dumps({'type': 'tool_start', 'name': event_name, 'input': to_jsonable(tool_input)})}\n\n"
                        
                    # Stream tool execution end and capture for persistence
                    elif event_type == "on_tool_end":   
                        raw_input = event.get("data", {}).get("input")
                        raw_output = event.get("data", {}).get("output")
                        
                        # Extract artifacts and content from Command -> ToolMessage if present
                        artifacts = None
                        tool_content = None
                        
                        # Case 1: Output is a Command object (from code_sandbox tool)
                        if hasattr(raw_output, "update") and isinstance(raw_output.update, dict):
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
                            tool_output_for_db = {"content": tool_content} if isinstance(tool_content, str) else to_jsonable(tool_content)
                        else:
                            # Fallback to jsonable representation
                            tool_output_for_db = to_jsonable(raw_output)
                        
                        # Extract tool_call_id from raw_output (Command object)
                        tool_call_id = None
                        if hasattr(raw_output, "update") and isinstance(raw_output.update, dict):
                            messages = raw_output.update.get("messages", [])
                            if messages and len(messages) > 0:
                                tool_msg = messages[0]
                                if hasattr(tool_msg, "tool_call_id"):
                                    tool_call_id = tool_msg.tool_call_id
                        
                        tool_calls.append({
                            "name": event_name,
                            "input": to_jsonable(raw_input),
                            "output": tool_output_for_db,
                            "tool_call_id": tool_call_id,
                        })
                        
                        # Include artifacts in SSE event for frontend
                        event_data = {
                            'type': 'tool_end',
                            'name': event_name,
                            'output': tool_output_for_db
                        }
                        if artifacts:
                            event_data['artifacts'] = artifacts
                        
                        yield f"data: {json.dumps(event_data)}\n\n"
                    
                    elif event_type == "on_chat_model_end" and checkpoint_ns.startswith("summarize_conversation:"):
                        yield f"data: {json.dumps({'type': 'summarizing', 'status': 'done'})}\n\n"
                        # Emit context reset immediately after summarization (token_count is now 0)
                        from backend.config import CONTEXT_WINDOW as DEFAULT_CONTEXT_WINDOW
                        max_tokens = cfg.context_window if cfg and cfg.context_window else DEFAULT_CONTEXT_WINDOW
                        yield f"data: {json.dumps({'type': 'context_update', 'tokens_used': 0, 'max_tokens': max_tokens})}\n\n"
                
                    # Capture final assistant message (but not from summarizer or its sub-calls)
                    elif event_type == "on_chat_model_end":
                        # Skip if inside summarization context
                        if checkpoint_ns.startswith("summarize_conversation:"):
                            continue
                        
                        output = event.get("data", {}).get("output")
                        if output and hasattr(output, "content"):
                            assistant_content = output.content
                
                # Handle the last step's content after the loop ends
                if current_step_content:
                    if current_step_has_tools:
                        # Last step had tool calls - it was thinking
                        yield f"data: {json.dumps({'type': 'thinking', 'content': current_step_content})}\n\n"
                    else:
                        # Last step had no tool calls - it was final response
                        all_streamed_content += current_step_content
                        yield f"data: {json.dumps({'type': 'token', 'content': current_step_content})}\n\n"
                
                # Persist using a short-lived session to avoid holding an open connection during SSE
                a_msg_id = None
                async with ASYNC_SESSION_MAKER() as write_sess:
                    # Tool messages first
                    for idx, tool_call in enumerate(tool_calls):
                        # Extract tool_call_id if available
                        tool_call_id = tool_call.get("id") or tool_call.get("tool_call_id")
                        
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
                            meta={"tool_call_id": tool_call_id} if tool_call_id else None,
                        )
                        write_sess.add(tool_msg)

                    # Assistant message - use content that was actually streamed to user
                    assistant_content_to_save = all_streamed_content if all_streamed_content else ""
                    logging.info(f"DEBUG: all_streamed_content='{all_streamed_content}', assistant_content='{assistant_content}', saving='{assistant_content_to_save}'")
                    a_msg = Message(
                        thread_id=t.id,
                        message_id=f"assistant:{payload.message_id}",
                        role="assistant",
                        content={"text": assistant_content_to_save} if isinstance(assistant_content_to_save, str) else assistant_content_to_save,
                    )
                    write_sess.add(a_msg)
                    await write_sess.commit()
                    a_msg_id = str(a_msg.id)

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

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# API Key Management Endpoints

class APIKeysRequest(BaseModel):
    openai_key: Optional[str] = None
    anthropic_key: Optional[str] = None


class APIKeysResponse(BaseModel):
    openai_key: Optional[str] = None  # Masked version
    anthropic_key: Optional[str] = None  # Masked version


@router.get("/users/{user_id}/api-keys", response_model=APIKeysResponse)
async def get_user_api_keys(
    user_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get user's API keys (masked for security)."""
    result = await session.execute(
        select(UserAPIKeys).where(UserAPIKeys.user_id == user_id)
    )
    user_keys = result.scalar_one_or_none()
    
    if not user_keys:
        return APIKeysResponse()
    
    return APIKeysResponse(
        openai_key=mask_api_key(decrypt_api_key(user_keys.openai_key)) if user_keys.openai_key else None,
        anthropic_key=mask_api_key(decrypt_api_key(user_keys.anthropic_key)) if user_keys.anthropic_key else None,
    )


@router.post("/users/{user_id}/api-keys", response_model=APIKeysResponse)
async def save_user_api_keys(
    user_id: str,
    keys: APIKeysRequest,
    session: AsyncSession = Depends(get_session)
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
    if 'openai_key' in keys.model_fields_set:
        user_keys.openai_key = encrypt_api_key(keys.openai_key) if keys.openai_key else None
    
    if 'anthropic_key' in keys.model_fields_set:
        user_keys.anthropic_key = encrypt_api_key(keys.anthropic_key) if keys.anthropic_key else None
    
    await session.commit()
    await session.refresh(user_keys)
    
    # Return masked versions
    return APIKeysResponse(
        openai_key=mask_api_key(decrypt_api_key(user_keys.openai_key)) if user_keys.openai_key else None,
        anthropic_key=mask_api_key(decrypt_api_key(user_keys.anthropic_key)) if user_keys.anthropic_key else None,
    )


@router.get("/users/{user_id}/api-keys/raw")
async def get_user_api_keys_raw(
    user_id: str,
    session: AsyncSession = Depends(get_session)
):
    """Get user's API keys in raw format (for internal use by LLM services)."""
    result = await session.execute(
        select(UserAPIKeys).where(UserAPIKeys.user_id == user_id)
    )
    user_keys = result.scalar_one_or_none()
    
    if not user_keys:
        return {"openai_key": None, "anthropic_key": None}
    
    return {
        "openai_key": decrypt_api_key(user_keys.openai_key) if user_keys.openai_key else None,
        "anthropic_key": decrypt_api_key(user_keys.anthropic_key) if user_keys.anthropic_key else None,
    }

