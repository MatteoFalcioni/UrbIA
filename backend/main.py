from __future__ import annotations

import os
from typing import List
from contextlib import asynccontextmanager
from asyncio import Lock


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def get_cors_origins() -> List[str]:
    """
    Parse the CORS_ORIGINS env var as a comma-separated list of origins.
    Example: "http://localhost:5173,http://localhost:3000".
    Returns an empty list if not set, in which case we default to "*" below.
    """
    raw = os.getenv("CORS_ORIGINS", "")
    return [o.strip() for o in raw.split(",") if o.strip()]


# Global state for LangGraph checkpointer (initialized once at startup, reused per request)
_checkpointer_cm = None
_thread_locks: dict[str, Lock] = {}


def get_thread_lock(thread_id: str) -> Lock:
    """
    Return (and create if needed) an asyncio.Lock for the given thread id.
    Ensures only one run occurs per thread at a time in this process.
    """
    lock = _thread_locks.get(thread_id)
    if lock is None:
        lock = Lock()
        _thread_locks[thread_id] = lock
    return lock


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize LangGraph checkpointer at startup (reused for all graph invocations); clean up on shutdown.
    Graphs are created per-request with thread-specific config.
    """
    global _checkpointer_cm
    from backend.graph.graph import get_checkpointer
    
    saver, cm = await get_checkpointer()
    _checkpointer_cm = (saver, cm)  # Store as tuple for easy access
    yield
    # Cleanup: close checkpointer context manager
    if hasattr(cm, "__aexit__"):
        await cm.__aexit__(None, None, None)


# Main FastAPI application instance used by the server (uvicorn/gunicorn)
app = FastAPI(title="LangGraph Chat Backend", lifespan=lifespan)

# CORS policy: allow configured origins (or "*" in dev) and standard headers/methods
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins() or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """
    Lightweight health check endpoint for readiness/liveness probes.
    Returns 200 with a simple JSON payload when the app is up.
    """
    return {"status": "ok"}

# API routers: threads, messages, artifacts
from backend.app import api as api_router_module
from backend.artifacts import api as artifacts_api

app.include_router(api_router_module.router, prefix="/api")
app.include_router(artifacts_api.router, prefix="/api")