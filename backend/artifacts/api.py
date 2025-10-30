# backend/artifacts/api.py
from __future__ import annotations
import uuid
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse, RedirectResponse

from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.session import get_session

from .storage import get_artifact_by_id, generate_artifact_download_url

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.get("/{artifact_id}")
async def download_artifact(
    artifact_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Download an artifact by ID.
    
    Server-side should authorize access. Redirect to a presigned S3 URL.
    """
    # 1) Look up artifact in PostgreSQL
    try:
        artifact_uuid = uuid.UUID(artifact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid artifact ID format")
    
    artifact = await get_artifact_by_id(session, artifact_uuid)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    # 2) Always use S3 presigned redirect in the new workflow
    try:
        url = generate_artifact_download_url(artifact)
        return RedirectResponse(url=url, status_code=302)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"S3 presign failed: {e}")


@router.get("/{artifact_id}/head")
async def head_artifact(
    artifact_id: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Get artifact metadata without downloading the file.
    
    Returns artifact metadata (size, mime type, SHA-256, etc.)
    """
    # 1) Look up artifact in PostgreSQL
    try:
        artifact_uuid = uuid.UUID(artifact_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid artifact ID format")
    
    artifact = await get_artifact_by_id(session, artifact_uuid)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    # 3) Return metadata
    return JSONResponse({
        "id": str(artifact.id),
        "sha256": artifact.sha256,
        "mime": artifact.mime,
        "filename": artifact.filename,
        "size": artifact.size,
        "created_at": artifact.created_at.isoformat(),
        "thread_id": str(artifact.thread_id),
        "session_id": artifact.session_id,
        "run_id": artifact.run_id,
        "tool_call_id": artifact.tool_call_id,
    })
