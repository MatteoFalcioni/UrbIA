# backend/artifacts/ingest.py
"""
S3-only artifact ingestion: insert metadata for files uploaded by Modal.
"""

from __future__ import annotations
import uuid
from typing import Dict, Optional
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

# Tokens removed; downloads use S3 presigned URLs via API


# ---------- small helpers ----------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# No local-file ingest in S3-only workflow


# ---------------- S3 metadata ingestion (no file bytes on backend) ----------------
async def ingest_artifact_metadata(
    session: AsyncSession,
    *,
    thread_id: uuid.UUID,
    s3_key: str,
    sha256: str,
    filename: str,
    mime: str,
    size: int,
    session_id: str,
    tool_call_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict:
    """
    Insert artifact metadata when the file is already persisted to S3 by Modal.

    Returns a descriptor consistent with ingest_files output.
    """
    from .storage import create_artifact

    artifact = await create_artifact(
        session=session,
        thread_id=thread_id,
        sha256=sha256,
        filename=filename,
        mime=mime,
        size=size,
        session_id=session_id,
        run_id=run_id,
        tool_call_id=tool_call_id,
        meta={"s3_key": s3_key},
    )

    await session.commit()

    desc: Dict = {
        "id": str(artifact.id),
        "name": artifact.filename,
        "mime": artifact.mime,
        "size": artifact.size,
        "sha256": artifact.sha256,
        "created_at": artifact.created_at.isoformat(),
    }

    # URL is provided by API via S3 presigned redirect

    return desc

# Legacy upsert logic removed (no local file copies)
