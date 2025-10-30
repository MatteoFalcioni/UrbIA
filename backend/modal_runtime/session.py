import os
import uuid

from dotenv import load_dotenv

load_dotenv()

# Single place to define/override the session id
# Default: one per backend process (stable across imports)
_SESSION_ID = f"host-{uuid.uuid4().hex[:8]}"
os.environ["LG_SESSION_ID"] = _SESSION_ID

def get_session_id() -> str:
    return _SESSION_ID

def set_session_id(value: str) -> None:
    """Optional: call early at process start if you want to force a value."""
    global _SESSION_ID
    _SESSION_ID = value
    os.environ["LG_SESSION_ID"] = value

def set_session_id(value: str) -> None:
    """Optional: call early at process start if you want to force a value."""
    global _SESSION_ID
    _SESSION_ID = value
    os.environ["LG_SESSION_ID"] = value

# add below get_session_id() / set_session_id()
def session_base_dir(session_id: str | None = None) -> str:
    sid = session_id or get_session_id()
    return f"/workspace/sessions/{sid}"

def volume_name() -> str:
    # KEEP A SINGLE SHARED VOLUME for now to avoid breaking @app.function mounts.
    # If you later move to per-session volumes, change this to f"lg-urban-{_SESSION_ID}"
    return "lg-urban"