import os
import uuid

# Resolve a stable session ID for this backend process
_SESSION_ID = os.getenv("LG_SESSION_ID") or f"host-{uuid.uuid4().hex[:8]}"
os.environ["LG_SESSION_ID"] = _SESSION_ID  # make it visible to everything else

def get_session_id() -> str:
    return _SESSION_ID

def volume_name() -> str:
    return f"lg-urban-session-{_SESSION_ID}"