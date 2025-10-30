# backend/modal_runtime/tools.py
import hashlib
import mimetypes
from pathlib import Path
import os
from typing import List, Dict, Any

import modal
import pandas as pd

# Import the Modal app from app.py
from .app import app, image
from .session import volume_name
WORKSPACE_VOLUME = modal.Volume.from_name(volume_name(), create_if_missing=True)

def _walk_files(base: Path, exts: set) -> List[Path]:
    files = []
    if base.exists():
        for p in base.rglob("*"):
            if p.is_file() and (not exts or p.suffix.lower() in exts):
                files.append(p)
    return files

def _session_base(session_id: str) -> Path:
    """Resolve per-session base dir; session_id must be provided by caller."""
    return Path("/workspace") / "sessions" / session_id

@app.function(
    image=image,
    volumes={"/workspace": WORKSPACE_VOLUME},
    timeout=60,
)
def list_loaded_datasets(
    session_id: str,
    subdir: str = "datasets"
) -> List[Dict[str, Any]]:
    """
    List datasets in the workspace. Return structured metadata.
    """
    base = _session_base(session_id) / subdir

    exts = {".csv", ".parquet", ".xlsx", ".xls"}
    out: List[Dict[str, Any]] = []
    for p in _walk_files(base, exts):
        stat = p.stat()
        rel = str(p.relative_to(base))
        out.append({
            "path": rel,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 3),
            "mtime": stat.st_mtime,
            "mime": mimetypes.guess_type(p.name)[0] or "application/octet-stream",
        })
    return out

@app.function(
    image=image,
    volumes={"/workspace": WORKSPACE_VOLUME},
    timeout=180,
    secrets=[modal.Secret.from_name("aws-credentials-IAM")],  # store AWS creds in Modal
)
def export_dataset(
    dataset_path: str,
    bucket: str,
    session_id: str
) -> Dict[str, Any]:
    """
    Upload a file from the Modal workspace to S3 and return metadata.
    """
    import boto3

    base = _session_base(session_id)
    full = base / dataset_path
    if not full.exists():
        return {"error": f"File not found: {dataset_path}"}

    try:
        data = full.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        mime = mimetypes.guess_type(full.name)[0] or "application/octet-stream"
        size = len(data)

        # Datasets exported under a separate prefix
        s3_key = f"output/datasets/{sha256[:2]}/{sha256[2:4]}/{sha256}"
        boto3.client("s3").put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=data,
            ContentType=mime,
        )

        return {
            "name": full.name,
            "path": str(full),
            "sha256": sha256,
            "mime": mime,
            "size": size,
            "s3_key": s3_key,
            "s3_url": f"s3://{bucket}/{s3_key}",
        }
    except Exception as e:
        return {"error": f"S3 upload failed: {e}"}

# Accept dataset bytes from backend and persist into the sandbox, returning summary
@app.function(
    image=image,
    volumes={"/workspace": WORKSPACE_VOLUME},
    timeout=180,
)
def write_dataset_bytes(
    dataset_id: str,
    data_b64: str,
    session_id: str,
    ext: str = "parquet",
    subdir: str = "datasets",
) -> Dict[str, Any]:
    import base64

    data = base64.b64decode(data_b64)
    base_dir = _session_base(session_id)
    datasets_dir = base_dir / subdir
    datasets_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{dataset_id}.{ext.lstrip('.')}"
    path = datasets_dir / filename
    path.write_bytes(data)
    # Ensure contents are flushed to the volume before returning
    try:
        with open(path, "rb", buffering=0) as fh:
            import os as _os
            _os.fsync(fh.fileno())
    except Exception:
        pass

    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    size = path.stat().st_size

    summary: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "path": str(path),
        "rel_path": str(path.relative_to(base_dir)),
        "mime": mime,
        "size_bytes": size,
        "size_mb": round(size / (1024 * 1024), 3),
        "ext": ext.lower(),
    }

    try:
        suf = path.suffix.lower()
        if suf == ".parquet":
            df = pd.read_parquet(path)
        elif suf == ".csv":
            df = pd.read_csv(path)
        elif suf in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            return {**summary, "note": f"Preview skipped: unsupported type {suf}"}

        summary.update({
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": list(map(str, df.columns)),
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "head": df.head(5).to_dict(orient="records"),
        })
    except Exception as e:
        summary["preview_error"] = f"{e}"

    return summary   