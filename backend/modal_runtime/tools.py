# backend/modal_runtime/tools.py
import os
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Any

import modal
import pandas as pd

from .app import app, image  # image must include pandas in requirements
from .session import volume_name

WORKSPACE_VOLUME = modal.Volume.from_name(volume_name(), create_if_missing=True)

def _walk_files(base: Path, exts: set) -> List[Path]:
    files = []
    if base.exists():
        for p in base.rglob("*"):
            if p.is_file() and (not exts or p.suffix.lower() in exts):
                files.append(p)
    return files

@app.function(
    image=image,
    volumes={"/workspace": WORKSPACE_VOLUME},
    workdir="/workspace",
    timeout=60,
)
def list_datasets(workspace_path: str = "/workspace") -> List[Dict[str, Any]]:
    """
    List datasets in the workspace. Return structured metadata.
    """
    base = Path(workspace_path)
    exts = {".csv", ".parquet", ".xlsx"}
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
    workdir="/workspace",   
    timeout=120,
)
def select_dataset(dataset_path: str, workspace_path: str = "/workspace") -> Dict[str, Any]:
    """
    Load a dataset in the sandbox and return summary info.
    """
    base = Path(workspace_path)
    full = base / dataset_path
    if not full.exists():
        return {"error": f"Dataset not found: {dataset_path}"}

    info: Dict[str, Any] = {"path": dataset_path}

    try:
        if full.suffix.lower() == ".csv":
            df = pd.read_csv(full)
        elif full.suffix.lower() == ".parquet":
            df = pd.read_parquet(full)
        elif full.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(full)
        else:
            return {"error": f"Unsupported file type: {full.suffix}"}

        info.update({
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": list(map(str, df.columns)),
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "missing_values": {str(c): int(v) for c, v in df.isna().sum().to_dict().items()},
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
            "head": df.head(5).to_dict(orient="records"),
        })
        return info
    except Exception as e:
        return {"error": f"Error loading dataset: {e}"}

@app.function(
    image=image,
    volumes={"/workspace": WORKSPACE_VOLUME},
    workdir="/workspace",
    timeout=180,
    secrets=[modal.Secret.from_name("aws-credentials")],  # store AWS creds in Modal
)
def export_dataset(dataset_path: str,
                   bucket: str,
                   workspace_path: str = "/workspace") -> Dict[str, Any]:
    """
    Upload a file from the Modal workspace to S3 and return metadata.
    """
    import boto3

    base = Path(workspace_path)
    full = base / dataset_path
    if not full.exists():
        return {"error": f"File not found: {dataset_path}"}

    try:
        data = full.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        mime = mimetypes.guess_type(full.name)[0] or "application/octet-stream"
        size = len(data)

        s3_key = f"output/artifacts/{sha256[:2]}/{sha256[2:4]}/{sha256}"
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