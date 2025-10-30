# backend/modal_runtime/tools.py
import hashlib
import mimetypes
from pathlib import Path
from typing import List, Dict, Any

import modal
import pandas as pd

app = modal.App("lg-urban-executor")
image = modal.Image.debian_slim(python_version="3.11")\
    .pip_install_from_requirements("backend/modal_runtime/requirements.txt")\
    .add_local_file("backend/modal_runtime/driver.py", "/root/driver.py")

WORKSPACE_VOLUME = modal.Volume.from_name("lg-urban", create_if_missing=True)

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
    timeout=60,
)
def list_available_datasets(workspace_path: str = "/workspace", subdir: str = "datasets") -> List[Dict[str, Any]]:
    """
    List datasets in the workspace. Return structured metadata.
    """
    datasets_dir = Path(workspace_path) / subdir

    exts = {".csv", ".parquet", ".xlsx", ".xls"}
    out: List[Dict[str, Any]] = []
    for p in _walk_files(datasets_dir, exts):
        stat = p.stat()
        rel = str(p.relative_to(datasets_dir))
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
def write_dataset_bytes(dataset_id: str, data_b64: str, ext: str = "parquet", subdir: str = "datasets") -> Dict[str, Any]:
    import base64

    data = base64.b64decode(data_b64)
    datasets_dir = Path("/workspace") / subdir
    datasets_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{dataset_id}.{ext.lstrip('.')}"
    path = datasets_dir / filename
    path.write_bytes(data)

    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    size = path.stat().st_size

    summary: Dict[str, Any] = {
        "dataset_id": dataset_id,
        "path": str(path),
        "rel_path": str(path.relative_to(Path("/workspace"))),
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