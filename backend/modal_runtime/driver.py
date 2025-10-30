import json
import sys
import os
import hashlib
import mimetypes
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

# driver runs inside sandbox, so it has access to the workspace

def scan_and_upload_artifacts(processed_artifacts: set, s3_bucket: str, s3_client):
    """Scan workspace for new artifacts and upload to S3."""
    artifacts = []
    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "/workspace/artifacts"))
    
    if not artifacts_dir.exists():
        return artifacts
    
    for file_path in artifacts_dir.rglob("*"):
        if file_path.is_file():
            try:
                # Compute SHA-256
                sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
                
                # Skip if already processed
                if sha256 in processed_artifacts:
                    continue
                
                # Mark as processed
                processed_artifacts.add(sha256)
                
                # Get metadata
                mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
                size = file_path.stat().st_size
                
                # Read file content
                file_bytes = file_path.read_bytes()
                
                # Generate S3 key (content-addressed)
                s3_key = f"output/artifacts/{sha256[:2]}/{sha256[2:4]}/{sha256}"
                
                # Upload to S3
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=s3_key,
                    Body=file_bytes,
                    ContentType=mime
                )
                
                artifacts.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "sha256": sha256,
                    "mime": mime,
                    "size": size,
                    "s3_key": s3_key,
                    "s3_url": f"s3://{s3_bucket}/{s3_key}"
                })
                
            except Exception as e:
                print(f"Failed to process artifact {file_path}: {e}", file=sys.stderr)
                continue
    
    return artifacts

def driver_program():
    """Driver program that maintains Python state across executions."""
    import boto3
    
    globals_dict = {}  # Persistent namespace for user code
    processed_artifacts = set()  # Track processed artifacts to avoid duplicates
    
    # Initialize S3 client
    s3_client = boto3.client('s3')
    s3_bucket = os.getenv('S3_BUCKET', 'lg-urban-prod')
    
    while True:
        try:
            command = json.loads(input())  # Read JSON command from stdin
            code = command.get("code")
            if not code:
                print(json.dumps({"error": "No code provided"}), flush=True)
                continue
            
            # Capture stdout/stderr
            stdout_io, stderr_io = StringIO(), StringIO()
            with redirect_stdout(stdout_io), redirect_stderr(stderr_io):
                try:
                    exec(code, globals_dict)
                except Exception as e:
                    print(f"Execution Error: {e}", file=sys.stderr)
            
            # Scan and upload artifacts after execution
            artifacts = scan_and_upload_artifacts(processed_artifacts, s3_bucket, s3_client)
            
            print(json.dumps({
                "stdout": stdout_io.getvalue(),
                "stderr": stderr_io.getvalue(),
                "artifacts": artifacts
            }), flush=True)
            
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    driver_program()

