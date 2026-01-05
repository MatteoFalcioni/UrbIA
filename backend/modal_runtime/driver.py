import json
import sys
import os
import hashlib
import mimetypes
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path


def scan_and_upload_artifacts(
    processed_artifacts: set, s3_bucket: str, s3_client, disable_upload: bool = False
):
    """Scan workspace for new artifacts and upload to S3."""
    artifacts = []

    # we cannot input these as parameters since the driver is a subprocess in sandbox
    artifacts_dir = (
        Path.cwd() / "artifacts"
    )  # Relative to current working dir (/workspace/sessions/{session_id}/)

    if not artifacts_dir.exists():
        return artifacts

    for file_path in artifacts_dir.rglob("*"):
        if file_path.is_file():
            dataset_exts = {".csv", ".parquet", ".xlsx", ".xls"}
            if file_path.suffix.lower() in dataset_exts:
                continue  # datasets are exported explicitly via tools
            try:
                # Compute SHA-256
                sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()

                # Skip if already processed
                if sha256 in processed_artifacts:
                    continue

                # Mark as processed
                processed_artifacts.add(sha256)

                # Get metadata
                mime = (
                    mimetypes.guess_type(file_path.name)[0]
                    or "application/octet-stream"
                )
                size = file_path.stat().st_size

                # Base metadata
                art = {
                    "name": file_path.name,
                    "path": str(file_path),
                    "sha256": sha256,
                    "mime": mime,
                    "size": size,
                }

                # Only upload to S3 if not disabled
                if not disable_upload:
                    # Read file content
                    file_bytes = file_path.read_bytes()
                    # Generate S3 key (content-addressed)
                    s3_key = f"output/artifacts/{sha256[:2]}/{sha256[2:4]}/{sha256}"
                    # Upload to S3
                    s3_client.put_object(
                        Bucket=s3_bucket, Key=s3_key, Body=file_bytes, ContentType=mime
                    )
                    art["s3_key"] = s3_key
                    art["s3_url"] = f"s3://{s3_bucket}/{s3_key}"

                artifacts.append(art)

            except Exception as e:
                print(f"Failed to process artifact {file_path}: {e}", file=sys.stderr)
                continue

    return artifacts


def driver_program():
    """Driver program that maintains Python state across executions."""
    globals_dict = {}  # Persistent namespace for user code
    processed_artifacts = set()  # Track processed artifacts to avoid duplicates

    # Check if S3 upload is disabled (for testing)
    disable_s3_upload = os.getenv("S3_DISABLE_UPLOAD", "0") == "1"

    # Initialize S3 client with Signature Version 4 and timeouts (only if upload is enabled)
    if not disable_s3_upload:
        # Import boto3 only when needed to avoid credential issues in CI
        import boto3
        from botocore.client import Config
        
        region = os.getenv("AWS_REGION", "eu-central-1")
        s3_client = boto3.client(
            "s3",
            region_name=region,
            config=Config(
                signature_version="s3v4",
                connect_timeout=5,
                read_timeout=10
            )
        )
        s3_bucket = os.getenv("S3_BUCKET", "lg-urban-prod")
    else:
        s3_client = None
        s3_bucket = None

    artifacts_dir = Path.cwd() / "artifacts"
    if not artifacts_dir.exists():
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Ensure unbuffered output
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    while True:
        try:
            # Use input() like Modal's example - cleaner than sys.stdin.readline()
            line = input()
            command = json.loads(line)

            if (code := command.get("code")) is None:
                result = {
                    "error": "No code to execute",
                    "stdout": "",
                    "stderr": "",
                    "artifacts": [],
                }
                print(json.dumps(result), flush=True)
                continue

            # Capture stdout/stderr
            stdout_io, stderr_io = StringIO(), StringIO()
            with redirect_stdout(stdout_io), redirect_stderr(stderr_io):
                try:
                    exec(code, globals_dict)
                except Exception as e:
                    print(f"Execution Error: {e}", file=sys.stderr)

            # Scan and upload artifacts after execution with timeout protection
            artifacts = []
            try:
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Artifact scan timed out")
                
                # Set 10 second timeout for artifact scanning
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
                
                artifacts = scan_and_upload_artifacts(
                    processed_artifacts,
                    s3_bucket,
                    s3_client,
                    disable_upload=disable_s3_upload,
                )
                
                signal.alarm(0)  # Cancel alarm if completed successfully
            except (TimeoutError, Exception):
                # If artifact scan times out or fails, continue without artifacts
                signal.alarm(0)  # Ensure alarm is cancelled
                artifacts = []

            # Emit exactly one JSON line per command
            result = {
                "stdout": stdout_io.getvalue(),
                "stderr": stderr_io.getvalue(),
                "artifacts": artifacts,
            }
            # CRITICAL: flush=True to ensure immediate output
            print(json.dumps(result), flush=True)

        except EOFError:
            # input() returns EOFError on EOF - exit gracefully
            break
        except json.JSONDecodeError:
            error_result = {
                "error": "Invalid JSON",
                "stdout": "",
                "stderr": "",
                "artifacts": [],
            }
            print(json.dumps(error_result), flush=True)
        except Exception as e:
            # Minimal error handling - don't pollute stderr with tracebacks
            error_result = {
                "error": f"Driver error: {e}",
                "stdout": "",
                "stderr": str(e),
                "artifacts": [],
            }
            print(json.dumps(error_result), flush=True)


if __name__ == "__main__":
    driver_program()
