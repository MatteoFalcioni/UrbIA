import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_driver_with_commands(commands, env=None, timeout=5):
    """Start the driver.py as a subprocess and exchange JSON lines."""
    # Support running from repo root or from backend/modal_runtime
    here = Path(__file__).resolve()
    modal_runtime_dir = here.parents[1]
    driver_path = modal_runtime_dir / "driver.py"

    # Ensure Python outputs are unbuffered
    full_env = env.copy() if env else os.environ.copy()
    full_env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-u", str(driver_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffered
        env=full_env,
    )

    try:
        outputs = []
        for cmd in commands:
            # Write command
            proc.stdin.write(json.dumps(cmd) + "\n")
            proc.stdin.flush()
            
            # Read response
            line = proc.stdout.readline()
            if not line:
                # Check if process died
                proc.poll()
                if proc.returncode is not None:
                    stderr_output = proc.stderr.read()
                    raise RuntimeError(f"Driver process ended unexpectedly with code {proc.returncode}. Stderr: {stderr_output}")
                raise RuntimeError("Driver process ended unexpectedly")
            
            outputs.append(json.loads(line.strip()))
        
        return outputs
    finally:
        # Gracefully close stdin so the driver loop can exit
        try:
            proc.stdin.close()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        except Exception:
            proc.kill()


def test_driver_state_persists_and_stdout_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare environment for the driver
        env = os.environ.copy()
        env.setdefault("ARTIFACTS_DIR", tmpdir)  # ensure no uploads attempted
        env.setdefault("S3_BUCKET", "unit-test-bucket")
        # Dummy AWS creds to avoid boto3 complaining if ever used
        env.setdefault("AWS_ACCESS_KEY_ID", "test")
        env.setdefault("AWS_SECRET_ACCESS_KEY", "test")
        env.setdefault("AWS_REGION", "eu-central-1")

        # First command defines a variable
        cmd1 = {"code": "x = 41"}
        # Second command uses the previously defined variable
        cmd2 = {"code": "print(x + 1)"}

        outputs = run_driver_with_commands([cmd1, cmd2], env=env)

        # First has empty stdout, second should print 42
        assert outputs[0]["stdout"] == ""
        assert outputs[0]["stderr"] == ""
        assert "42" in outputs[1]["stdout"].strip()
        assert outputs[1]["stderr"] == ""


