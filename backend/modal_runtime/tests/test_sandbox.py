import os
import pytest
from dotenv import load_dotenv

load_dotenv()


def test_modal_sandbox_executor_e2e():
    # Require Modal tokens to run this real integration test
    if not (os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")):
        pytest.skip("Modal tokens not configured; skipping real Modal integration test")

    # Configure env to avoid real S3 writes during the test
    os.environ.setdefault("ARTIFACTS_DIR", "/workspace/artifacts")
    os.environ.setdefault("S3_BUCKET", "unit-test-bucket")
    os.environ["S3_DISABLE_UPLOAD"] = "1"

    from backend.modal_runtime.executor import SandboxExecutor

    execu = SandboxExecutor(session_id="it-e2e")
    try:
        # 1) First run: basic stdout and state initialization
        r1 = execu.execute("x = 2\nprint('boot')")
        assert r1["stderr"] == ""
        assert "boot" in r1["stdout"]

        # 2) Second run: stateful use of x, artifact creation, stdout check
        code = (
            "import pathlib; p=pathlib.Path('/workspace/artifacts'); p.mkdir(parents=True, exist_ok=True)\n"
            "(p/'artifact.txt').write_text('ok')\n"
            "print(x+5)\n"
            "print('ready')\n"
        )
        r2 = execu.execute(code)
        assert r2["stderr"] == ""
        assert "7" in r2["stdout"]  # 2 + 5
        assert "ready" in r2["stdout"]
        assert isinstance(r2.get("artifacts"), list)
        # If artifacts found, at least our artifact.txt should be there
        if r2["artifacts"]:
            names = {a.get("name") for a in r2["artifacts"]}
            assert "artifact.txt" in names

    finally:
        # 3) Terminate the sandbox cleanly
        execu.terminate()



