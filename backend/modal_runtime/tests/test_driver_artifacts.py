import os
import sys
from pathlib import Path
from types import SimpleNamespace

import tempfile

# Ensure parent directory (which contains driver.py) is importable when running from backend/modal_runtime
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from backend.modal_runtime.driver import scan_and_upload_artifacts
except ModuleNotFoundError:
    from driver import scan_and_upload_artifacts


class FakeS3Client:
    def __init__(self):
        self.put_calls = []

    def put_object(self, Bucket, Key, Body, ContentType):
        self.put_calls.append(
            SimpleNamespace(Bucket=Bucket, Key=Key, Body=Body, ContentType=ContentType)
        )


def test_scan_and_upload_artifacts_uploads_and_returns_metadata(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = Path(tmpdir)
        # ensure directory exists
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # create two files
        f1 = artifacts_dir / "a.txt"
        f1.write_text("hello")
        f2 = artifacts_dir / "b.bin"
        f2.write_bytes(b"\x00\x01\x02")

        # point driver to our temp dir
        monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))

        processed = set()
        fake_s3 = FakeS3Client()
        bucket = "unit-test-bucket"

        result = scan_and_upload_artifacts(processed, bucket, fake_s3)

        # two artifacts returned
        assert len(result) == 2
        # two S3 uploads performed
        assert len(fake_s3.put_calls) == 2
        # processed set updated
        assert len(processed) == 2
        # keys use content-addressed layout
        for art in result:
            sha = art["sha256"]
            assert art["s3_key"].endswith(sha)
            assert f"/{sha[:2]}/{sha[2:4]}/" in art["s3_key"]
            assert art["s3_url"] == f"s3://{bucket}/{art['s3_key']}"


def test_scan_and_upload_artifacts_is_idempotent(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts_dir = Path(tmpdir)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        f1 = artifacts_dir / "a.txt"
        f1.write_text("hello")

        monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts_dir))

        processed = set()
        fake_s3 = FakeS3Client()
        bucket = "unit-test-bucket"

        first = scan_and_upload_artifacts(processed, bucket, fake_s3)
        second = scan_and_upload_artifacts(processed, bucket, fake_s3)

        # first pass finds 1, second finds 0 due to dedup
        assert len(first) == 1
        assert len(second) == 0
        # only one upload done
        assert len(fake_s3.put_calls) == 1


