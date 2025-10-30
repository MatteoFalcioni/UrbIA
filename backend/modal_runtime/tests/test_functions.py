import os
import base64
from dotenv import load_dotenv

from backend.modal_runtime.functions import app

def _check_modal_tokens() -> bool:
    load_dotenv()
    if not bool(os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET")):
        raise ValueError("Modal tokens not configured; skipping Modal integration tests")
    return True

_check_modal_tokens()

@app.local_entrypoint()  # used in order to run `modal run test_functions.py` from shell
def test_dataset_functions_flow_write_list_then_optional_export():

    from backend.modal_runtime.functions import write_dataset_bytes, list_available_datasets, export_dataset

    # Prepare a tiny CSV
    csv_data = b"a,b\n1,2\n3,4\n"
    b64 = base64.b64encode(csv_data).decode("ascii")


    print("Running in sandbox")
    
    # 1) Write into sandbox under /workspace/datasets/unit.csv
    res = write_dataset_bytes.remote(dataset_id="unit", data_b64=b64, ext="csv")
    assert res["dataset_id"] == "unit"
    assert res["rel_path"].endswith("datasets/unit.csv")
    assert res.get("columns") == ["a", "b"]
    assert res.get("shape") == [2, 2]

    # 2) List datasets should include our file (allow brief sync time)
    import time
    found = False
    names = set()
    for _ in range(10):  # up to ~5s
        files = list_available_datasets.remote()
        names = {f.get("path") for f in files}
        if "unit.csv" in names:
            found = True
            break
        time.sleep(0.5)
    assert found, f"unit.csv not found in listed datasets: {names}"

    # 3) Optionally export to S3 if creds and bucket are configured
    have_s3 = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY") and os.getenv("S3_BUCKET"))
    if have_s3:
        uploaded = export_dataset.remote("datasets/unit.csv", bucket=os.environ["S3_BUCKET"])
        assert uploaded.get("s3_key") and uploaded.get("s3_url")
