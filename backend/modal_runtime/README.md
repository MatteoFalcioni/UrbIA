# Modal Runtime

In this folder you can find the implementation of the core funcionality of UrbIA: its data analysis capabilities. 

We assume that the ability to execute python code with context knowledge and complex reasoning can give rise to high data-analysis capabilities; therefore we give the agent the possibility to write and run Python code **in a sandbox environment**. The latter is managed by *Modal*. 

## Implementation Details

Following [Modal's documentation](\link) we first create an **app** in [app.py](./app.py)

```python
app = modal.App("lg-urban-executor")
```

and a **driver** in [driver.py](./driver.py). 

The purpose of this driver is to be always running while the sandbox exist, in order to keep it warm, and to redirect code input with stdin/stdout: it can do this since the driver is running *inside the sandbox*.

Furthermore, it also: 
- mantains the sandbox stateful by updating the `globals` dictionary with the local variables produced at each run;
- scans the work directory for produced artifacts and uploads them to S3 directly from inside the sandbox using `boto3`.

S3 specifics:
- The bucket name is read from the `S3_BUCKET` env var inside the sandbox.
- AWS credentials must be available in the sandbox (Modal Secret or env vars).
- Uploads use content-addressed keys: `output/artifacts/<sha256[:2]>/<sha256[2:4]>/<sha256>`.

We put everything together inside the [**SandboxExecutor**](./executor.py) class: at inizialition, it creates the sandbox with a mounted volume for persistence of the workspace during the run, and starts the driver. 

It only has 2 methods: 

- `execute` to execute a given code (through the driver);
- `terminate` to terminate the sandbox istance at session end;

## Tests

Inside the [tests/](./tests/) folder we wrote basic tests to check the implementation:

- `test_driver_artifacts.py`: unit tests for `scan_and_upload_artifacts` in the driver using a `FakeS3Client`.
  - Verifica upload e metadati (chiavi content-addressed, url, mime, size)
  - Verifica idempotenza/dedup alla seconda scansione

- `test_driver_e2e.py`: test end-to-end leggero che avvia `driver.py` in un subprocess e comunica via stdin/stdout.
  - Verifica persistenza dello stato con `exec(code, globals_dict)` (esempio: `x=41` poi `print(x+1)`)
  - Usa `ARTIFACTS_DIR` temporaneo per evitare upload a S3 durante il test

- `test_code_exec.py`: placeholder per futuri test del `SandboxExecutor` (instanziazione, esecuzione e terminate).  

### Run tests

```bash
cd backend/modal_runtime

# (opzionale) creare un venv e installare pytest
# python -m venv .venv && source .venv/bin/activate
# pip install -r ../requirements.txt pytest

# Eseguire tutti i test
pytest -q

# Eseguire solo i test del driver artifacts
pytest -q tests/test_driver_artifacts.py

# Eseguire solo l'e2e del driver
pytest -q tests/test_driver_e2e.py
```



