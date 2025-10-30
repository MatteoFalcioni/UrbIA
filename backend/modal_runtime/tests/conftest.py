import os
import sys
from pathlib import Path

here = Path(__file__).resolve()
repo_root = here.parents[3]

# Ensure imports like `backend.*` work regardless of where pytest is launched from
sys.path.insert(0, str(repo_root))

# Ensure CWD is repo root so relative paths in app.py (e.g. backend/modal_runtime/driver.py) resolve
try:
    os.chdir(repo_root)
except Exception:
    pass