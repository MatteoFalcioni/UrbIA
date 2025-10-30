# LG-Urban Production Migration Plan

## Overview

Transition from Docker-based local development to scalable cloud infrastructure:

- **Modal.com**: Stateful serverless code execution (replaces Docker sandboxes)
- **AWS RDS**: Managed PostgreSQL (main app + LangGraph checkpoints)
- **AWS S3**: Single bucket with `input/` and `output/` prefixes for datasets and artifacts
- **Modal Volume**: Per-session workspace for isolated runtime environments, also handles ingestion pipeline (simplification)
- **Railway/Render**: Backend (FastAPI) + Frontend (React) hosting

**Philosophy**: Replace custom infrastructure with managed services. Simplify artifact pipeline (Modal handles all file operations). One Modal instance per session for true isolation.

---

## Phase 1: Replace Docker Sandbox with Modal

**Goal**: Replace Docker-in-Docker sandbox with Modal stateful functions sharing a persistent workspace.

Ingestion pipeline will be greatly simplified: Modal will manage all artifacts uploads to S3 (see later on, point 2.4)

### 1.1 Modal Setup & Configuration (DONE)

**Files to create**:

- `backend/modal_runtime/` (new directory)
- `backend/modal_runtime/__init__.py`
- `backend/modal_runtime/app.py` (Modal app definition)
- `backend/modal_runtime/executor.py` (stateful code execution class)
- `backend/modal_runtime/tools.py` (dataset & export tools)
- `backend/modal_runtime/requirements.txt` (Modal image dependencies)

**Key decisions**:

- Modal workspace mounted at `/workspace` (shared across all executions)
- Session state maintained in Modal class instance (`self.globals` dict for Python variables)
- Use `modal.Volume.from_name("lg-urban-workspace", create_if_missing=True)` for persistence

**Environment variables to add**:

```bash
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret
MODAL_WORKSPACE_NAME=lg-urban-workspace
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=eu-central-1
S3_BUCKET=lg-urban-prod  # Single bucket for both input and output
```

### 1.2 Build Modal Image with Dependencies (DONE)

**Create** `backend/modal_runtime/requirements.txt`:

```
pandas==2.2.0
numpy==1.26.3
matplotlib==3.8.2
seaborn==0.13.1
scikit-learn==1.4.0
geopandas==0.14.2
shapely==2.0.2
folium==0.15.1
plotly==5.18.0
pyarrow==15.0.0
boto3==1.34.34
openpyxl==3.1.2
pillow==10.2.0
requests==2.31.0
beautifulsoup4==4.12.3
```

**Create** `backend/modal_runtime/app.py`:

```python
import modal

# Define Modal image with all data science dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install_from_requirements("requirements.txt")

# Create Modal app
app = modal.App("lg-urban-executor")

# Create persistent volume for workspace
volume = modal.Volume.from_name("lg-urban-workspace", create_if_missing=True)
```

### 1.3 Implement Stateful Code Executor with Modal Sandbox (DONE)

**Modal approach**: Use `modal.Sandbox.create()` with a driver program for stateful execution.

**Create** `backend/modal_runtime/driver.py` (runs inside sandbox):

```python
import json
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import hashlib
import mimetypes

def driver_program():
    """Driver program that maintains Python state across executions."""
    globals_dict = {}  # Persistent namespace for user code
    
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
            
            # Detect new artifacts in /workspace/artifacts/
            # needs to be integrated with s3 bucket
            # ... 
            
            print(json.dumps({
                "stdout": stdout_io.getvalue(),
                "stderr": stderr_io.getvalue(),
                "artifacts": artifacts
            }), flush=True)
            
        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)

if __name__ == "__main__":
    driver_program()
```

**Create** `backend/modal_runtime/executor.py` (backend interface):

```python
import modal
import json
import boto3
import os

# Look up or create Modal app
app = modal.App.lookup("lg-urban-executor", create_if_missing=True)

# Define image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")  # are these installed every time and slow down app start? 
    .copy_local_file("driver.py", "/root/driver.py")  # Copy driver to image
)

class SandboxExecutor:
    """Manages per-session Modal Sandboxes."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        
        # Create per-session volume for persistent workspace
        self.volume = modal.Volume.from_name(
            f"lg-urban-session-{session_id}", 
            create_if_missing=True
        )
        
        # Create sandbox with volume mounted
        self.sandbox = modal.Sandbox.create(
            app=app,
            image=image,
            timeout=3600,  # 1 hour session timeout
            idle_timeout=600,  # 10 min idle timeout
            volumes={"/workspace": self.volume},
            workdir="/workspace"
        )
        
        # Start driver program with stdin/stdout pipes (why pipes? it was suggested by claude but idk)
        self.process = self.sandbox.exec(
            "python", "/root/driver.py",
            stdin=modal.PIPE,
            stdout=modal.PIPE
        )
    
    def execute(self, code: str, timeout: int = 120) -> dict:
        """Execute code and return results."""
        command = json.dumps({"code": code})
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        
        # Read response line
        result_line = self.process.stdout.readline()
        result = json.loads(result_line)
        
        # Upload artifacts to S3 using production-ready Volume access
        # Note: Files must be synced from sandbox to volume first (happens on exec completion)
        # to be sure about this volumes sync (and if it is the best approach) check Modal docs
        if result.get("artifacts"):
            s3_client = boto3.client('s3')
            bucket = os.getenv('S3_BUCKET', 'lg-urban-prod')
            
            for artifact in result["artifacts"]:
                # Read file from Volume (production-ready approach)
                # Files are automatically synced to volume when process completes
                file_bytes = b""
                for chunk in self.volume.read_file(artifact["path"].replace("/workspace/", "")):
                    file_bytes += chunk
                
                # Upload to S3 with content-addressed path
                # This is the replacement for the old ingestion pipeline - much needed simplification 
 
                s3_key = f"output/artifacts/{artifact['sha256'][:2]}/{artifact['sha256'][2:4]}/{artifact['sha256']}"
                s3_client.put_object(
                    Bucket=bucket,
                    Key=s3_key,
                    Body=file_bytes,
                    ContentType=artifact["mime"]
                )
                artifact["s3_key"] = s3_key
                artifact["s3_url"] = f"s3://{bucket}/{s3_key}"
        
        return result
    
    def terminate(self):
        """Clean up sandbox and persist volume."""
        self.process.wait()  # Wait for process to finish
        self.sandbox.terminate()
        self.sandbox.wait(raise_on_termination=False)  # Ensure volume syncs back all files
```

**Key implementation details**:

- One `modal.Sandbox` per session (true isolation)
- Driver program runs continuously, maintains `globals_dict` state
- Communication via JSON over stdin/stdout
- Artifacts auto-detected in `/workspace/artifacts/`
- Direct S3 upload from backend (sandbox returns artifact metadata only)

### 1.4 Implement Dataset Selection Tool (DONE)

**Create Modal function** in `backend/modal_runtime/tools.py`:

- `select_dataset(dataset_id: str, session_id: str) -> dict`
- Logic:

  1. Check if dataset exists in S3 bucket (the input part for heavy datasets)
  2. If yes: download from S3 to `/workspace/{dataset_id}.parquet` (how to input files into sandbox? need to check Modal docs...)
  3. If no: assume API dataset, fetch via Bologna OpenData API
  4. Save to `/workspace/{dataset_id}.parquet` (same as above: how exactly?)

- Return path in workspace

### 1.5 Implement Export Tool (DONE)

**Create Modal function** in `backend/modal_runtime/tools.py`:

- `export_dataset(workspace_path: str, session_id: str) -> dict`
- Read file from `/workspace/{path}`
- Upload to S3 bucket with timestamp prefix
- Return S3 presigned URL (24h expiry- or  maybe more)

### 1.6 Implement List Files Tool (DONE)

**Create Modal function** in `backend/modal_runtime/tools.py`:

- `list_datasets(session_id: str) -> list[dict]`
- List available datasets in S3 bucket together with files in `/workspace/` (Modal Volume)
- Return file metadata (name, size, modified time)

### 1.7 Refactor LangGraph Tools

**Modify** `backend/tool_factory/make_tools.py`:

- Replace `make_code_sandbox_tool()` to call Modal `CodeExecutor.execute()`
- Replace `make_select_dataset_tool()` to call Modal `select_dataset()`
- Replace `make_export_datasets_tool()` to call Modal `export_dataset()`
- Replace `make_list_datasets_tool()` to call Modal `list_datasets()`

**Key changes**:

- Remove Docker container dependencies
- Call Modal functions via Modal client: `executor.execute.remote(code)`
- Handle Modal authentication via `modal.Secret` or env vars
- Update artifact ingestion to use S3 keys instead of local paths

### 1.8 Create new prompt 

- Update/create prompt for analyst agent informing him about the tools and the details of how he needs to execute code.

---

## Phase 2: AWS S3 Storage Configuration

**Goal**: Replace local blobstore with S3 for artifacts and heavy datasets.

### 2.1 Create S3 Bucket (DONE)

**AWS Console / CLI**:

```bash
# Single bucket for both input and output
aws s3 mb s3://lg-urban-prod --region eu-central-1
aws s3api put-public-access-block --bucket lg-urban-prod --public-access-block-configuration "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
```

**Bucket structure**:

```
s3://lg-urban-prod/
├── input/datasets/           # Heavy datasets (admin uploads)
│   ├── dataset-1.parquet
│   └── dataset-2.parquet
└── output/artifacts/         # Agent-generated artifacts
    ├── ab/cd/abcd123...png
    └── ef/gh/efgh456...csv
```

**Lifecycle policies**:

- `output/artifacts/`: Transition to Glacier after 90 days, delete after 1 year
- `input/datasets/`: No lifecycle (permanent storage)

### 2.2 Upload Heavy Datasets  (DONE)

**One-time migration**:

```bash
# Upload existing heavy_llm_data/ to S3
aws s3 sync ./heavy_llm_data/ s3://lg-urban-datasets-prod/ --storage-class STANDARD
```

**Expected structure**:

```
s3://lg-urban-datasets-prod/
├── dataset-name-1.parquet
├── dataset-name-2.parquet
└── dataset-name-3.parquet
```

### 2.3 Configure IAM Permissions (DONE)

**Create IAM user**: `lg-urban-modal-service`

**Policy for Modal (attach to user)**:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::lg-urban-datasets-prod/*"
    },
    {
      "Effect": "Allow",
      "Action": ["s3:PutObject", "s3:GetObject"],
      "Resource": "arn:aws:s3:::lg-urban-artifacts-prod/*"
    }
  ]
}
```

**Create access keys**, save to Modal secrets:

```bash
modal secret create aws-credentials \
  AWS_ACCESS_KEY_ID=AKIA... \
  AWS_SECRET_ACCESS_KEY=... \
  AWS_REGION=eu-central-1
```

### 2.4 Simplify Artifact Ingestion Pipeline (DONE)

**Key simplification**: Modal handles all file operations. Backend only stores metadata.

**New flow**:

1. Modal executor generates artifact → uploads to S3 → returns metadata (S3 key, sha256, mime, size)
2. Backend receives metadata → inserts into Postgres `artifacts` table
3. No file handling on backend at all

**Modify** `backend/artifacts/ingest.py`:

- **Simplify** `ingest_files()` to accept artifact metadata dict (not file paths):
  ```python
  async def ingest_artifact_metadata(
      session: AsyncSession,
      thread_id: uuid.UUID,
      s3_key: str,
      sha256: str,
      filename: str,
      mime: str,
      size: int,
      session_id: str,
      tool_call_id: str
  ) -> Artifact:
      """Insert artifact metadata (file already in S3)."""
      artifact = Artifact(
          thread_id=thread_id,
          sha256=sha256,
          filename=filename,
          mime=mime,
          size=size,
          session_id=session_id,
          tool_call_id=tool_call_id,
          meta={"s3_key": s3_key}  # Store S3 location
      )
      session.add(artifact)
      await session.flush()
      return artifact
  ```

- **Remove**: `copy_to_blobstore()`, `file_sha256()`, `safe_delete_file()` - all file operations
- **Keep**: Database operations only

**Modify** `backend/artifacts/storage.py`:

- **Replace** `blob_path_for_sha()` with `s3_key_for_artifact()`:
  ```python
  def s3_key_for_artifact(artifact: Artifact) -> str:
      """Get S3 key from artifact metadata."""
      return artifact.meta.get("s3_key") or f"output/artifacts/{artifact.sha256[:2]}/{artifact.sha256[2:4]}/{artifact.sha256}"
  ```

- **Remove**: All filesystem operations (`Path`, `copy`, `delete` functions)
- **Keep**: Database query helpers (`get_artifact_by_id`, `find_artifact_by_sha`)

### 2.5 Implement S3 Presigned URLs (DONE)

**Modify** `backend/artifacts/api.py`:

- Replace `FileResponse` with S3 presigned URL generation
- `download_artifact()` should call `boto3.client('s3').generate_presigned_url()`
- Set expiration to 24 hours (configurable via env var)
- Return redirect response to presigned URL

**Add helper function** in `backend/artifacts/storage.py`:

```python
def generate_artifact_download_url(artifact: Artifact, expiry_seconds: int = 86400) -> str:
    """Generate presigned S3 URL for artifact download."""
    s3_client = boto3.client('s3')
    s3_key = f"{artifact.sha256[:2]}/{artifact.sha256[2:4]}/{artifact.sha256}"
    return s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': os.getenv('S3_ARTIFACTS_BUCKET'), 'Key': s3_key},
        ExpiresIn=expiry_seconds
    )
```

---

## Phase 3: Migrate PostgreSQL to AWS RDS

**Goal**: Move PostgreSQL from Docker to managed AWS RDS.

### 3.1 Provision RDS Instance

**Recommended configuration**:

- Engine: PostgreSQL 16
- Instance class: `db.t4g.micro` (development), `db.t4g.small` (production)
- Storage: 20 GB GP3 (auto-scaling enabled)
- Multi-AZ: No (dev), Yes (prod)
- Backup retention: 7 days
- Enhanced monitoring: Enabled
- Public accessibility: No (access via VPC only)

**AWS Console steps**:

1. RDS → Create database → PostgreSQL 16
2. Template: Dev/Test (or Production)
3. DB instance identifier: `lg-urban-prod`
4. Master username: `postgres`
5. Master password: (generate strong password, save to secrets manager)
6. VPC: Default or custom
7. Security group: Create new (allow port 5432 from backend IPs)
8. Database name: `chat`

**Connection string format**:

```
postgresql+asyncpg://postgres:{password}@lg-urban-prod.xxxxx.eu-central-1.rds.amazonaws.com:5432/chat
```

### 3.2 Database Migration

**Export from local Docker**:

```bash
docker exec lg_urban_db pg_dump -U postgres -d chat -F c -b -v -f /tmp/chat_dump.backup
docker cp lg_urban_db:/tmp/chat_dump.backup ./chat_dump.backup
```

**Import to RDS**:

```bash
pg_restore -h lg-urban-prod.xxxxx.eu-central-1.rds.amazonaws.com \
  -U postgres -d chat -v ./chat_dump.backup
```

**Run Alembic migrations** (ensure schema is up-to-date):

```bash
DATABASE_URL="postgresql+asyncpg://postgres:{password}@lg-urban-prod.xxxxx.rds.amazonaws.com:5432/chat" \
  alembic upgrade head
```

### 3.3 Update Application Configuration

**Modify** `backend/config.py` or `.env`:

```bash
DATABASE_URL=postgresql+asyncpg://postgres:{password}@lg-urban-prod.xxxxx.rds.amazonaws.com:5432/chat
ALEMBIC_DATABASE_URL=postgresql+psycopg2://postgres:{password}@lg-urban-prod.xxxxx.rds.amazonaws.com:5432/chat
LANGGRAPH_CHECKPOINT_DB_URL=postgresql://postgres:{password}@lg-urban-prod.xxxxx.rds.amazonaws.com:5432/chat
```

**Security best practices**:

- Store password in environment variable or secrets manager
- Use IAM authentication for RDS (optional, advanced)
- Enable SSL/TLS connections: `?sslmode=require` in connection string

### 3.4 Test Database Connectivity

**Create test script** `backend/scripts/test_db.py`:

```python
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from backend.db.models import Base

async def test_connection():
    engine = create_async_engine(os.getenv("DATABASE_URL"))
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT version()"))
        print(result.scalar())
    await engine.dispose()

asyncio.run(test_connection())
```

---

## Phase 4: Deploy Backend & Frontend to Cloud

**Goal**: Deploy FastAPI backend and React frontend to production hosting.

### 4.1 Choose Hosting Platform

**Recommended**: **Railway** (simplicity) or **AWS ECS/Fargate** (full control)

**Railway advantages**:

- Zero-config deployments from GitHub
- Automatic HTTPS, environment variables UI
- Built-in monitoring, logs
- Generous free tier, pay-per-use scaling

**AWS ECS advantages**:

- Full control over infrastructure
- Better integration with other AWS services
- More complex setup, higher operational overhead

**Decision**: Use Railway for MVP, migrate to ECS if scaling needs arise.

### 4.2 Backend Deployment (Railway)

**Steps**:

1. Create Railway account, connect GitHub repository
2. Create new project: "LG-Urban Backend"
3. Add service from repo: select `/backend`
4. Configure build:

   - **Build command**: (auto-detected from Dockerfile.backend)
   - **Start command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`

5. Add environment variables:
   ```
   DATABASE_URL=postgresql+asyncpg://...
   LANGGRAPH_CHECKPOINT_DB_URL=postgresql://...
   OPENAI_API_KEY=...
   MODAL_TOKEN_ID=...
   MODAL_TOKEN_SECRET=...
   S3_ARTIFACTS_BUCKET=lg-urban-artifacts-prod
   S3_DATASETS_BUCKET=lg-urban-datasets-prod
   AWS_ACCESS_KEY_ID=...
   AWS_SECRET_ACCESS_KEY=...
   AWS_REGION=eu-central-1
   CORS_ORIGINS=https://your-frontend-domain.com
   ```

6. Deploy (Railway auto-deploys on push to main)

**Note**: Remove Docker socket mount from production (no sandboxes to spawn)

### 4.3 Frontend Deployment (Vercel)

**Steps**:

1. Create Vercel account, import GitHub repository
2. Configure project:

   - **Framework**: Vite (React)
   - **Root directory**: `frontend`
   - **Build command**: `npm run build`
   - **Output directory**: `dist`

3. Add environment variable:
   ```
   VITE_API_URL=https://lg-urban-backend.up.railway.app
   ```

4. Deploy (Vercel auto-deploys on push to main)

**Alternative**: Netlify (similar process)

### 4.4 Configure CORS & Networking

**Update backend** `backend/main.py`:

```python
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**Set in Railway**:

```
CORS_ORIGINS=https://your-app.vercel.app,https://lg-urban.com
```

### 4.5 End-to-End Production Test

**Test scenarios**:

1. Create new thread via frontend
2. Send message, verify agent response streams correctly
3. Execute code via code_sandbox tool:

   - Verify Modal function is called
   - Check artifact generated and uploaded to S3
   - Confirm presigned URL works for download

4. Select dataset (API mode):

   - Fetch from Bologna OpenData
   - Verify saved to Modal Volume (`/workspace/`)

5. Select heavy dataset:

   - Download from S3_DATASETS_BUCKET to Modal Volume
   - Execute code using the dataset

6. Export modified dataset:

   - Upload to S3_ARTIFACTS_BUCKET
   - Return presigned URL

7. Verify message history persists in RDS

---

## Phase 5: Cleanup & Decommissioning

**Goal**: Remove legacy Docker infrastructure and local development artifacts.

### 5.1 Remove Docker Files

**Delete**:

- `docker-compose.yml`
- `Dockerfile.sandbox`
- `backend/sandbox/` (entire directory)
- `backend/dataset_manager/` (if fully replaced by Modal)
- `start.sh`
- `.dockerignore`

**Keep**:

- `Dockerfile.backend` (for Railway/ECS deployments)
- `Dockerfile.frontend` (for Railway/ECS deployments)

### 5.2 Archive Local Data

**Backup before deletion**:

```bash
# Database dump (if not already migrated)
docker exec lg_urban_db pg_dump -U postgres chat > backup_$(date +%Y%m%d).sql

# Local blobstore
tar -czf blobstore_backup_$(date +%Y%m%d).tar.gz blobstore/

# Heavy datasets (already in S3, but keep local backup)
tar -czf heavy_data_backup_$(date +%Y%m%d).tar.gz heavy_llm_data/
```

**Delete local volumes**:

```bash
docker-compose down -v
docker volume prune -f
rm -rf blobstore/ sessions/
```

### 5.3 Update Documentation

**Modify** `README.md`:

- Remove Docker setup instructions
- Add production deployment guide (Railway, Modal, AWS)
- Update architecture diagram (remove Docker, add Modal + S3)
- Add environment variables reference for production

**Update** `DB-README.md`:

- Change database examples to use RDS endpoint
- Remove Docker-specific connection instructions

---

## Phase 6: Security, Monitoring & Hardening

**Goal**: Production-grade security, observability, and cost management.

### 6.1 Security Hardening

**Backend security**:

- Enable HTTPS only (enforced by Railway/Vercel)
- Rotate database credentials quarterly
- Use AWS Secrets Manager for sensitive env vars
- Enable RDS encryption at rest
- Restrict RDS security group to backend IP only
- Enable S3 bucket versioning (for artifact recovery)
- Configure S3 bucket policies (deny public access)

**Modal security**:

- Rotate Modal API tokens quarterly
- Use Modal Secrets for all AWS credentials
- Enable Modal function timeout limits (prevent runaway executions)
- Monitor Modal usage for suspicious patterns

### 6.2 Monitoring & Logging

**Backend monitoring**:

- Integrate Sentry for error tracking
- Add structured logging (JSON format)
- Set up uptime monitoring (UptimeRobot, Pingdom)
- Configure Railway logs forwarding to Logtail

**Database monitoring**:

- Enable RDS Enhanced Monitoring
- Set CloudWatch alarms: CPU > 80%, storage < 10% free
- Monitor slow queries (pg_stat_statements)

**Modal monitoring**:

- Track execution times and failures in Modal dashboard
- Set up alerts for high Modal costs
- Monitor workspace volume usage

### 6.3 Cost Optimization

**AWS cost controls**:

- Set billing alerts at $50, $100, $200/month
- Use S3 Lifecycle policies (move old artifacts to Glacier)
- Consider RDS Reserved Instances (1-year commitment) for prod
- Use AWS Cost Explorer to track spending by service

**Modal cost controls**:

- Set per-function timeout limits (prevent infinite loops)
- Monitor CPU/memory usage, optimize image size
- Use Modal's cold start optimization

**Railway cost controls**:

- Monitor usage dashboard
- Scale down backend during off-peak hours (if usage allows)

### 6.4 Backup & Disaster Recovery

**RDS backups**:

- Automated daily backups (7-day retention)
- Manual snapshot before major schema changes
- Test restore process quarterly

**S3 backups**:

- Enable versioning on both buckets
- Configure cross-region replication (optional, for critical data)

**Modal workspace backups**:

- Periodic exports of workspace volume to S3 (cron job)
- Automated backup script:
  ```python
  # backend/scripts/backup_modal_workspace.py
  import modal
  import boto3
  
  volume = modal.Volume.from_name("lg-urban-workspace")
  # Copy volume contents to S3 backup bucket
  ```


---

## Implementation Order & Dependencies

**Phase 1 (Modal)**: 2-3 weeks

- **Critical path**: Modal executor, dataset tools, LangGraph integration
- **Blockers**: None (can be developed in parallel with current system)

**Phase 2 (S3)**: 1 week

- **Critical path**: Bucket setup, artifact pipeline refactor, presigned URLs
- **Blockers**: Depends on Phase 1 (Modal needs S3 for artifacts)

**Phase 3 (RDS)**: 3-5 days

- **Critical path**: RDS provisioning, migration, connection testing
- **Blockers**: None (can be done in parallel with Phase 1-2)

**Phase 4 (Deployment)**: 3-5 days

- **Critical path**: Railway setup, environment configuration, DNS
- **Blockers**: Depends on Phases 1-3 (all services must be ready)

**Phase 5 (Cleanup)**: 1-2 days

- **Critical path**: Docker removal, documentation updates
- **Blockers**: Depends on Phase 4 (production must be stable)

**Phase 6 (Hardening)**: Ongoing

- **Critical path**: Monitoring setup, security audits
- **Blockers**: None (can start immediately after Phase 4)

**Total estimated time**: 4-6 weeks (full-time equivalent)

---

## Rollback Plan

**If Modal migration fails**:

- Revert `backend/tool_factory/make_tools.py` to Docker version
- Keep Docker Compose setup until Modal is stable
- Run hybrid: Docker for code execution, S3 for storage

**If RDS migration fails**:

- Restore local Docker PostgreSQL from backup
- Update `DATABASE_URL` to local instance
- Investigate RDS issues (connectivity, performance)

**If deployment fails**:

- Use Railway rollback feature (one-click revert)
- Restore previous environment variables
- Check logs for configuration issues

---

## Success Metrics

**Performance**:

- Code execution latency < 5s (p95)
- API response time < 200ms (p95)
- Artifact upload time < 2s for 10MB file

**Reliability**:

- Uptime > 99.5%
- Zero data loss incidents
- RDS failover time < 60s (if Multi-AZ enabled)

**Cost**:

- Total monthly cost < $200 (dev), < $500 (prod with traffic)
- Modal costs < $50/month
- AWS costs < $150/month (RDS + S3 + data transfer)

**Security**:

- Zero exposed credentials in logs
- All data encrypted at rest and in transit
- Security audit passed (no critical vulnerabilities)