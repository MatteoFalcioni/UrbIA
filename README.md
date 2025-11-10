# LG-Urban

**AI-powered urban data analysis platform for Bologna with sandboxed Python execution.**

Built on LangGraph, this production-ready application combines conversational AI with secure code execution, civic dataset integration, and geographic visualization tools.

---

## 🎯 Core Features

### 💬 Intelligent Conversations
- Multi-threaded chat with streaming responses
- Automatic context summarization (configurable window per thread)
- PostgreSQL-backed persistence with full message history
- Per-thread LLM configuration (model, temperature, system prompt)

### 🐍 Modal Sandbox Code Execution


### 📊 Bologna OpenData Integration
- **API tools** for civic dataset discovery:

### 🗺️ Geographic Visualization


### 📁 S3-based Dataset Management

### 🎨 Modern UI

---

## 🚀 Local Quick Start

clone the repo, then

### 2. Set env vars (create .env in project root from template)

### 3. Run backend
```bash
cd ~/LG-Urban
uvicorn backend.main:app --reload --port 8000
```
### 4. Run frontend (separate terminal)
```bash
cd frontend
npm run dev
```

---

## 🏗️ Architecture

```
┌─────────────┐
│  React UI   │  (SSE streaming, artifact display)
└──────┬──────┘
       │ HTTP/SSE
┌──────▼──────────────────────────┐
│  FastAPI Backend                │
│  ┌──────────────────────────┐   │
│  │  LangGraph Agent         │   │
│  │  ├─ Internet Search      │   │
│  │  ├─ Code Sandbox         │◄──┼─── Docker containers
│  │  ├─ Bologna OpenData API │   │
│  │  ├─ Datasets Management  │   │
│  │  └─ Geographic Tools     │   │
│  └──────────────────────────┘   │
└─────┬────────┬──────────┬───────┘
      │        │          │
  ┌───▼──┐  ┌──▼─────┐  ┌─▼───────┐
  │ PG   │  │  PG    │  │Blobstore│
  │ DB   │  │Checkpt │  │ (files) │
  └──────┘  └────────┘  └─────────┘
```
