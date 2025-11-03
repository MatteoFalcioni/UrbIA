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

### 🐍 Sandboxed Code Execution
- Isolated Docker containers with 15+ data science libraries
- Persistent variables across executions within a thread
- Automatic artifact ingestion with SHA-256 deduplication
- Pre-installed: pandas, numpy, matplotlib, seaborn, scikit-learn, geopandas, folium

### 📊 Bologna OpenData Integration
- **6 API tools** for civic dataset discovery:
  - `list_catalog` - Search 100+ public datasets by keyword
  - `preview_dataset` - Preview first rows
  - `get_dataset_description` - Metadata and descriptions
  - `get_dataset_fields` - Field definitions
  - `is_geo_dataset` - Check for geographic data
  - `get_dataset_time_info` - Temporal coverage

### 🗺️ Geographic Visualization
- **Ortofoto Tools**: View and compare aerial photographs (2012-2023)
- **3D City Model**: Interactive 3D visualization of Bologna
- **Folium Integration**: Dynamic map generation with overlays
- **GeoParquet Support**: Native handling of geographic datasets

### 📁 S3-based Dataset Management

### 🎨 Modern UI
- React + TypeScript frontend with dark mode
- Real-time SSE streaming with tool execution visibility
- Inline artifact display (images, HTML maps, tables)
- Responsive design with TailwindCSS

---

## 🚀 Quick Start

1. Start RDS tunnel
```bash
~/start-rds-tunnel.sh
```
### 2. Set env vars (create .env in project root)

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

Below the older startup method with compose (we trashing that)

### Prerequisites
- Docker & Docker Compose
- OpenAI API key (required)
- Tavily API key (optional, for web search)
- Google Maps API key (optional, for geocoding)

### 1. Setup

```bash
# Clone or navigate to project
cd LG-Urban

# Create environment file
cat > .env << EOF
OPENAI_API_KEY=your_key_here
TAVILY_API_KEY=your_tavily_key  # optional
GOOGLE_MAPS_API_KEY=your_google_key  # optional

DEFAULT_MODEL=gpt-4o
DEFAULT_TEMPERATURE=0.7
CONTEXT_WINDOW=30000

DATASET_ACCESS=API
EOF

# Create Docker network
docker network create langgraph-network
```

### 2. Build Sandbox Image

```bash
docker build -f Dockerfile.sandbox -t sandbox:latest .
```

### 3. Start Application

```bash
docker compose up -d
```

Services will start:
- **Frontend**: http://localhost (port 80)
- **Backend API**: http://localhost:8000
- **Database Admin**: http://localhost:8080 (Adminer)
- **PostgreSQL**: port 5432

### 4. Access

Open http://localhost in your browser and start chatting!

---

## 📖 Usage Examples

### Data Analysis
```
"Show me active tattoo studios in Bologna and suggest where to open a new one"
→ Fetches elenco-esercizi dataset, filters by activity type, creates geographic heatmap
```

### Geographic Visualization
```
"Compare Giardini Margherita in 2017 and 2023"
→ Creates side-by-side ortofoto comparison map
```

### Custom Visualizations
```
"Load the bike-sharing dataset and create a time series of daily usage"
→ Fetches dataset, analyzes temporal patterns, generates matplotlib plot
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

### Storage Strategy

| Store | Contents | Purpose |
|-------|----------|---------|
| **PostgreSQL** | Threads, messages, configs, artifact metadata | Source of truth, queryable, relational |
| **PostgreSQL** | LangGraph checkpoints, agent state | Persistent, concurrent, scalable |
| **Blobstore** | Artifact bytes (images, CSVs, maps) | Content-addressed, deduplicated, scalable |

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `DEFAULT_MODEL` | `gpt-4o` | LLM model name |
| `DEFAULT_TEMPERATURE` | `0.7` | Sampling temperature |
| `CONTEXT_WINDOW` | `30000` | Max tokens before summarization |
| `DATASET_ACCESS` | `API` | Dataset mode: `NONE`, `API`, `LOCAL_RO`, `HYBRID` |
| `SESSION_STORAGE` | `TMPFS` | Sandbox storage: `TMPFS` (RAM) or `BIND` (disk) |
| `MAX_ARTIFACT_SIZE_MB` | `50` | Max file size per artifact |
| `CORS_ORIGINS` | `http://localhost,http://localhost:80` | Allowed origins |

### Dataset Access Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `NONE` | No datasets | Basic Python sandbox |
| `API` | Fetch from Bologna OpenData API | Default, flexible |
| `LOCAL_RO` | Mount local datasets read-only | Heavy/offline datasets |
| `HYBRID` | API + local heavy datasets | Best of both |

---

## 🛠️ Development

### Local Development (without Docker)

**Backend:**
```bash
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL
docker compose up -d db

# Run migrations
docker exec lg_urban_backend alembic upgrade head

# Start backend
uvicorn backend.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev  # http://localhost:5173
```

### Database Migrations

```bash
# Check current version
docker exec lg_urban_backend alembic current

# Apply migrations
docker exec lg_urban_backend alembic upgrade head

# Create new migration
docker exec lg_urban_backend alembic revision --autogenerate -m "description"
```

See [DB-README.md](./DB-README.md) for detailed database documentation.

---

## 📂 Project Structure

```
LG-Urban/
├── backend/
│   ├── app/              # API routes (threads, messages)
│   ├── db/               # SQLAlchemy models, Alembic migrations
│   ├── graph/            # LangGraph agent, tools, prompts
│   ├── sandbox/          # Docker sandbox manager
│   ├── artifacts/        # Blobstore storage & ingestion
│   ├── opendata_api/     # Bologna OpenData client
│   ├── dataset_manager/  # Dataset download & caching
│   └── tool_factory/     # LangChain tool creation
├── frontend/
│   └── src/
│       ├── components/   # React UI (MessageList, ArtifactCard, etc.)
│       ├── hooks/        # SSE streaming, API calls
│       ├── store/        # Zustand state management
│       └── types/        # TypeScript definitions
├── docker-compose.yml    # Full stack orchestration
├── Dockerfile.backend    # FastAPI + LangGraph
├── Dockerfile.frontend   # Nginx + React SPA
├── Dockerfile.sandbox    # Python sandbox environment
└── DB-README.md          # Database documentation
```

---

## 🐛 Troubleshooting

### "Network langgraph-network not found"
```bash
docker network create langgraph-network
```

### Backend can't spawn sandbox containers
Ensure Docker socket is mounted:
```yaml
volumes:
  - /var/run/docker.sock:/var/run/docker.sock
```

### Frontend can't connect to backend
Check CORS in `.env`:
```bash
CORS_ORIGINS=http://localhost,http://localhost:80,http://localhost:5173
```

### Database migration needed
```bash
docker exec lg_urban_backend alembic upgrade head
```

### Reset everything
```bash
docker compose down -v  # WARNING: Deletes all data
docker network create langgraph-network
docker build -f Dockerfile.sandbox -t sandbox:latest .
docker compose up -d
```

---

## 📚 Documentation

- **[DB-README.md](./DB-README.md)** - Database architecture & Alembic guide
- **[PLAN.md](./PLAN.md)** - Implementation plan & design decisions

---

## 🧪 Available Python Libraries in Sandbox

**Data Science:**
- pandas, numpy, scipy, statsmodels
- scikit-learn, scikit-image
- matplotlib, seaborn, plotly

**Geographic:**
- geopandas, shapely, folium, rasterio

**Formats:**
- pyarrow (Parquet/GeoParquet), openpyxl, xlrd

**Utilities:**
- requests, beautifulsoup4, pillow

---

## 📄 License

MIT

---

## 🙏 Credits

Fused from:
- **LG-App-Template** - Chat foundation with PostgreSQL
- **LangGraph-Sandbox** - Sandboxed code execution architecture

Special thanks to the Bologna Open Data initiative for providing rich civic datasets.
