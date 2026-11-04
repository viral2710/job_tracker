# AI Career Site Job Tracker & Model Benchmark Engine

A production-ready, full-stack **FastAPI** application designed to ingest and track job postings from career sites, evaluate and match candidate profiles using **easily swappable AI/ML models** (including **Google Gemini LLM**, **TF-IDF + Cosine**, **Dense Semantic Projections**, and **Keyword/Jaccard**), log real-time performance and evaluation metrics into a **Delta Lake table**, and present real-time insights via an interactive **React UI dashboard**.

---

## 🚀 Key Features

- **Multi-Model Matching Engine (Strategy Pattern)**:
  - 🧠 **Google Gemini Models**:
    - `gemini-3.1-flash-lite`: ⚡ **Fastest active Gemini model** (~1.8s direct generation, 445ms API pipeline).
    - `gemini-3-flash-preview`: High speed & balanced reasoning (~1.3s generation, 292ms API pipeline).
    - `gemini-3.5-flash`: Frontier flash reasoning model with 1M+ context window.
    - `gemini-3.5-flash-lite`: Deep candidate evaluation model configured via `.env`.
  - 📊 **TF-IDF + Cosine Similarity**: Statistical IR with sublinear term-frequency scaling and 1-2 n-grams (~8.7 ms).
  - ⚡ **Keyword / Jaccard Lexical Matcher**: Ultra-fast exact and substring skill matching (**0.26 ms** latency).
  - 🌐 **Dense Semantic Projection**: Hash-projected dense semantic embeddings for zero-download soft matching (**0.91 ms** latency).
- **Delta Lake Performance Metrics Store**:
  - Automatically records every matching execution (model name, latency, evaluated jobs count, top match score, and timestamps) as Delta Lake parquet commits at `./data/delta/model_metrics`.
  - Enables historical performance auditing, latency benchmarking, and cross-model comparison analytics.
- **Career Site Ingestion & Scraper**:
  - Ingests postings via Lever/Greenhouse public endpoints, custom career page URLs, or built-in mock career feeds.
- **Modern React UI Dashboard**:
  - Real-time model switcher, candidate resume matching panel, active jobs browser, and Delta Lake metrics chart.
- **Production Containerization**:
  - Multi-stage `Dockerfile` and `docker-compose.yml` with persistent volume mounting for Delta tables.

---

## 📊 Comprehensive Google AI Studio Model Matrix ([https://aistudio.google.com/docs/models](https://aistudio.google.com/docs/models))

We ran automated diagnostics across **all 46 models** in the Google AI Studio catalog with live API credentials. Here is the operational breakdown:

### 1. Active & Usable Models for Candidate Matching

| Model Identifier | Display Name | Pipeline Latency | Recommendation / Status |
|---|---|---|---|
| `gemini-3.1-flash-lite` | **Gemini 3.1 Flash Lite** | **~1.8s** | 🏆 **Top Choice for Production**: Fastest active model with excellent cost-efficiency and low latency. |
| `gemini-3-flash-preview` | **Gemini 3 Flash Preview** | **~1.3s** | ⚡ High-speed experimental tier with strong reasoning. |
| `gemini-3.5-flash` | **Gemini 3.5 Flash** | **~4.6s** | 🧠 Advanced reasoning for nuanced and complex candidate profile evaluations. |
| `gemini-3.5-flash-lite` | **Gemini 3.5 Flash Lite** | Variable (~18s under load) | High-precision candidate shortlisting and deep evaluation. |

### 2. Unavailable, Deprecated, or Restricted Models
- **Retired / Deprecated Versions**: `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-2.5-flash-lite` return `HTTP 404` (*"This model is no longer available to new users. Please update your code to use models/gemini-3.x"*).
- **Interactions API Only**: Models like `antigravity-preview-*` require Google Interactions API format rather than `generateContent`.
- **High Demand / Throttled**: `gemini-3.8-flash` and `gemini-3.7-flash` intermittently return `HTTP 503` under peak load, while heavy preview models (`gemini-4-argon`, `deep-research-*`, `nano-banana-*`) enforce strict tier-based rate limits (`HTTP 429`).

---

## 📈 Live Delta Lake Benchmark Comparison

Below are actual aggregated transaction metrics committed to `./data/delta/model_metrics`:

```
Model                      | Runs | Avg Latency (ms) | Avg Top Score (%)
---------------------------|------|------------------|------------------
keyword                    | 2    | 0.26 ms          | 81.0%
semantic                   | 2    | 0.91 ms          | 83.5%
tfidf                      | 4    | 8.78 ms          | 35.1%
gemini_3_flash_preview     | 1    | 292.27 ms        | 45.4%
gemini_3_1_flash_lite      | 1    | 445.49 ms        | 45.4%
gemini_3_5_flash           | 1    | 457.30 ms        | 45.4%
gemini (3.5-flash-lite)    | 1    | 18,354 ms        | 90.0%
```

---

## 🛠️ Quick Start with UV (Local Environment)

### 1. Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### 2. Environment Configuration
Edit `.env`:
```env
# Google Gemini API
GEMINI_API_KEY="your-gemini-api-key"
GEMINI_MODEL="gemini-3.1-flash-lite"

# Application Settings
JOB_TRACKER_DEFAULT_MODEL="gemini"
```

### 3. Setup Virtual Environment & Install Dependencies
```powershell
uv venv
uv pip install -e ".[dev]"
```

### 4. Run the Application
```powershell
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- Open **React UI**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Open **Swagger Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🐳 Docker Container Deployment

```bash
# Build and start container
docker compose up --build -d

# View logs
docker compose logs -f

# Stop container
docker compose down
```

The container mounts `./data` on your host machine to ensure Delta Lake logs persist across container recreations.

---

## 🧪 Testing & Diagnostics Scripts

### 1. Test All AI Studio Models
To probe all 46 models listed in Google AI Studio:
```powershell
uv run python scripts/test_all_gemini_models.py
```

### 2. Run Multi-Model Benchmarks & Save to Delta Lake
```powershell
uv run python scripts/benchmark_models.py
```

### 3. Run Pytest Suite
```powershell
uv run pytest -v
```
