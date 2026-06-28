# Distributed Monte Carlo Risk Engine

A quantitative risk system that runs Monte Carlo simulations on large loan portfolios to forecast Expected Credit Loss (ECL) under macroeconomic stress — with a fast PyTorch surrogate, REST API, optional Ollama scenario translation, and Redis caching.

---

## Feature overview

| Capability | Status | How |
|------------|--------|-----|
| Vectorized / multicore Monte Carlo ECL | ✅ | `src/computations/` |
| Distributed simulation via Redis queue | ✅ | `src/redis/` + Docker `worker` / `api` |
| Synthetic training data generation | ✅ | `generate_training_data.py` |
| PyTorch ECL surrogate (ms inference) | ✅ | `train.py` → `models/surrogate_v1.pt` |
| Numeric REST API | ✅ | `POST /api/v2/predict` |
| Natural-language scenarios + report | ✅ | `POST /api/v2/predict_shock` + Ollama |
| Redis ECL result cache | ✅ | `ECL_CACHE_*` env vars |
| Docker surrogate API service | ✅ | `surrogate-api` in compose |

**MVP** (Phases 0–3): generate data → train model → numeric API.

**Full stack** (Phases 0–5): everything above, including LLM shock scenarios and Redis cache.

---

## Setup (Poetry)

**Prerequisites:** Python 3.13+, [Poetry](https://python-poetry.org/docs/#installation).

```bash
poetry install
cp .env.example .env   # then edit as needed
```

Copy `.env.example` to `.env` and adjust simulation size, Redis, Ollama, and cache settings.

---

## ML surrogate pipeline (local)

Run from the **project root**:

```bash
# 1. Generate labeled training CSV (Monte Carlo labels)
poetry run python src/ai_surrogate/generate_training_data.py

# 2. Validate dataset
poetry run python src/ai_surrogate/validate_dataset.py

# 3. Train surrogate → models/surrogate_v1.pt + models/scaler_v1.pkl
poetry run python src/ai_surrogate/train.py

# 4. Evaluate against validation gates
poetry run python src/ai_surrogate/evaluate.py

# 5. Start API
poetry run uvicorn ai_surrogate.app:app --app-dir src --reload --port 8080
```

### API endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| `GET` | `/health` | — | status + cache info |
| `POST` | `/api/v2/predict` | `{ unemployment_rate, interest_rate, housing_price_index }` | ECL + `cached` flag |
| `POST` | `/api/v2/predict_shock` | `{ scenario_description }` | macro coords, ECL, executive summary |

**Example (PowerShell):**

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8080/api/v2/predict" `
  -ContentType "application/json" `
  -Body '{"unemployment_rate": 6.5, "interest_rate": 5.25, "housing_price_index": 95.0}'

$r = Invoke-RestMethod -Method Post -Uri "http://localhost:8080/api/v2/predict_shock" `
  -ContentType "application/json" `
  -Body '{"scenario_description": "Severe recession with unemployment spiking"}'
$r.executive_summary
```

Interactive docs: **http://localhost:8080/docs**

### Ollama (free local LLM)

1. Install [Ollama](https://ollama.com) and run `ollama pull llama3.2`
2. In `.env`: `LLM_MOCK=false`, `OLLAMA_MODEL=llama3.2`
3. Restart the API

Set `LLM_MOCK=true` to skip Ollama (keyword-based mock translation + template report).

---

## Running Monte Carlo simulations (local)

```bash
# Naive loop — writes results/naive_results.txt
poetry run python src/computations/loop_calc.py

# Vectorized NumPy — writes results/vectorized_results.txt
poetry run python src/computations/vectorized_calc.py

# Multi-core parallel — writes results/multicore_results.txt
poetry run python src/computations/multicore_calc.py

# Run all three and merge into results/all_results.txt
poetry run python src/computations/run_all.py
```

**Redis distributed workers** (separate from ECL cache):

```bash
# Terminal 1
poetry run python src/redis/consumer.py

# Terminal 2
poetry run python src/redis/producer.py
```

---

## Testing

```bash
# Unit + integration tests (excludes live Redis)
poetry run python -m pytest -m "not integration"

# All tests including live Redis (skip if Redis is down)
poetry run python -m pytest

# Full pipeline: generate → train → evaluate → API smoke test
poetry run python scripts/validate_pipeline.py

# Windows: tests + pipeline in one command
./scripts/validate_pipeline.ps1
```

---

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `N_LOANS` | `100000000` | Portfolio size for Monte Carlo sims |
| `N_SAMPLES` | `1000` | Training dataset row count |
| `TRAINING_N_LOANS` | `500000` | Loans per label during data generation |
| `REDIS_HOST` | `localhost` | Redis host (queue + ECL cache) |
| `REDIS_PORT` | `6379` | Redis port |
| `ECL_CACHE_ENABLED` | `true` | Cache surrogate predictions in Redis |
| `ECL_CACHE_TTL` | `86400` | Cache TTL in seconds (24 h) |
| `LLM_MOCK` | `false` | Skip Ollama when `true` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local Ollama API |
| `OLLAMA_MODEL` | `llama3.2` | Model name for scenario/report |

See `.env.example` for macro bounds, hazard rate, and evaluation thresholds.

---

## Running with Docker

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) running.

### Monte Carlo queue (existing)

| Command | What it does |
|---------|--------------|
| `docker compose build` | Build Python image |
| `docker compose up -d --scale worker=3` | Redis + 3 simulation workers |
| `docker compose run --rm api` | Run simulation **producer** once |
| `docker compose down` | Stop containers |

The `api` service is the **simulation producer**, not the ML gateway.

### Surrogate ML API (with Redis ECL cache)

Train locally first so `models/surrogate_v1.pt` and `models/scaler_v1.pkl` exist:

```bash
docker compose build
docker compose up -d redis surrogate-api
curl http://localhost:8080/health
```

Repeat the same `POST /api/v2/predict` body to see `"cached": true` on the second call.

Ollama from Docker uses `http://host.docker.internal:11434` (host must run Ollama). Set `LLM_MOCK=true` on `surrogate-api` to disable LLM in containers.

### RedisInsight

With Redis running: **http://localhost:8001** — inspect `simulation_jobs` / `simulation_results` queues and `ecl_cache:*` keys.

---

## Architecture

```
Macro inputs → ecl_engine (Monte Carlo labels) → synthetic CSV
     ↓
Train PyTorch MLP + scalers → models/surrogate_v1.pt
     ↓
POST /api/v2/predict ──→ Redis cache? ──→ surrogate inference → ECL

POST /api/v2/predict_shock ──→ Ollama (scenario → macros) ──→ predict ──→ Ollama (report)
```

The original TypeScript prototype (branch `v1-typescript-prototype`) validated core PD/LGD math. This branch adds Python vectorization, Redis distribution, neural surrogate acceleration, and an agent layer for natural-language stress testing.
