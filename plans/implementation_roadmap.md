# ECL Neural Surrogate — Implementation Roadmap

Concrete, incremental checklist for adding a fast PyTorch surrogate on top of the existing Monte Carlo engine. Each step states **what to do**, **which files to touch**, and **why it belongs in that phase**.

---

## What exists today (baseline)

| Area | Current state | Reuse strategy |
|------|---------------|----------------|
| Simulation core | `PD = 1 - exp(-HAZARD_RATE * TIME_HORIZON)` over `N_LOANS` loans | Extract into a shared, parameterized function |
| Fast path | `src/computations/vectorized_calc.py` — NumPy vectorized | **Primary engine for synthetic data labels** |
| Parallel path | `src/computations/multicore_calc.py` — `simulation_chunk()` | Optional for large batch generation |
| Distributed path | `src/redis/producer.py` + `consumer.py` — job queue | Keep as-is; add separate Redis **cache** layer for inference |
| Config | `src/config.py` + `.env` | Extend with macro ranges, model paths, API keys |
| Docker | `docker-compose.yml` — Redis + worker + producer (`api`) | Add surrogate API service; keep existing queue workers |
| Tests | `pytest` in dev deps only; no `tests/` yet | Create from scratch in Phase 6 |

**Important gap:** The original plan assumes macro inputs (`unemployment_rate`, `interest_rate`, `housing_price_index`) map to a scalar **Expected Credit Loss**. The current engine outputs **default counts/rates** using a fixed `HAZARD_RATE`. Phase 0 closes that gap before any ML work.

---

## MVP definition (ship this first)

**MVP = Phase 0 + Phase 1 + Phase 2 + Phase 3**

A working end-to-end path where you can:

1. Generate a synthetic CSV of macro inputs → ECL labels using the real (refactored) simulation.
2. Train and save a small PyTorch MLP + feature scaler.
3. Call `POST /api/v2/predict` with **numeric** macro coordinates and get back a predicted ECL in milliseconds.

**Not in MVP:** Claude translation, executive report synthesis, Redis caching, Docker service for the API. Those are Phase 4–5.

---

## Phase 0 — Refactor simulation for parameterized ECL

**Goal:** One reusable function: `(macro_features, n_loans) → expected_credit_loss` that all downstream code calls.

### Step 0.1 — Define macro → hazard/ECL mapping

- [x] **What:** Add a pure function that maps 3 macro features to a hazard rate (or directly to PD), then to portfolio ECL. Start simple, e.g.:
  - `hazard_rate = base_hazard * f(unemployment, interest_rate, housing_price_index)`
  - `ECL = default_rate * avg_exposure * LGD` (use constants from config for exposure/LGD until loan-level data exists)
- [ ] **Files:**
  - **Create** `src/computations/ecl_engine.py` — `compute_ecl(unemployment, interest_rate, hpi, n_loans, seed=None) -> float`
  - **Modify** `src/config.py` — add `BASE_HAZARD_RATE`, `AVG_EXPOSURE`, `LGD`, macro sampling bounds
  - **Modify** `.env.example` — document new vars
- [ ] **Why here:** Without a parameterized ECL function, synthetic data and the surrogate have nothing meaningful to learn.

### Step 0.2 — Wire existing calculators to the shared engine

- [x] **What:** Refactor `vectorized_calc.py` and `multicore_calc.py` to call `ecl_engine.compute_ecl()` instead of inlining PD logic. Keep CLI behavior unchanged when macro args are omitted (use defaults from config).
- [ ] **Files:**
  - **Modify** `src/computations/vectorized_calc.py`
  - **Modify** `src/computations/multicore_calc.py` — update `simulation_chunk()` signature if needed
  - **Modify** `src/computations/loop_calc.py` — optional; keeps parity for small tests
- [ ] **Why here:** Proves the refactor works against known code paths before building ML on top.

### Step 0.3 — Smoke-test the refactored engine

- [x] **What:** Run vectorized sim with default config; confirm output file still written and numbers are sensible.
- [x] **Command:** `poetry run python src/computations/vectorized_calc.py`
- [x] **Validation:** `poetry run pytest tests/test_phase0_smoke.py`
- [ ] **Why here:** Catch regressions before investing in dataset generation.

---

## Phase 1 — Synthetic data generation

**Goal:** Build `data/synthetic_ecl_dataset.csv` by sampling macro inputs and labeling each row with `compute_ecl()`.

### Step 1.1 — Scaffold the AI surrogate package

- [x] **What:** Create package dirs and baseline deps (no ML training yet).
- [ ] **Files:**
  - **Create** `src/ai_surrogate/__init__.py`
  - **Create** `data/.gitkeep`, `models/.gitkeep`
  - **Modify** `pyproject.toml` — add `pandas`, `scikit-learn` (Phase 2 adds `torch`; Phase 3 adds `fastapi`, `uvicorn`, `anthropic`)
  - **Modify** `.gitignore` — ignore `data/synthetic_ecl_dataset.csv`, `models/*.pt`, `models/*.pkl` (keep `.gitkeep`)
- [ ] **Why here:** Establishes folder layout from the plan without pulling in heavy deps too early.

### Step 1.2 — Implement the parameter sampler

- [x] **What:** Sample `N_SAMPLES` rows from configured macro ranges (uniform or Latin-hypercube). Clip to safe bounds.
- [ ] **Files:**
  - **Create** `src/ai_surrogate/sampling.py` — `sample_macro_scenarios(n, rng) -> np.ndarray` shape `(n, 3)`
  - **Modify** `src/config.py` — `N_SAMPLES`, `MACRO_BOUNDS` dict
- [ ] **Why here:** Separates “how we pick inputs” from “how we label them.”

### Step 1.3 — Implement the data generation pipeline

- [x] **What:** For each sampled scenario, call `ecl_engine.compute_ecl()` with a **reduced** `N_LOANS` (e.g. 100k–1M) for speed. Write CSV.
- [ ] **Files:**
  - **Create** `src/ai_surrogate/generate_training_data.py` — CLI entry point
  - **Output** `data/synthetic_ecl_dataset.csv`
- [ ] **Schema:**

  | Column | Type |
  |--------|------|
  | `unemployment_rate` | float |
  | `interest_rate` | float |
  | `housing_price_index` | float |
  | `expected_credit_loss` | float |

- [ ] **Why here:** This is the training set the surrogate mimics; labels must come from the real engine, not guesses.

### Step 1.4 — Validate the dataset

- [x] **What:** Check row count, no NaNs, feature ranges, label distribution; spot-check 3 rows by re-running `compute_ecl()` manually.
- [ ] **Files:**
  - **Create** `src/ai_surrogate/validate_dataset.py` — prints summary stats, exits non-zero on failure
- [ ] **Command:** `poetry run python src/ai_surrogate/validate_dataset.py`
- [ ] **Why here:** Bad data silently produces a useless model.

---

## Phase 2 — PyTorch surrogate training pipeline

**Goal:** Train an MLP on the CSV, persist model weights + scaler, and verify inference locally.

### Step 2.1 — Add PyTorch dependency

- [x] **What:** `poetry add torch`
- [ ] **Files:** `pyproject.toml`, `poetry.lock`
- [ ] **Why here:** Keeps Phase 1 runnable without installing torch.

### Step 2.2 — Define the MLP architecture

- [x] **What:** 3-layer MLP: `3 → 64 → 32 → 1`, ReLU activations, single regression output.
- [ ] **Files:**
  - **Create** `src/ai_surrogate/model.py` — `ECLSurrogate(nn.Module)` with `forward(x)`
- [ ] **Why here:** Isolated, testable model definition.

### Step 2.3 — Build Dataset + training loop

- [x] **What:**
  - Load CSV with pandas
  - Split train/val (e.g. 80/20)
  - Fit `StandardScaler` on training features; save scaler alongside model
  - Train with MSE loss, Adam optimizer (~100–500 epochs or early stopping on val loss)
  - Log train/val MSE and MAE
- [ ] **Files:**
  - **Create** `src/ai_surrogate/train.py` — CLI: reads CSV, writes artifacts
  - **Create** `src/ai_surrogate/dataset.py` — PyTorch `Dataset` wrapper
  - **Output** `models/surrogate_v1.pt`, `models/scaler_v1.pkl`
- [ ] **Why here:** Offline training must complete before any API work.

### Step 2.4 — Add a local inference helper

- [x] **What:** Load model + scaler; `predict_ecl(unemployment, interest_rate, hpi) -> float`. Clip inputs to training bounds.
- [ ] **Files:**
  - **Create** `src/ai_surrogate/inference.py`
- [ ] **Why here:** Shared by CLI smoke tests, FastAPI (Phase 3), and unit tests (Phase 6).

### Step 2.5 — Training validation gate

- [x] **What:** Before moving on, confirm:
  - Val MAE is acceptably low relative to label range (define a threshold, e.g. < 5% of mean ECL)
  - Manual spot-check: 5 held-out rows, surrogate vs. `compute_ecl()` agree within tolerance
- [ ] **Files:**
  - **Create** `src/ai_surrogate/evaluate.py` — prints metrics, optional plot-free summary
- [ ] **Why here:** Prevents wiring a bad model into the API.

---

## Phase 3 — FastAPI inference endpoint (MVP completion)

**Goal:** HTTP API that accepts numeric macro coordinates and returns predicted ECL. No LLM, no Redis yet.

### Step 3.1 — Add FastAPI deps

- [ ] **What:** `poetry add fastapi uvicorn`
- [ ] **Files:** `pyproject.toml`, `poetry.lock`
- [ ] **Why here:** API layer depends on a trained model from Phase 2.

### Step 3.2 — Implement the API app

- [ ] **What:**
  - Load model + scaler once at startup (`lifespan` handler)
  - `POST /api/v2/predict` — body: `{ unemployment_rate, interest_rate, housing_price_index }`
  - Response: `{ input_macro_coordinates, predicted_ecl, inference_ms }`
  - Input validation via Pydantic; clip to safe ranges
- [ ] **Files:**
  - **Create** `src/ai_surrogate/app.py`
  - **Create** `src/ai_surrogate/schemas.py` — Pydantic request/response models
- [ ] **Why here:** This is the MVP deliverable — fast surrogate inference over HTTP.

### Step 3.3 — Run and manually test the API

- [ ] **Command:** `poetry run uvicorn ai_surrogate.app:app --app-dir src --reload --port 8080`
- [ ] **Test:** `curl -X POST http://localhost:8080/api/v2/predict -H "Content-Type: application/json" -d '{"unemployment_rate": 6.5, "interest_rate": 5.25, "housing_price_index": 95.0}'`
- [ ] **Why here:** Confirms end-to-end MVP before adding LLM/Redis complexity.

---

## Phase 4 — Claude agent translation + report synthesis

**Goal:** Accept natural-language crisis scenarios; return predicted ECL plus an executive summary.

### Step 4.1 — Add Anthropic SDK

- [ ] **What:** `poetry add anthropic`
- [ ] **Files:**
  - **Modify** `pyproject.toml`, `poetry.lock`
  - **Modify** `.env.example` — `ANTHROPIC_API_KEY=`
  - **Modify** `src/config.py` — load `ANTHROPIC_API_KEY`
- [ ] **Why here:** LLM is optional until numeric API works.

### Step 4.2 — Implement scenario → macro JSON translator

- [ ] **What:**
  - Define a Claude tool schema with 3 numeric fields
  - `translate_scenario(text) -> { unemployment_rate, interest_rate, housing_price_index }`
  - Clip/clamp outputs to configured bounds
  - Support a `--mock` mode for tests without API calls
- [ ] **Files:**
  - **Create** `src/ai_surrogate/agentic_translator.py`
  - **Create** `src/ai_surrogate/prompts.py` — system prompt + tool definition
- [ ] **Why here:** Separates LLM I/O from inference logic.

### Step 4.3 — Implement report synthesis

- [ ] **What:** Second Claude call (or same thread) that takes macro coords + predicted ECL and returns a short executive summary markdown string.
- [ ] **Files:**
  - **Create** `src/ai_surrogate/report_synthesizer.py`
- [ ] **Why here:** Keeps translation and narrative generation independently testable.

### Step 4.4 — Add the natural-language endpoint

- [ ] **What:**
  - `POST /api/v2/predict_shock` — body: `{ scenario_description: str }`
  - Flow: translate → predict → synthesize report
  - Response: `{ input_macro_coordinates, predicted_ecl, executive_summary }`
- [ ] **Files:**
  - **Modify** `src/ai_surrogate/app.py`
  - **Modify** `src/ai_surrogate/schemas.py`
- [ ] **Why here:** This is the full user-facing feature from the original plan.

---

## Phase 5 — Redis caching + Docker integration

**Goal:** Cache repeated scenarios in Redis; run the surrogate API alongside existing Redis queue infrastructure in Docker.

### Step 5.1 — Implement Redis ECL cache (separate from job queue)

- [ ] **What:**
  - Key: `ecl_cache:{round(u,2)}:{round(i,2)}:{round(hpi,2)}`
  - Value: JSON `{ predicted_ecl, timestamp }`
  - TTL: 86400 seconds (24 h)
  - On cache hit, skip model inference (still run report synthesis if requested)
- [ ] **Files:**
  - **Create** `src/ai_surrogate/cache.py`
  - **Modify** `src/ai_surrogate/app.py` — check/set cache in predict handlers
  - **Modify** `src/config.py` — `ECL_CACHE_TTL`, `ECL_CACHE_ENABLED`
- [ ] **Why here:** Reuses existing Redis (`src/config.py` already has `REDIS_HOST`/`REDIS_PORT`) without conflating cache keys with `simulation_jobs` / `simulation_results` queues.

### Step 5.2 — Update Docker for the surrogate API

- [ ] **What:**
  - Add `surrogate-api` service running uvicorn
  - Mount `models/` and optionally `data/` as volumes (or bake model into image for prod)
  - Pass `ANTHROPIC_API_KEY`, `REDIS_HOST=redis` via environment
  - Rename or document that existing `api` service is the **simulation producer**, not the ML API
- [ ] **Files:**
  - **Modify** `docker-compose.yml`
  - **Modify** `Dockerfile` — ensure `models/`, `data/` dirs exist; copy if baking artifacts in
- [ ] **Why here:** Docker changes depend on a working app from Phases 3–4.

### Step 5.3 — Docker smoke test

- [ ] **Commands:**
  ```bash
  docker compose build
  docker compose up -d redis surrogate-api
  curl -X POST http://localhost:8080/api/v2/predict ...
  ```
- [ ] **Why here:** Validates container wiring, env vars, and Redis connectivity.

---

## Phase 6 — Validation, tests, and documentation

**Goal:** Automated checks so refactors do not silently break the surrogate pipeline.

### Step 6.1 — Unit tests

- [ ] **What:**
  - `ecl_engine`: deterministic output given fixed seed
  - `model`: forward pass `(1,3) → (1,1)`
  - `inference`: clipped inputs stay in range
  - `cache`: key formatting is deterministic
- [ ] **Files:**
  - **Create** `tests/test_ecl_engine.py`
  - **Create** `tests/test_surrogate.py`
  - **Create** `tests/test_cache.py`
- [ ] **Command:** `poetry run pytest`
- [ ] **Why here:** Fast feedback on core logic without API keys or Redis.

### Step 6.2 — Integration tests

- [ ] **What:**
  - FastAPI `TestClient` against `/api/v2/predict` with known inputs
  - `agentic_translator` mock mode returns valid JSON
  - Optional: Redis integration test behind `@pytest.mark.integration` (skipped in CI without Redis)
- [ ] **Files:**
  - **Create** `tests/test_api.py`
  - **Create** `tests/test_agentic_translator.py`
  - **Create** `tests/conftest.py` — fixtures for model path, test client
- [ ] **Why here:** Catches wiring bugs between layers.

### Step 6.3 — End-to-end validation script

- [ ] **What:** Single script that runs: generate small dataset → train → evaluate → hit API locally.
- [ ] **Files:**
  - **Create** `scripts/validate_pipeline.sh` (or `.ps1` for Windows)
- [ ] **Why here:** One command to prove the full pipeline after changes.

### Step 6.4 — Update README

- [ ] **What:** Document MVP vs. full feature set, new commands, env vars, Docker service names.
- [ ] **Files:** **Modify** `README.md`
- [ ] **Why here:** Future-you should not need this roadmap to run the system.

---

## Dependency install schedule (by phase)

| Phase | Command |
|-------|---------|
| 1 | `poetry add pandas scikit-learn` |
| 2 | `poetry add torch` |
| 3 | `poetry add fastapi uvicorn` |
| 4 | `poetry add anthropic` |

Existing deps (`numpy`, `redis`, `python-dotenv`) stay unchanged.

---

## Suggested work order (at a glance)

```
Phase 0  Refactor engine for macro → ECL
   ↓
Phase 1  Generate synthetic_ecl_dataset.csv
   ↓
Phase 2  Train surrogate_v1.pt + scaler_v1.pkl
   ↓
Phase 3  ★ MVP: POST /api/v2/predict (numeric only)
   ↓
Phase 4  Claude translation + POST /api/v2/predict_shock
   ↓
Phase 5  Redis cache + Docker surrogate-api service
   ↓
Phase 6  Tests + README
```

---

## File tree after full implementation

```
.
├── data/
│   └── synthetic_ecl_dataset.csv          # generated; gitignored
├── models/
│   ├── surrogate_v1.pt                    # generated; gitignored
│   └── scaler_v1.pkl                        # generated; gitignored
├── scripts/
│   └── validate_pipeline.ps1
├── src/
│   ├── config.py                          # extended
│   ├── computations/
│   │   ├── ecl_engine.py                  # NEW — shared ECL core
│   │   ├── vectorized_calc.py             # modified — calls ecl_engine
│   │   ├── multicore_calc.py              # modified — calls ecl_engine
│   │   └── ...
│   ├── redis/                             # unchanged queue workers
│   └── ai_surrogate/                      # NEW
│       ├── generate_training_data.py
│       ├── sampling.py
│       ├── validate_dataset.py
│       ├── model.py
│       ├── dataset.py
│       ├── train.py
│       ├── evaluate.py
│       ├── inference.py
│       ├── cache.py
│       ├── agentic_translator.py
│       ├── report_synthesizer.py
│       ├── prompts.py
│       ├── schemas.py
│       └── app.py
├── tests/
│   ├── test_ecl_engine.py
│   ├── test_surrogate.py
│   ├── test_cache.py
│   ├── test_api.py
│   └── test_agentic_translator.py
├── docker-compose.yml                     # surrogate-api service added
└── Dockerfile                             # models/data dirs
```

---

## Risks and decisions to make early

1. **Macro → ECL formula:** Phase 0 needs a concrete mapping. Keep it simple and document it in `ecl_engine.py`; you can refine later without changing the surrogate pipeline shape.
2. **Label cost vs. accuracy:** Use reduced `N_LOANS` (e.g. 500k) for dataset generation; optionally re-label a small holdout set with full `N_LOANS` to measure label noise.
3. **Redis namespace:** Use `ecl_cache:*` keys only — do not reuse `simulation_jobs` / `simulation_results` list keys.
4. **Docker service naming:** Consider renaming `api` → `simulation-producer` in a follow-up to avoid confusion with `surrogate-api`.
