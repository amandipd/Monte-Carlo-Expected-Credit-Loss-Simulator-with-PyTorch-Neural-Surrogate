# Resume & LinkedIn Bullet Points

**Project:** Distributed Monte Carlo Risk Engine  
**Context:** Portfolio proof-of-concept | General software engineering focus  
**Stack:** Python 3.13, NumPy, PyTorch, FastAPI, Redis, Docker, Ollama  
**Scale tested:** 50M-loan portfolio simulations  
**Hardware (benchmarks):** Intel Core Ultra 9 185H (16C/22T), 16 GB RAM, Windows 11  

> Metrics below were measured via `poetry run python scripts/benchmark_metrics.py --train` and supplemental scripts on this machine. See `results/benchmark_metrics.json` for raw data.

---

## Measured Results (Reference)

| Metric | Value |
|--------|-------|
| NumPy vectorization speedup (50M loans) | **~10×** (1.80s → 0.18s naive vs vectorized) |
| Redis distributed simulation (50M loans) | **0.29s** (10 job chunks; worker count not logged in original run) |
| Surrogate inference latency (median) | **0.43 ms** |
| Monte Carlo ground truth (500K loans / label) | **3.6 ms** mean |
| Surrogate vs 500K Monte Carlo | **~14×** faster |
| Surrogate vs 50M Monte Carlo (live run) | **~770×** faster (331 ms → 0.43 ms) |
| Surrogate validation MAE ratio | **0.45%** (gate: < 5%) |
| Engine spot-check error | **0.15%–1.7%** (gate: < 10%) |
| FastAPI `/predict` latency (median) | **2.6 ms** |
| FastAPI `/predict_shock` mock latency (median) | **4.2 ms** |
| Ollama live agent E2E (translate + infer + report) | **~21 s** (llama3.2, local) |

---

## Systems Engineering

1. **Architected a containerized quantitative risk platform** spanning Redis job queues, horizontally scalable simulation workers, and a FastAPI ML gateway with health checks and graceful cache degradation, **as measured by a 4-service Docker Compose stack processing 50M-loan simulation jobs split across 10 Redis-backed chunks**, by decomposing Monte Carlo workloads into producer/consumer orchestration with blocking result aggregation and timeout safeguards.

2. **Engineered a production-style REST API for credit loss forecasting** with Pydantic request validation, lifespan-managed model loading, and Redis-backed prediction caching (24h TTL), **as measured by 14 test modules (~55 cases) and sub-3 ms median latency on `/api/v2/predict`**, by building a FastAPI service exposing numeric and natural-language stress-testing endpoints with structured error handling.

3. **Integrated a local LLM agent layer for natural-language stress testing**, enabling analysts to submit crisis narratives instead of macro tuples, **as measured by an end-to-end Ollama pipeline completing scenario translation, inference, and executive report synthesis in ~21 seconds on llama3.2**, by orchestrating JSON-mode Ollama calls with mock-first fallbacks for CI and offline development.

4. **Resolved a Python module shadowing conflict** between a local `src/redis/` job-queue package and the pip `redis` client, **as measured by reliable ECL cache operation alongside distributed simulation workers in one monorepo**, by implementing dynamic import isolation so queue workers and inference caching coexist without import collisions.

---

## Performance Optimization

5. **Reduced Monte Carlo simulation runtime ~10× on a 50M-loan portfolio**, **as measured by wall-clock time dropping from 1.80s (naive loop) to 0.18s (NumPy vectorization) on identical macro inputs**, by replacing per-loan Python iteration with vectorized Bernoulli trials and hazard-rate PD computation in NumPy.

6. **Built a progressive simulation benchmark harness** comparing naive, vectorized, multicore, and Redis-distributed execution strategies, **as measured by automated benchmarking across four execution paths with consolidated reporting**, by centralizing ground-truth logic in a shared `ecl_engine` with interchangeable runners and reproducible result artifacts.

7. **Enabled distributed portfolio simulation via Redis job queues**, **as measured by completing a 50M-loan run in 0.29s with work split into 10 configurable job chunks (`N_JOBS=10`)**, by implementing a producer/consumer pattern with JSON job payloads and blocking result collection for horizontal worker scaling in Docker.

8. **Accelerated interactive stress testing from hundreds of milliseconds to sub-millisecond inference**, **as measured by reducing 50M-loan Monte Carlo latency from 331 ms to 0.43 ms median surrogate inference (~770×) while holding validation MAE ratio to 0.45%**, by training a PyTorch MLP (3→64→32→1) on 1,000 Latin Hypercube macro scenarios labeled with 500K-loan Monte Carlo runs.

---

## Data Architecture

9. **Built a synthetic labeling pipeline for ML surrogate training**, **as measured by 1,000 reproducible macro→ECL training rows with automated schema validation**, by combining SciPy Latin Hypercube sampling, Monte Carlo ground-truth labels, and pre-training dataset QA gates.

10. **Established dual validation gates tying the surrogate to the simulation engine**, **as measured by passing enforced thresholds of <5% validation MAE ratio and <10% relative error on live engine spot-checks (best spot-check: 0.15% error)**, by comparing predictions against both stored labels and fresh `compute_ecl()` recomputation at label-generation scale.

11. **Standardized feature/label scaling and model artifact versioning** for reproducible inference, **as measured by paired persistence of `surrogate_v1.pt` + `scaler_v1.pkl` with early-stopping checkpoint selection**, by applying `StandardScaler` to macro inputs and ECL outputs with joblib serialization consumed by a shared inference path across CLI and API.

12. **Centralized macro stress bounds and hazard-rate configuration** across simulation, training, and inference, **as measured by consistent clipped inputs across unemployment (2–15%), interest (0–12%), and HPI (70–130) ranges**, by driving all subsystems from a single env-backed `config.py` consumed by the engine, dataset generator, and FastAPI gateway.

---

## Recommended Top 5 (Single-Page Resume)

Pick these five for maximum impact on a **general software engineering** resume:

1. **Bullet 5** — Clear, quantified performance win (~10× vectorization).
2. **Bullet 8** — Strongest ML/systems story (~770× inference acceleration with 0.45% MAE).
3. **Bullet 2** — Demonstrates API/platform engineering with test coverage and latency.
4. **Bullet 1** — Shows distributed systems design (Redis, Docker, worker scaling).
5. **Bullet 3** — Differentiator: LLM agent layer for NL stress testing.

---

## LinkedIn Summary (Optional)

Built a portfolio proof-of-concept **Distributed Monte Carlo Risk Engine** — a Python platform that forecasts Expected Credit Loss on 50M-loan portfolios using vectorized and Redis-distributed Monte Carlo simulation, a PyTorch surrogate achieving **sub-millisecond inference** with **0.45% validation error**, and a FastAPI + Ollama agent layer for natural-language stress scenarios. Stack: Python, NumPy, PyTorch, FastAPI, Redis, Docker.

---

## Suggested Resume Entry Header

**Distributed Monte Carlo Risk Engine** | Personal Portfolio (Proof of Concept)  
Python · NumPy · PyTorch · FastAPI · Redis · Docker · Ollama

---

## Reproduce Metrics

```bash
# Full benchmark (generates/trains if artifacts missing)
poetry run python scripts/benchmark_metrics.py --train

# Live Ollama agent latency (requires Ollama + llama3.2)
poetry run python scripts/benchmark_ollama.py
```

Output: `results/benchmark_metrics.json`, `results/ollama_benchmark.txt`

---

## Open Item (Optional Follow-Up)

**Redis worker count:** The committed `redis_results.txt` (0.29s) does not log how many consumer processes were running. If you remember running e.g. `--scale worker=3`, we can add that to bullet 7 for extra precision.
