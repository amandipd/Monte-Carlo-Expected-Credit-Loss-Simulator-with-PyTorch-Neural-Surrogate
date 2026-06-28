# Distributed Monte Carlo Risk Engine
A quantitative risk system designed to execute Monte Carlo simulations on 10M+ loan portfolios to forecast Expected Credit loss under volatile economic conditions.

## Setup (Poetry)

**Prerequisites:** Python 3.13+, [Poetry](https://python-poetry.org/docs/#installation).

Install dependencies and create the virtual environment:

```bash
poetry install
```

(Optional) Activate the environment so you can run `python` directly:

```bash
poetry shell
```

## Running the simulations (local)

From the project root, run any of these with Poetry so the correct env is used:

```bash
# Naive loop (slow baseline) — writes results/naive_results.txt
poetry run python src/computations/loop_calc.py

# Vectorized NumPy (fast single-process) — writes results/vectorized_results.txt
poetry run python src/computations/vectorized_calc.py

# Multi-core parallel (uses all CPU cores) — writes results/multicore_results.txt
poetry run python src/computations/multicore_calc.py

# Run all three in sequence and merge into results/all_results.txt
poetry run python src/computations/run_all.py
```

**Redis (distributed workers):** Start Redis locally, then in one terminal run a consumer and in another run the producer:

```bash
# Terminal 1: start consumer(s)
poetry run python src/redis/consumer.py

# Terminal 2: run producer (pushes jobs, waits for results, writes results/redis_results.txt)
poetry run python src/redis/producer.py
```

If you're already in `poetry shell`, you can drop the `poetry run` prefix.

---

## Running with Docker

Docker runs Redis, the worker(s), and the producer in isolated **containers** so you don't need to install Redis or manage terminals by hand. Everything is defined in `Dockerfile` and `docker-compose.yml`.

**Prerequisites:** [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and **running** (you must see the Docker icon in your system tray; if `docker compose` says "cannot connect", start Docker Desktop and wait until it's ready).

### Docker commands (in order)

Run these from the **project root** (where `docker-compose.yml` lives).

| What you're doing | Command | What it does |
|-------------------|--------|---------------|
| **Build the app image** | `docker compose build` | Builds the Python image from the Dockerfile (installs deps, copies code). Do this once, or after changing code/dependencies. |
| **Start Redis + workers** | `docker compose up -d redis worker` | Starts Redis and one worker in the background (`-d` = detached). |
| **Scale to more workers** | `docker compose up -d --scale worker=3` | Runs 3 worker containers so jobs are processed in parallel. Combine with the line above or run after. |
| **Run the producer once** | `docker compose run --rm api` | Runs the producer in a one-off container: pushes jobs, waits for results, prints totals and time, then exits. `--rm` removes the container after. |
| **See worker logs** | `docker compose logs -f worker` | Streams logs from the worker(s). Useful to confirm "Connected to Redis. Waiting for jobs..." and job completion. `Ctrl+C` to stop following. |
| **See what's running** | `docker compose ps` | Lists running containers (redis, worker-1, worker-2, …). |
| **Stop everything** | `docker compose down` | Stops and removes the containers. Redis data in the container is discarded. |

### Quick start (copy-paste)

```bash
# 1. Build the image (first time or after code changes)
docker compose build

# 2. Start Redis and 3 workers in the background
docker compose up -d --scale worker=3

# 3. Wait a few seconds, then run the producer (you'll see output in this terminal)
docker compose run --rm api

# 4. When done, stop containers
docker compose down
```

### Optional: RedisInsight

With Redis running via Docker, open **http://localhost:8001** to use RedisInsight. You can inspect the `simulation_jobs` and `simulation_results` lists to see the queue in action.

### One-liner (start everything including api)

To start Redis, workers, and the producer in one go (producer runs once and exits; workers keep running until you stop them):

```bash
docker compose up --build --scale worker=3
```

The producer (`api`) will run once; Redis and workers stay up. Use `docker compose down` to stop.

### Surrogate ML API (with Redis ECL cache)

Train the model locally first so `models/surrogate_v1.pt` and `models/scaler_v1.pkl` exist, then:

```bash
docker compose build
docker compose up -d redis surrogate-api
```

The surrogate API listens on **http://localhost:8080**. Health check:

```bash
curl http://localhost:8080/health
```

Predict (repeat the same request to see `"cached": true` on the second call):

```bash
curl -X POST http://localhost:8080/api/v2/predict \
  -H "Content-Type: application/json" \
  -d '{"unemployment_rate": 6.5, "interest_rate": 5.25, "housing_price_index": 95.0}'
```

**Note:** The `api` service in `docker-compose.yml` is the **Monte Carlo simulation producer**, not the ML gateway. The ML gateway is `surrogate-api`.

For Ollama from inside Docker, the compose file points at `http://host.docker.internal:11434` (Ollama must be running on your host). Set `LLM_MOCK=true` in the `surrogate-api` environment to skip LLM calls in containers.

---

## Architecture & Roadmap (v2 Python Migration)
The initial prototype I created was in TypeScript (archived in branch v1-typescript-prototype), where I validated the core PD/LGD algorithms used to calculate ECL. In this main branch, I plan to implement a distributed architecture using Python, Numpy, and Redis. Instead of using the Node.js event loop, I plan to use vectorized multiprocessing to split the simulation tasks across CPU cores in hopes of significantly reducing calculation time.
