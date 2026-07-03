import json
import time

import redis

from risk_engine.config import REDIS_HOST, REDIS_PORT
from risk_engine.monte_carlo.multicore_calc import simulation_chunk

QUEUE_JOBS = "simulation_jobs"
QUEUE_RESULTS = "simulation_results"
POP_TIMEOUT = 1  # seconds; short so we can react to shutdown

# Retry loop so consumer waits instead of failing
def connect_with_retry(host=None, port=None, max_retries=10, retry_delay=2):
    host = host or REDIS_HOST
    port = port or REDIS_PORT
    for attempt in range(max_retries):
        try:
            r = redis.Redis(host=host, port=port, db=0)
            r.ping()
            return r
        except (redis.ConnectionError, redis.TimeoutError):
            if attempt < max_retries - 1:
                print(f"Connection failed (attempt {attempt + 1}/{max_retries}). Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Exiting.")
                raise

def main():
    r = connect_with_retry()
    print("Connected to Redis. Waiting for jobs...")

    while True:
        raw = r.blpop(QUEUE_JOBS, timeout=POP_TIMEOUT)
        if raw is None:
            continue
        _, job_json = raw
        job = json.loads(job_json)
        job_id = job["id"]
        load = job["load"]

        defaults = simulation_chunk(load)
        result = {"id": job_id, "defaults": int(defaults)}
        r.rpush(QUEUE_RESULTS, json.dumps(result))
        print(f"  Job {job_id} done: {defaults:,} defaults (load={load:,})")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nConsumer stopped.")

