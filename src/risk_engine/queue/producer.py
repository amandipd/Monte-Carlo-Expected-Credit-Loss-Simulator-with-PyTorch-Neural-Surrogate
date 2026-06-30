"""
Producer: pushes simulation jobs to Redis. Workers (consumers) process them.
After pushing, waits for all results and prints the total defaults.
"""
import json
import time

import redis

from risk_engine.config import N_LOANS, N_JOBS, REDIS_HOST, REDIS_PORT, RESULTS_DIR

QUEUE_JOBS = "simulation_jobs"
QUEUE_RESULTS = "simulation_results"
RESULTS_TIMEOUT_SECONDS = 300

def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    chunk_size = N_LOANS // N_JOBS

    start_time = time.time()

    # Pushing N_JOBS onto redis list
    print(f"Pushing {N_JOBS} jobs (chunk size = {chunk_size:,} loans each)...")
    for i in range(1, N_JOBS + 1):
        job = {"id": i, "load": chunk_size}
        r.rpush(QUEUE_JOBS, json.dumps(job))
        print(f"  Pushed job {i}: load={chunk_size:,}")

    # Wait for all results from workers
    print("Waiting for results...")
    total_defaults = 0
    for _ in range(N_JOBS):
        # script pauses on this line and "blocks" further execution until something appears in simulation_results queue
        raw = r.blpop(QUEUE_RESULTS, timeout=RESULTS_TIMEOUT_SECONDS)
        if raw is None:
            raise TimeoutError(
                "Timed out waiting for a result from workers. "
                "Make sure consumers are running and processing jobs."
            )
        _, result_json = raw
        result = json.loads(result_json)
        total_defaults += result["defaults"]

    elapsed_time = time.time() - start_time
    default_rate = total_defaults / N_LOANS
    print(f"Results: {total_defaults:,} defaults / {N_LOANS:,} total.")
    print(f"Default rate: {default_rate:.6f} ({default_rate * 100:.4f}%)")
    print(f"Time Taken: {elapsed_time:.4f} seconds (Redis)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "redis_results.txt"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("Simulation Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total Loans: {N_LOANS:,}\n")
            f.write(f"Defaults: {total_defaults:,}\n")
            f.write(
                f"Default Rate: {default_rate:.6f} ({default_rate * 100:.4f}%)\n"
            )
            f.write(f"Time Taken: {elapsed_time:.4f} seconds\n")
            f.write("Method: Redis (distributed workers)\n")
        print(f"\nResults saved to {output_file}")
    except OSError as e:
        print(f"\nFailed to write results file: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProducer stopped.")
