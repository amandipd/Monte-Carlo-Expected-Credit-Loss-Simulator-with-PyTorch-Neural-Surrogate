"""
Runner script to execute all simulation variants:
- Naive loop implementation
- NumPy vectorized implementation
- Multicore / ProcessPoolExecutor implementation
"""
from datetime import datetime

from risk_engine.config import RESULTS_DIR
from risk_engine.monte_carlo.multicore_calc import run_parallel as run_multicore_simulation
from risk_engine.monte_carlo.vectorized_calc import run_simulation as run_vectorized_simulation
from risk_engine.monte_carlo.loop_calc import run_simulation as run_naive_simulation

def main():
    """Run all simulations (naive, vectorized, multicore)."""
    print("=" * 70)
    print("RUNNING NAIVE SIMULATION")
    print("=" * 70)
    print()
    run_naive_simulation()

    print()
    print()
    print("=" * 70)
    print("RUNNING VECTORIZED SIMULATION")
    print("=" * 70)
    print()
    run_vectorized_simulation()

    print()
    print()
    print("=" * 70)
    print("RUNNING MULTICORE SIMULATION")
    print("=" * 70)
    print()
    run_multicore_simulation()

    print()
    print("=" * 70)
    print("ALL SIMULATIONS COMPLETE")
    print("=" * 70)

    # Aggregate individual result files into a single summary file
    result_files = [
        (RESULTS_DIR / "naive_results.txt", "NAIVE SIMULATION RESULTS"),
        (RESULTS_DIR / "vectorized_results.txt", "VECTORIZED SIMULATION RESULTS"),
        (RESULTS_DIR / "multicore_results.txt", "MULTICORE SIMULATION RESULTS"),
    ]
    all_results_path = RESULTS_DIR / "all_results.txt"
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(all_results_path, "w", encoding="utf-8") as out:
            out.write(f"Run completed: {run_timestamp}\n")
            out.write("=" * 50 + "\n\n")
            for path, heading in result_files:
                out.write(f"{heading}\n")
                out.write("-" * len(heading) + "\n")
                try:
                    with open(path, "r", encoding="utf-8") as infile:
                        out.write(infile.read().rstrip())
                except FileNotFoundError:
                    out.write(f"(Missing file: {path})")
                out.write("\n\n")

        print(f"\nCombined results written to {all_results_path}")
    except OSError as e:
        print(f"\nFailed to write combined results file: {e}")

if __name__ == "__main__":
    main()
