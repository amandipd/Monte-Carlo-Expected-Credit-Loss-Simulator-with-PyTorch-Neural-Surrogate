import argparse
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from risk_engine.config import N_LOANS, RESULTS_DIR
from risk_engine.monte_carlo.ecl_engine import (
    pd_from_macro_inputs,
    portfolio_ecl_from_defaults,
    resolve_macro_inputs,
)

N_CORES = multiprocessing.cpu_count()

def simulation_chunk(
    n_sims: int,
    unemployment: float | None = None,
    interest_rate: float | None = None,
    hpi: float | None = None,
) -> int:
    """
    Run a smaller chunk of simulations using NumPy.
    This function will run on a separate CPU core.
    """
    pd = pd_from_macro_inputs(unemployment, interest_rate, hpi)
    random_rolls = np.random.random(n_sims)
    return int(np.count_nonzero(random_rolls < pd))

def run_parallel(
    unemployment: float | None = None,
    interest_rate: float | None = None,
    hpi: float | None = None,
    n_loans: int | None = None,
):
    n_loans = N_LOANS if n_loans is None else n_loans
    unemployment, interest_rate, hpi = resolve_macro_inputs(
        unemployment, interest_rate, hpi
    )

    start_time = time.time()
    print(f"Starting simulation for {n_loans:,} loans ...")
    print(
        f"Macro scenario: unemployment={unemployment:.2f}%, "
        f"interest={interest_rate:.2f}%, hpi={hpi:.2f}"
    )

    chunk_size = n_loans // N_CORES
    chunks = [chunk_size] * N_CORES

    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        results = list(
            executor.map(
                simulation_chunk,
                chunks,
                [unemployment] * N_CORES,
                [interest_rate] * N_CORES,
                [hpi] * N_CORES,
            )
        )

    total_defaults = sum(results)
    ecl = portfolio_ecl_from_defaults(total_defaults)
    elapsed_time = time.time() - start_time
    default_rate = total_defaults / n_loans

    results_dict = {
        "defaults": total_defaults,
        "total_loans": n_loans,
        "default_rate": default_rate,
        "expected_credit_loss": ecl,
        "time_taken_seconds": elapsed_time,
        "method": "Multicore (ProcessPoolExecutor)",
    }

    print(f"Results: {total_defaults:,} defaults / {n_loans:,} total.")
    print(f"Expected Credit Loss: ${ecl:,.2f}")
    print(f"Time Taken: {elapsed_time:.4f} seconds (Multicore)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "multicore_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Simulation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Loans: {n_loans:,}\n")
        f.write(f"Defaults: {total_defaults:,}\n")
        f.write(
            f"Default Rate: {default_rate:.6f} ({default_rate * 100:.4f}%)\n"
        )
        f.write(f"Expected Credit Loss: ${ecl:,.2f}\n")
        f.write(f"Time Taken: {elapsed_time:.4f} seconds\n")
        f.write(f"Method: {results_dict['method']}\n")
        f.write(
            f"Macro Scenario: unemployment={unemployment:.2f}%, "
            f"interest={interest_rate:.2f}%, hpi={hpi:.2f}\n"
        )

    print(f"\nResults saved to {output_file}")
    return results_dict

def _parse_args():
    parser = argparse.ArgumentParser(description="Run multicore Monte Carlo ECL simulation.")
    parser.add_argument("--unemployment", type=float, default=None, help="Unemployment rate (%).")
    parser.add_argument("--interest", type=float, default=None, help="Interest rate (%).")
    parser.add_argument("--hpi", type=float, default=None, help="Housing price index.")
    parser.add_argument("--n-loans", type=int, default=None, help="Override N_LOANS from config.")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_parallel(
        unemployment=args.unemployment,
        interest_rate=args.interest,
        hpi=args.hpi,
        n_loans=args.n_loans,
    )
