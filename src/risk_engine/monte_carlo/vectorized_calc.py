import argparse
import time

from risk_engine.config import N_LOANS, RESULTS_DIR
from risk_engine.monte_carlo.ecl_engine import (
    portfolio_ecl_from_defaults,
    resolve_macro_inputs,
    simulate_defaults,
)

def run_simulation(
    unemployment: float | None = None,
    interest_rate: float | None = None,
    hpi: float | None = None,
    n_loans: int | None = None,
    seed: int | None = None,
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

    defaults = simulate_defaults(
        n_loans,
        unemployment=unemployment,
        interest_rate=interest_rate,
        hpi=hpi,
        seed=seed,
    )
    ecl = portfolio_ecl_from_defaults(defaults)

    elapsed_time = time.time() - start_time
    default_rate = defaults / n_loans

    results = {
        "defaults": defaults,
        "total_loans": n_loans,
        "default_rate": default_rate,
        "expected_credit_loss": ecl,
        "time_taken_seconds": elapsed_time,
        "method": "NumPy Vectorization",
    }

    print(f"Results: {defaults:,} defaults / {n_loans:,} total.")
    print(f"Expected Credit Loss: ${ecl:,.2f}")
    print(f"Time Taken: {elapsed_time:.4f} seconds (NumPy Vectorization)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / "vectorized_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Simulation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Loans: {n_loans:,}\n")
        f.write(f"Defaults: {defaults:,}\n")
        f.write(
            f"Default Rate: {default_rate:.6f} ({default_rate * 100:.4f}%)\n"
        )
        f.write(f"Expected Credit Loss: ${ecl:,.2f}\n")
        f.write(f"Time Taken: {elapsed_time:.4f} seconds\n")
        f.write(f"Method: {results['method']}\n")
        f.write(
            f"Macro Scenario: unemployment={unemployment:.2f}%, "
            f"interest={interest_rate:.2f}%, hpi={hpi:.2f}\n"
        )

    print(f"\nResults saved to {output_file}")
    return results

def _parse_args():
    parser = argparse.ArgumentParser(description="Run vectorized Monte Carlo ECL simulation.")
    parser.add_argument("--unemployment", type=float, default=None, help="Unemployment rate (%).")
    parser.add_argument("--interest", type=float, default=None, help="Interest rate (%).")
    parser.add_argument("--hpi", type=float, default=None, help="Housing price index.")
    parser.add_argument("--n-loans", type=int, default=None, help="Override N_LOANS from config.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    run_simulation(
        unemployment=args.unemployment,
        interest_rate=args.interest,
        hpi=args.hpi,
        n_loans=args.n_loans,
        seed=args.seed,
    )
