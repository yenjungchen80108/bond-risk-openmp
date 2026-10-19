# Parallel Bond Risk: Yield-to-Maturity, Duration and Convexity with OpenMP

CS596 Scientific Computing and Visualization, final project (USC). Yen Jung (Emily) Chen.

Full title: _Parallelizing Yield-to-Maturity Solving and Interest-Rate Risk Analysis for Large Bond Portfolios_

## Abstract

**Problem.** Banks and funds hold portfolios of tens of thousands of bonds and must recompute every bond's yield to maturity (YTM) and its price sensitivity (duration and convexity) whenever interest rates move. **However, YTM has no closed-form solution, so every bond needs its own iterative solve, and it is unclear when the classic duration–convexity shortcut for repricing is still needed once parallel hardware makes full repricing cheap.**

**Finding (so far).** I implement the pipeline in C with OpenMP. At the same 10⁻⁸ tolerance, Newton–Raphson needs 4.4 iterations on average versus 27 for bisection, but is only about 1.3× faster in wall-clock time because each iteration is more expensive. The per-bond loop scales almost linearly, reaching a 7.9× speedup (98% efficiency) on 8 threads for 10⁶ bonds. _In progress:_ the comparison of exact repricing against the duration–convexity approximation across shock sizes and maturities.

**So what.** _In progress._ The goal is a clear rule for when the cheap approximation is accurate enough and when exact (parallel) repricing is worth its cost.

## Research question and expected results

Once parallel hardware makes repricing cheap, is the duration–convexity approximation (ΔP/P ≈ −D·Δy + ½·C·Δy²) still worth using?

Hypotheses to be tested (these are expectations, not results):

1. **Accuracy.** The approximation error is third order in the shock Δy and grows with maturity, so it should be small for small shocks and large for large shocks on long bonds.
2. **Cost.** For parallel shifts of the yield curve, portfolio-level duration and convexity can be aggregated once, so each extra scenario costs O(1). Exact repricing costs O(N·S) for N bonds and S scenarios.
3. **Parallelism.** Parallel speedup narrows, but does not remove, the cost gap between exact repricing and the aggregated approximation.

## Status

- [x] YTM by bisection and Newton–Raphson (same tolerance, iteration counts compared)
- [x] Duration and convexity by central finite differences
- [x] OpenMP parallel loop over bonds, baseline timing over N, speedup and efficiency over thread counts
- [x] Unit tests (`make test`)
- [ ] Interest-rate scenarios: exact repricing versus the duration–convexity approximation
- [ ] End-to-end timing (file I/O, Amdahl's law) and scheduling comparison (static, dynamic, guided)
- [ ] Written report (`report/`)
- [ ] GPU (CUDA) implementation: optional, depends on available hardware

## Repository layout

| File                           | Purpose                                                                                                               |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `1.ytm.c`, `1.ytm.h`           | Bond price, derivative, bisection and Newton–Raphson YTM solvers (with `_iter` variants that report iteration counts) |
| `2.duration_convexity.c`, `.h` | Duration and convexity by central finite differences                                                                  |
| `3.gen_portfolio.py`           | Generates a synthetic portfolio `bonds.csv` (each bond priced from a known yield)                                     |
| `3.portfolio.c`                | Reads `bonds.csv`, solves YTM, duration and convexity with OpenMP, writes `results.csv`                               |
| `4.plot_DC_curve.py`           | Plots portfolio duration/convexity and a price–yield curve                                                            |
| `5.convergence.c`              | Compares bisection and Newton–Raphson on the whole portfolio                                                          |
| `test_project.c`               | Unit tests (`make test`)                                                                                              |
| `Makefile`                     | Build and test targets                                                                                                |
| `report/`                      | Written report (in progress)                                                                                          |

`bonds.csv` and `results.csv` are generated files and are not tracked in git.

## Requirements

- A C compiler with OpenMP. macOS: `brew install libomp` (the Makefile uses clang with libomp). Linux: `gcc`.
- Python 3 with `numpy`, `pandas` and `matplotlib` (data generation and plots only).

## Build and run

```bash
make test                 # unit tests
make data N=100000        # generate bonds.csv with N bonds
make                      # build ./portfolio and ./convergence

OMP_NUM_THREADS=8 ./portfolio     # solve all bonds, prints time, writes results.csv
./convergence                     # bisection vs Newton on the whole portfolio
python3 4.plot_DC_curve.py        # figures from bonds.csv and results.csv
```

## Results so far

All numbers were measured on an Apple M4 Pro. Timers cover only the compute loop (no file I/O). Each value is the median of five runs.

**Sequential baseline** (`OMP_NUM_THREADS=1`)

| N         | time (s) | µs per bond |
| --------- | -------- | ----------- |
| 1,000     | 0.002257 | 2.26        |
| 10,000    | 0.022749 | 2.27        |
| 100,000   | 0.231128 | 2.31        |
| 1,000,000 | 2.216896 | 2.22        |

**Speedup and efficiency** (N = 1,000,000)

| threads | time (s) | speedup | efficiency |
| ------- | -------- | ------- | ---------- |
| 1       | 2.229205 | 1.00    | 100.0%     |
| 2       | 1.113167 | 2.00    | 100.1%     |
| 4       | 0.556026 | 4.01    | 100.2%     |
| 8       | 0.283055 | 7.88    | 98.4%      |

Values slightly above 100% are run-to-run noise (about 2%).

**Bisection vs Newton–Raphson** (N = 100,000, tolerance 1e-8 for both)

| method    | avg iterations | max iterations | time (s) | max abs error vs true YTM |
| --------- | -------------- | -------------- | -------- | ------------------------- |
| bisection | 27.00          | 27             | 0.185039 | 7.451e-09                 |
| newton    | 4.43           | 7              | 0.139508 | 1.407e-15                 |

Newton needs about 6× fewer iterations but is only about 1.3× faster in wall-clock time, because each iteration evaluates both the price and its derivative.

## Method notes

- Bond: annual level coupon `C`, face value `F`, `n` years, price `P(y) = Σ C/(1+y)^t + F/(1+y)^n`.
- YTM is the root of `f(y) = P(y) − P_market`. Bisection brackets the root in [0, 1]; Newton–Raphson starts from 5%.
- Duration and convexity use central differences with `h = 1e-4`:
  `D = (P₋ − P₊)/(2·P₀·h)`, `C = (P₋ + P₊ − 2·P₀)/(P₀·h²)`.
- The per-bond computations are independent, so the loop uses `#pragma omp parallel for schedule(dynamic)`; per-bond cost varies with maturity and iteration count.

## Limitations

- Synthetic portfolio, annual coupons, flat per-bond yield, no day-count conventions or accrued interest.
- Only parallel shifts of the yield are considered for the approximation.
- Timings are from one machine and one compiler configuration.
