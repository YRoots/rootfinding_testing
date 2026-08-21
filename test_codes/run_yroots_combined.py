"""Benchmark yroots on the sparse and dense polynomial test suites.

Set the parameters in the Config class below, then run ``python benchmark_yroots.py``.
One script covers the three variants that used to live in separate files:

    suite="sparse", backend="multipower"   sparse terms expanded to a dense
                                           coefficient tensor, solved as MultiPower
    suite="sparse", backend="lambda"       sparse terms evaluated by a Python closure
    suite="dense",  backend="multipower"   dense coefficient tensors (Mathematica
                                           JSON or .npy), solved as MultiPower

For every degree in [mindeg, maxdeg] it solves num_tests systems of dim
polynomials on [-1, 1]^dim and writes, under the run's base directory
(``<result_dir>/dim{dim}/nonzero{nonzero}`` for sparse,
``<result_dir>/dim{dim}`` for dense):

    avg_times.txt              average solve time, one value per degree
    max_resids.txt             max |f_i(root)|, one value per degree
    avg_resids.txt             mean |f_i(root)|, one value per degree
    sum_resids.txt             sum of |f_i(root)|, one value per degree
    roots/roots_deg{deg}.json  {test number: roots} for that degree

The .txt files hold one plain value per line in degree order; the JSON files
match the layout the Julia benchmark writes, so the two can be diffed directly.
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Import yroots from this checkout instead of whatever is installed.
YROOTS_DIR = Path("/data1/acme/Roots/hyun/Rootfinding-serial")
if YROOTS_DIR.is_dir():
    sys.path.insert(0, str(YROOTS_DIR))

import yroots as yr  # noqa: E402
from yroots import MultiPower  # noqa: E402

SUMMARIES = ("avg_times", "max_resids", "avg_resids", "sum_resids")


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

@dataclass
class Config:
    suite: str = "dense"          # "sparse" | "dense"
    backend: str = "multipower"    # "multipower" | "lambda"  (lambda: sparse only)
    dim: int = 3
    mindeg: int = 2
    maxdeg: int = 20
    nonzero: int = 3               # sparse only: terms per polynomial
    num_tests: int = 300
    dense_format: str = "npy"     # dense only: "json" (Mathematica .txt) | "npy"
    coeff_dir: Path | None = None  # defaults per suite
    result_dir: Path | None = None
    append: bool = False           # True to add degrees to an existing run

    def __post_init__(self):
        if self.suite not in ("sparse", "dense"):
            raise ValueError(f"unknown suite: {self.suite}")
        if self.suite == "dense" and self.backend == "lambda":
            raise ValueError("the lambda backend expects sparse term lists")
        if self.coeff_dir is None:
            if self.suite == "sparse":
                self.coeff_dir = Path("../sparse/coeffs")
            elif self.suite == "dense":
                if self.dense_format == "json": 
                    self.coeff_dir = Path("../dense/coeffs_mathematica_json")
                else:
                    self.coeff_dir = Path("../dense/coeffs")
        if self.result_dir is None:
            self.result_dir = Path(f"../{self.suite}/results/yroots_results-optimize")
        self.coeff_dir = Path(self.coeff_dir)
        self.result_dir = Path(self.result_dir)

    @property
    def base_dir(self) -> Path:
        """Where this run's summaries and roots/ folder live."""
        d = self.result_dir / f"dim{self.dim}"
        return d / f"nonzero{self.nonzero}" if self.suite == "sparse" else d


# ----------------------------------------------------------------------------
# Loading coefficients
# ----------------------------------------------------------------------------

def load_degree(cfg, deg):
    """Load every test for one degree.

    sparse -> (n_tests, dim, n_terms, dim+1); the last axis holds an exponent
    per dimension followed by the coefficient.
    dense  -> (n_tests, dim, deg+1, ..., deg+1) coefficient tensors.
    """
    if cfg.suite == "sparse":
        return np.load(cfg.coeff_dir / f"dim{cfg.dim}" / f"deg{deg}" / f"num{cfg.nonzero}.npy")
    if cfg.dense_format == "json":
        with open(cfg.coeff_dir / f"dim{cfg.dim}_deg{deg}_randn.txt") as f:
            return np.array(json.load(f))
    return np.load(cfg.coeff_dir / f"dim{cfg.dim}_deg{deg}_randn.npy")


# ----------------------------------------------------------------------------
# Building the system for one test
# ----------------------------------------------------------------------------

def coeff_matrix(test, dim, deg):
    """Expand one test's sparse terms into a dense coefficient tensor per function."""
    coeffs = np.zeros((dim,) + (deg + 1,) * dim)
    exponents = test[:, :, :dim].astype(int)
    values = test[:, :, dim]
    for func in range(dim):
        for idx, value in zip(exponents[func], values[func]):
            coeffs[(func, *idx)] = value
    return coeffs


def poly_from_terms(terms, dim):
    """Build a closure evaluating one sparse polynomial, called as poly(*x)."""
    exps = np.array([list(map(int, t[:dim])) for t in terms], dtype=np.int64)
    coeffs = np.array([float(t[dim]) for t in terms], dtype=np.float64)

    def poly(*x):
        s = 0.0
        for k in range(exps.shape[0]):
            m = 1.0
            for j in range(dim):
                ej = exps[k, j]
                if ej != 0:
                    m *= x[j] ** ej
            s += coeffs[k] * m
        return s

    return poly


def build_funcs(cfg, test, deg):
    """Turn one test's coefficient block into a list of dim callables."""
    if cfg.backend == "lambda":
        return [poly_from_terms(c, cfg.dim) for c in test]
    if cfg.suite == "sparse":
        return [MultiPower(c) for c in coeff_matrix(test, cfg.dim, deg)]
    return [MultiPower(c) for c in test]


# ----------------------------------------------------------------------------
# Solving and scoring
# ----------------------------------------------------------------------------

def residuals(cfg, funcs, roots):
    """|f_i(root_j)| for every function i and root j, shape (len(funcs), len(roots))."""
    roots = np.asarray(roots)
    if roots.size == 0:
        return np.zeros((len(funcs), 0))
    if cfg.backend == "lambda":
        return np.array([[abs(f(*root)) for root in roots] for f in funcs])
    return np.array([np.abs(f(roots)) for f in funcs])


def solve_test(funcs, dim):
    """Solve one system on [-1, 1]^dim, returning the roots and the elapsed seconds."""
    start = time.perf_counter()
    roots = yr.solve(funcs, -np.ones(dim), np.ones(dim))
    return roots, time.perf_counter() - start


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

def write_roots_json(path, roots_by_test):
    """Write {test number: list of roots}, one root per line, sorted by test number."""
    with open(path, "w") as f:
        f.write("{\n")
        items = sorted(roots_by_test.items())
        for i, (test, roots) in enumerate(items):
            rows = ",\n".join("    " + json.dumps(root, separators=(",", ":"))
                              for root in np.asarray(roots).tolist())
            body = f"[\n{rows}\n  ]" if len(roots) else "[]"
            f.write(f'  "{test}": {body}')
            f.write(",\n" if i < len(items) - 1 else "\n")
        f.write("}\n")


def append_value(path, value):
    with open(path, "a") as f:
        f.write(f"{value:.17g}\n")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def run(cfg):
    base_dir = cfg.base_dir
    roots_dir = base_dir / "roots"
    roots_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = {name: base_dir / f"{name}.txt" for name in SUMMARIES}
    if not cfg.append:
        for path in summary_paths.values():
            path.write_text("")

    for deg in range(cfg.mindeg, cfg.maxdeg + 1):
        print(f"--- {cfg.suite} dim {cfg.dim} degree {deg}/{cfg.maxdeg} ---", flush=True)
        tests = load_degree(cfg, deg)
        reps = min(cfg.num_tests, len(tests))

        if deg == cfg.mindeg:
            # Warm up numba's JIT so it isn't charged to the first timed test.
            solve_test(build_funcs(cfg, tests[0], deg), cfg.dim)

        times, all_res, roots_by_test = [], [], {}
        for test in range(reps):
            if (test + 1) % 10 == 0:
                print(f"  test {test + 1}/{reps}", flush=True)

            funcs = build_funcs(cfg, tests[test], deg)
            roots, elapsed = solve_test(funcs, cfg.dim)

            times.append(elapsed)
            all_res.extend(residuals(cfg, funcs, roots).ravel())
            roots_by_test[test + 1] = roots

        write_roots_json(roots_dir / f"roots_deg{deg}.json", roots_by_test)

        res = np.array(all_res)
        append_value(summary_paths["avg_times"], np.mean(times) if times else np.nan)
        append_value(summary_paths["max_resids"], res.max() if res.size else np.nan)
        append_value(summary_paths["avg_resids"], res.mean() if res.size else np.nan)
        append_value(summary_paths["sum_resids"], res.sum() if res.size else np.nan)

    print("Program Finished")


if __name__ == "__main__":
    run(Config())
