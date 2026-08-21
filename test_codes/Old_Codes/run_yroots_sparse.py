"""Benchmark yroots on the sparse polynomial test suite.

For every degree in the range below, this solves num_tests systems of dim
sparse power basis polynomials on [-1, 1]^dim and writes, under
``<result_dir>/dim{dim}/nonzero{nonzero}/``:

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
from pathlib import Path

import numpy as np

# Import yroots from this checkout instead of whatever is installed.
yroots_dir = Path("/data1/acme/Roots/hyun/Rootfinding-serial")
sys.path.insert(0, str(yroots_dir))

import yroots as yr  # noqa: E402
from yroots import MultiPower  # noqa: E402

# Test configuration
dim = 7
mindeg = 2
maxdeg = 7
nonzero = 3
num_tests = 100

coeff_dir = Path("../sparse/coeffs")
result_dir = Path("../sparse/results/yroots_results-optimize")
append = False  # True to add degrees to an existing run instead of overwriting it

SUMMARIES = ("avg_times", "max_resids", "avg_resids", "sum_resids")


def load_degree(coeff_dir, dim, deg, nonzero):
    """Load every test for one dim/degree/sparsity, shape (n_tests, dim, n_terms, dim+1).

    The last axis holds an exponent per dimension followed by the coefficient.
    """
    return np.load(coeff_dir / f"dim{dim}" / f"deg{deg}" / f"num{nonzero}.npy")


def coeff_matrix(test, dim, deg):
    """Expand one test's sparse terms into a dense coefficient tensor per function."""
    coeffs = np.zeros((dim,) + (deg + 1,) * dim)
    exponents = test[:, :, :dim].astype(int)
    values = test[:, :, dim]
    for func in range(dim):
        for idx, value in zip(exponents[func], values[func]):
            coeffs[(func, *idx)] = value
    return coeffs


def residuals(funcs, roots):
    """|f_i(root_j)| for every function i and root j, shape (len(funcs), len(roots))."""
    return np.array([np.abs(f(roots)) for f in funcs])


def solve_test(funcs, dim):
    """Solve one system on [-1, 1]^dim, returning the roots and the elapsed seconds."""
    start = time.perf_counter()
    roots = yr.solve(funcs, -np.ones(dim), np.ones(dim))
    return roots, time.perf_counter() - start


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


def main():
    base_dir = result_dir / f"dim{dim}" / f"nonzero{nonzero}"
    roots_dir = base_dir / "roots"
    roots_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = {name: base_dir / f"{name}.txt" for name in SUMMARIES}
    if not append:
        for path in summary_paths.values():
            path.write_text("")

    for deg in range(mindeg, maxdeg + 1):
        print(f"--- Dim {dim} Degree {deg}/{maxdeg} ---", flush=True)
        tests = load_degree(coeff_dir, dim, deg, nonzero)
        reps = min(num_tests, len(tests))

        if deg == mindeg:
            # Warm up numba's JIT so it isn't charged to the first timed test.
            solve_test([MultiPower(c) for c in coeff_matrix(tests[0], dim, deg)], dim)

        times, all_res, roots_by_test = [], [], {}
        for test in range(reps):
            if (test + 1) % 10 == 0:
                print(f"  test {test + 1}/{reps}", flush=True)

            funcs = [MultiPower(c) for c in coeff_matrix(tests[test], dim, deg)]
            roots, elapsed = solve_test(funcs, dim)

            times.append(elapsed)
            all_res.extend(residuals(funcs, roots).ravel())
            roots_by_test[test + 1] = roots

        write_roots_json(roots_dir / f"roots_deg{deg}.json", roots_by_test)

        res = np.array(all_res)
        append_value(summary_paths["avg_times"], np.mean(times) if times else np.nan)
        append_value(summary_paths["max_resids"], res.max() if res.size else np.nan)
        append_value(summary_paths["avg_resids"], res.mean() if res.size else np.nan)
        append_value(summary_paths["sum_resids"], res.sum() if res.size else np.nan)

    print("Program Finished")


if __name__ == "__main__":
    main()