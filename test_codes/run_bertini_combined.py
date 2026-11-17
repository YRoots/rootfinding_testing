"""Benchmark Bertini on the sparse and dense polynomial test suites.

Set SUITE below to "sparse" or "dense" (plus the other parameters), then run
``python run_bertini_combined.py`` from anywhere.

    SUITE = "sparse"  reads sparse/coeffs/dim{dim}/deg{deg}/num{nonzero}.npy,
                      shape (n_tests, dim, nonzero, dim + 1); each term is
                      [e_0, ..., e_{dim-1}, coeff]
    SUITE = "dense"   reads dense/coeffs/dim{dim}_deg{deg}_randn.npy,
                      shape (n_tests, dim, deg+1, ..., deg+1); entry
                      [f, e_0, ..., e_{dim-1}] is the coefficient of
                      x0^e_0 * ... * x{dim-1}^e_{dim-1} in f. With
                      zero_constant_term = True the constant term
                      [f, 0, ..., 0] is forced to zero, so every dense
                      system has the root x = 0

Both are turned into the same list of (exponents, coefficients) per equation,
so one function writes the Bertini input and one computes residuals.

For every degree in [MINDEG, MAXDEG] it writes, under
    sparse/results/bertini_results/<version>/dim{dim}/nonzero{nonzero}   or
    dense/results/bertini_results/<version>/dim{dim}:

    avg_times.txt        Bertini's own CPU time per system, averaged (see run_bertini)
    avg_wall_times.txt   wall-clock time per system, averaged, for reference
    max_resids.txt       max |f_i(root)| over the real roots in the box
    avg_resids.txt       mean |f_i(root)|
    sum_resids.txt       sum of |f_i(root)|
    roots/roots_deg{deg}.json
                         {test number: real roots in the box}, the same layout
                         yroots and jroots write; coordinates in order
                         (x0, ..., x{dim-1})
    skipped_tests.txt    tests where Bertini produced no solution file

Each summary file holds one value per line, in degree order.

Bertini finds every real root, wherever it lies, but yroots, jroots and MTV only
search the box [-1, 1]^dim. Roots outside the box (see DOMAIN_RADIUS) are left
out of both the roots JSON and the residual summaries, so every method reports
the same set of roots. Roots far outside the box are often very inaccurate and
would otherwise dominate Bertini's residuals.

Several copies can run at once (e.g. under nohup with different SUITE / dim
settings): each run works in its own temporary directory, so they never share
Bertini's input or output files. Two runs with the SAME settings still write
the same results directory, so don't do that.
"""

import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np

# -----------------------
# Configuration
# -----------------------
SUITE = "dense"  # "sparse" | "dense"
dim = 5           # number of variables (square system: n_eq = dim)
mindeg = 2
maxdeg = 15
nonzero = 3       # sparse only: terms per polynomial (used in paths)
num_tests = 100   # max tests per degree
zero_constant_term = True  # dense only: force the constant coefficient to 0

# Stop early once a degree's average Bertini CPU time per system (the value
# written to avg_times.txt) exceeds this many seconds. That degree is still
# finished and saved; higher degrees are skipped. None = run up to maxdeg.
max_avg_time = 2000

# Keep only real roots with max |x_i| <= DOMAIN_RADIUS + DOMAIN_TOL, the box
# [-1, 1]^dim the other solvers search. DOMAIN_TOL = 0 matches compare_roots.py.
# Set DOMAIN_RADIUS = None to keep every real root Bertini returns.
DOMAIN_RADIUS = 1.0
DOMAIN_TOL = 0.0

# Bertini version tag: change this to "v1.6", "v1.7", etc.
BERT_VERSION = "v1.6"

if SUITE not in ("sparse", "dense"):
    raise ValueError(f"unknown SUITE: {SUITE!r}")

# -----------------------
# Paths based on this file's location
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

# Bertini executable: assumes BertiniLinux64_vX.Y is one level up from test_codes
BERT_EXEC = os.path.abspath(
    os.path.join(ROOT_DIR, f"BertiniLinux64_{BERT_VERSION}", "bertini")
)

# Bertini reads "input" from, and writes all its output files to, the directory
# it runs in. Each run of this script gets its own private directory so that
# several runs can go at once without overwriting each other's files. It is
# deleted when the script exits.
WORK_DIR = tempfile.mkdtemp(prefix=f"bertini_{SUITE}_dim{dim}_")
atexit.register(shutil.rmtree, WORK_DIR, ignore_errors=True)

# `kill <pid>` sends SIGTERM, which would otherwise skip the cleanup above.
signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))

# Coefficients: <suite>/coeffs, one level up from test_codes
COEFF_DIR = os.path.join(ROOT_DIR, SUITE, "coeffs")

# Results: <suite>/results/bertini_results/<version>/dimX[/nonzeroY]
RESULTS_BASE = os.path.join(
    ROOT_DIR, SUITE, "results", "bertini_results_new", BERT_VERSION, f"dim{dim}"
)
if SUITE == "sparse":
    RESULTS_BASE = os.path.join(RESULTS_BASE, f"nonzero{nonzero}")
os.makedirs(RESULTS_BASE, exist_ok=True)
ROOTS_DIR = os.path.join(RESULTS_BASE, "roots")
os.makedirs(ROOTS_DIR, exist_ok=True)

SUMMARIES = ("avg_times", "avg_wall_times", "max_resids", "avg_resids", "sum_resids")

# File to log skipped tests
skipped_log_path = os.path.join(RESULTS_BASE, "skipped_tests.txt")


# -----------------------
# Loading coefficients
# -----------------------
def coeff_path(deg):
    if SUITE == "sparse":
        return os.path.join(COEFF_DIR, f"dim{dim}", f"deg{deg}", f"num{nonzero}.npy")
    return os.path.join(COEFF_DIR, f"dim{dim}_deg{deg}_randn.npy")


def to_terms(test_coeffs):
    """
    Turn ONE test's coefficients (either suite) into a list with one entry per
    equation: (exps, vals), where exps is an int array of shape (n_terms, dim)
    and vals the matching coefficients. Zero coefficients are dropped.
    """
    system = []
    for poly in test_coeffs:
        poly = np.asarray(poly, dtype=float)
        if SUITE == "sparse":
            # poly[term] = [e_0, ..., e_{dim-1}, coeff]
            exps = poly[:, :dim].astype(int)
            vals = poly[:, dim]
        else:
            # poly[e_0, ..., e_{dim-1}] = coeff
            if zero_constant_term:
                poly = poly.copy()
                poly[(0,) * dim] = 0.0
            exps = np.argwhere(poly != 0)
            vals = poly[tuple(exps.T)]
        keep = vals != 0
        system.append((exps[keep], vals[keep]))
    return system


# -----------------------
# Helper: write Bertini input file
# -----------------------
def set_up_input(system):
    """Write the Bertini input file for one system into WORK_DIR/input."""
    n_eq = len(system)

    lines = [
        "CONFIG",
        "SECURITYMAXNORM: 1;",
        "END;",
        "INPUT",
        "function " + ", ".join(f"f{i}" for i in range(n_eq)) + ";",
        "variable_group " + ", ".join(f"x{i}" for i in range(dim)) + ";",
    ]

    for eq_idx, (exps, vals) in enumerate(system):
        terms = [
            repr(float(c)) + "".join(f"*x{i}^{e}" for i, e in enumerate(exp) if e != 0)
            for exp, c in zip(exps, vals)
        ]
        # In case all coefficients were zero, we still want something valid
        lines.append(f"f{eq_idx} = " + (" + ".join(terms) if terms else "0") + ";")

    lines.append("END;")

    with open(os.path.join(WORK_DIR, "input"), "w") as f:
        f.write("\n".join(lines))


# -----------------------
# Helper: run Bertini and measure the time it actually spent
# -----------------------
def run_bertini():
    """
    Run Bertini on WORK_DIR/input and return (cpu_time, wall_time) in seconds.

    Bertini v1.6/v1.7 does not report its own run time, so cpu_time is the
    user + system CPU time the kernel charged to the Bertini process, read
    with os.wait4. This leaves out Python/subprocess start-up overhead and
    time the process spent waiting to be scheduled, which wall_time includes.
    Bertini's serial build is single-threaded, so cpu_time is its compute time.
    """
    start = perf_counter()
    proc = subprocess.Popen(
        [BERT_EXEC, "input"],
        cwd=WORK_DIR,
        stdout=subprocess.DEVNULL,
    )
    try:
        _, status, usage = os.wait4(proc.pid, 0)
    except BaseException:
        # Interrupted (Ctrl-C / kill): don't leave Bertini running on its own
        proc.kill()
        proc.wait()
        raise
    wall_time = perf_counter() - start
    proc.returncode = os.waitstatus_to_exitcode(status)

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, proc.args)

    return usage.ru_utime + usage.ru_stime, wall_time


# -----------------------
# Reading roots and residuals
# -----------------------
def read_bertini_solutions(path, nvars, imag_tol=None):
    """
    Real parts of the roots in a real_finite_solutions file. With imag_tol set,
    roots with any imaginary part above it are dropped; with None, every root
    Bertini itself classed as real is kept.
    """
    with open(path, "r") as f:
        # strip blank lines
        lines = [ln.strip() for ln in f if ln.strip()]

    nsol = int(lines[0])
    sols = []
    idx = 1
    for _ in range(nsol):
        sol = []
        is_real = True
        for _ in range(nvars):
            re_str, im_str = lines[idx].split()
            re, im = float(re_str), float(im_str)
            if imag_tol is not None and abs(im) > imag_tol:
                is_real = False
            sol.append(re)  # keep just real part
            idx += 1
        if is_real:
            sols.append(sol)
    return sols


def in_box(roots):
    """The roots with every coordinate inside [-DOMAIN_RADIUS, DOMAIN_RADIUS],
    within DOMAIN_TOL. Keeps every root when DOMAIN_RADIUS is None."""
    if DOMAIN_RADIUS is None:
        return roots
    return [r for r in roots
            if all(abs(v) <= DOMAIN_RADIUS + DOMAIN_TOL for v in r)]


def write_roots_json(path, roots_by_test):
    """Write {test number: list of roots}, one root per line, sorted by test number.
    Same layout as write_roots_json in run_yroots_combined.py."""
    with open(path, "w") as f:
        f.write("{\n")
        items = sorted(roots_by_test.items())
        for i, (test, roots) in enumerate(items):
            rows = ",\n".join("    " + json.dumps(root, separators=(",", ":"))
                              for root in roots)
            body = f"[\n{rows}\n  ]" if len(roots) else "[]"
            f.write(f'  "{test}": {body}')
            f.write(",\n" if i < len(items) - 1 else "\n")
        f.write("}\n")


def get_resid(system, roots):
    """|f_i(root)| for every equation and root, from the same terms Bertini saw."""
    res = []
    for exps, vals in system:
        for root in roots:
            monomials = np.prod(np.asarray(root) ** exps, axis=1)
            res.append(abs(vals @ monomials))
    return res


# -----------------------
# Main experiment
# -----------------------
summaries = {name: [] for name in SUMMARIES}

# open the skipped-log file once for this run (overwrite each time)
with open(skipped_log_path, "w") as skipped_log:

    for deg in range(mindeg, maxdeg + 1):
        times = []
        wall_times = []
        resids = []
        roots_by_test = {}  # tests with no solution file are left out

        path = coeff_path(deg)
        # Allow pickle because sparse structures are often saved as object arrays
        coeffs = np.load(path, allow_pickle=True)
        n_tests = min(num_tests, len(coeffs))

        for test in range(n_tests):
            system = to_terms(coeffs[test])
            set_up_input(system)

            # Run Bertini in WORK_DIR so all its outputs land there
            t, wall_t = run_bertini()
            times.append(t)
            wall_times.append(wall_t)

            sol_file = os.path.join(WORK_DIR, "real_finite_solutions")

            if os.path.exists(sol_file):
                real_solutions = in_box(
                    read_bertini_solutions(sol_file, dim, imag_tol=1e-12))
                resids += get_resid(system, real_solutions)

                # Every root in the box Bertini classed as real (its own
                # real/non-real test)
                roots_by_test[test + 1] = in_box(read_bertini_solutions(sol_file, dim))

                # Remove the individual solution file after consuming it
                os.remove(sol_file)

            else:
                skipped_log.write(
                    f"SKIPPED: suite={SUITE}, dim={dim}, deg={deg}, "
                    + (f"nonzero={nonzero}, " if SUITE == "sparse" else "")
                    + f"test={test + 1}, coeff_file={path}, "
                    f"bertini_version={BERT_VERSION}\n"
                )
                print(
                    f"[warn] test {test + 1} (deg={deg}): "
                    f"no '{sol_file}' produced — skipping"
                )

        write_roots_json(os.path.join(ROOTS_DIR, f"roots_deg{deg}.json"), roots_by_test)

        # Averages for this degree (NaN if no test found a real root)
        summaries["avg_times"].append(np.mean(times))
        summaries["avg_wall_times"].append(np.mean(wall_times))
        summaries["max_resids"].append(np.max(resids) if resids else np.nan)
        summaries["avg_resids"].append(np.mean(resids) if resids else np.nan)
        summaries["sum_resids"].append(np.sum(resids))

        # Save incrementally so a long run keeps finished degrees
        for name, values in summaries.items():
            np.savetxt(os.path.join(RESULTS_BASE, f"{name}.txt"), values)

        print(f"deg {deg}: avg Bertini CPU time {summaries['avg_times'][-1]:.4g} s")

        if (max_avg_time is not None and deg < maxdeg
                and summaries["avg_times"][-1] > max_avg_time):
            print(f"Stopping: deg {deg} averaged {summaries['avg_times'][-1]:.4g} s "
                  f"per system, over max_avg_time = {max_avg_time} s. "
                  f"Degrees {deg + 1}-{maxdeg} skipped.")
            break

