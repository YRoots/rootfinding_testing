from yroots import MultiPower
import yroots as yr
import numpy as np
from time import time
import json
import os


def residuals(dim, funcs, roots):
    """Compute |f_i(roots[j])| for each i,j."""
    res = []
    for i in range(dim):
        row = []
        for root in roots:
            row.append(abs(funcs[i](root))[0])
        res.append(row)
    return res


def run_test(dim, deg, nonzero, num, roots_only=True):
    """
    Load the num-th test’s coefficient block from
      coeffs/dim{dim}/deg{deg}/num{nonzero}.npy
    and solve.
    """
    data_path = f"../dense/coeffs_mathematica_json/dim{dim}_deg{deg}_randn.txt"
    with open(data_path, 'r') as f:
        data = json.load(f)
    # shape = (300, dim, (deg+1), (deg+1))
    data = np.array(data)

    # data = np.load(f"../dense/coeffs/dim{dim}_deg{deg}_randn.npy")

    coeffs = data[num - 1]
    a = -np.ones(dim)
    b = np.ones(dim)

    funcs = [MultiPower(c) for c in coeffs]
    if roots_only:
        return yr.solve(funcs, a, b)
    else:
        start = time()
        roots = yr.solve(funcs, a, b)
        t = time() - start
        return residuals(dim, funcs, roots), roots, t


if __name__ == "__main__":
    dim = 4
    mindeg = 2
    maxdeg = 2
    nonzero = 3
    num_tests = 1

    # Warm-up (JIT, caches, etc.)
    run_test(dim, 2, nonzero, 1)

    # Base output directories
    base_dir = f"../dense/results/yroots_results/roots/dim{dim}"
    os.makedirs(base_dir, exist_ok=True)

    # Summary file paths (parent folder)
    avg_time_path  = os.path.join(base_dir, "avg_times.txt")
    max_resid_path = os.path.join(base_dir, "max_resids.txt")
    avg_resid_path = os.path.join(base_dir, "avg_resids.txt")
    sum_resid_path = os.path.join(base_dir, "sum_resids.txt")

    # Helper: append one plain value per degree (no headers, no degree number)
    def append_value(path, value):
        with open(path, "a") as f:
            f.write(f"{value:.17g}\n")

    for deg in range(mindeg, maxdeg + 1):
        print(f"--- Degree {deg}/{maxdeg} ---")

        # Per-degree folder for roots
        out_dir = os.path.join(base_dir, f"deg{deg}")
        os.makedirs(out_dir, exist_ok=True)

        all_times = []
        all_res   = []

        for test in range(num_tests):
            if (test + 1) % 10 == 0:
                print(f"  test {test + 1}/{num_tests}")

            res, roots, t = run_test(dim, deg, nonzero, test + 1, roots_only=False)
            all_times.append(t)
            all_res.extend(np.array(res).flatten())

            # Save roots per test into the per-degree folder
            np.savetxt(os.path.join(out_dir, f"roots_test{test + 1}.txt"), roots)

        # Compute summaries for this degree
        arr = np.array(all_res)
        avg_time  = float(np.mean(all_times)) if all_times else float("nan")
        max_resid = float(arr.max())          if arr.size   else float("nan")
        avg_resid = float(arr.mean())         if arr.size   else float("nan")
        sum_resid = float(arr.sum())          if arr.size   else float("nan")

        # Append values (one per line, in degree order)
        append_value(avg_time_path,  avg_time)
        append_value(max_resid_path, max_resid)
        append_value(avg_resid_path, avg_resid)
        append_value(sum_resid_path, sum_resid)