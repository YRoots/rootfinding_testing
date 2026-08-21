from yroots import MultiPower
import yroots as yr
import numpy as np
from time import time
import os
import json


def residuals(dim, funcs, roots):
    """Compute |f_i(roots[j])| for each i,j."""
    res = []
    for i in range(dim):
        row = []
        for root in roots:
            row.append(abs(funcs[i](*root)))
        res.append(row)
    return res

def poly_from_terms(terms, dim):
    exps = np.array([list(map(int, t[:dim])) for t in terms], dtype=np.int64)
    coeffs = np.array([float(t[dim]) for t in terms], dtype=np.float64)

    def poly(*x):
        s = 0.0
        n_terms = exps.shape[0]
        for k in range(n_terms):
            m = 1.0
            for j in range(dim):
                ej = exps[k, j]
                if ej != 0:
                    m *= x[j] ** ej
            s += coeffs[k] * m
        return s

    return poly

def run_test(dim, deg, nonzero, num, roots_only=True):
    """
    Load the num-th test’s coefficient block from
      coeffs/dim{dim}/deg{deg}/num{nonzero}.npy
    and solve.
    """
    data = np.load(f"../sparse/coeffs/dim{dim}/deg{deg}/num{nonzero}.npy")
    coeffs = data[num - 1]
    a = -np.ones(dim)
    b = np.ones(dim)

    funcs = [poly_from_terms(c, dim) for c in coeffs]

    if roots_only:
        return yr.solve(funcs, a, b)
    else:
        start = time()
        roots = yr.solve(funcs, a, b)
        t = time() - start
        return residuals(dim, funcs, roots), roots, t
    
# Writes roots to a JSON file
def write_to_json(output_file, data):
    with open(output_file, "w") as f:
        f.write("{\n")
        
        items = list(data.items())
        for i, (test_num, roots) in enumerate(items):
            f.write(f'  "{test_num}": ')
            
            if len(roots) == 0:
                f.write("[]")
            else:
                f.write("[\n")
                for j, root in enumerate(roots):
                    f.write("    " + json.dumps(root.tolist()))
                    if j < len(roots) - 1:
                        f.write(",")
                    f.write("\n")
                f.write("  ]")
            
            if i < len(items) - 1:
                f.write(",")
            f.write("\n")
        
        f.write("}\n")


if __name__ == "__main__":
    dim = 6
    mindeg = 2
    maxdeg = 10
    nonzero = 3
    num_tests = 100

    # Warm-up (JIT, caches, etc.)
    run_test(dim, 2, nonzero, 1)
    
    # Warm-up all of degree 2 (these results in lower averge time for deg 2)
    # for i in range(num_tests):
    #     run_test(dim, 2, nonzero, i)

    # Base output directories
    base_dir = f"../sparse/results/yroots_lambda_results/dim{dim}/nonzero{nonzero}"
    os.makedirs(base_dir, exist_ok=True)

    # Summary file paths (parent folder)
    avg_time_path  = os.path.join(base_dir, "avg_times.txt")
    max_resid_path = os.path.join(base_dir, "max_resids.txt")
    avg_resid_path = os.path.join(base_dir, "avg_resids.txt")
    sum_resid_path = os.path.join(base_dir, "sum_resids.txt")

    # Wipe file data
    for file in  [avg_time_path, max_resid_path, avg_resid_path, sum_resid_path]:
        with open(file, "w") as f: # Change this to 'a' when adding number of degrees
            pass

    # Helper: append one plain value per degree (no headers, no degree number)
    def append_value(path, value):
        with open(path, "a") as f:
            f.write(f"{value:.17g}\n")

    for deg in range(mindeg, maxdeg + 1):
        print(f"--- Dim {dim} Degree {deg}/{maxdeg} ---")

        # Put all root files in one folder
        roots_dir = os.path.join(base_dir, "roots")
        os.makedirs(roots_dir, exist_ok=True)
        roots_file = os.path.join(roots_dir, f"roots_deg{deg}.json")

        all_times = []
        all_res   = []
        root_dic = {}

        for test in range(num_tests):
            if (test + 1) % 10 == 0:
                print(f"  test {test + 1}/{num_tests}")

            res, roots, t = run_test(dim, deg, nonzero, test + 1, roots_only=False)
            all_times.append(t)
            all_res.extend(np.array(res).flatten())

            # Save roots per test into the per-degree folder
            root_dic[str(test + 1)] = roots

        write_to_json(roots_file, root_dic)

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

    print("Program Finished")