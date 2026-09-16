"""
Compute residuals for every root that the solvers disagree about.

compare_roots.py reports *that* jroots / yroots / yroots-optimize disagree with
bertini; it cannot say who is right. This script answers that, by evaluating the
actual polynomial system at each disputed root:

    residual(r) = max_i |f_i(r)|

A disputed root with a residual at the noise floor (~1e-15) is a genuine root
that whoever omitted it missed. A disputed root with an O(1) residual is
spurious -- whoever reported it is wrong.

Coefficients come from the sparse suite's coeff files, which are NOT in git
(.gitignore excludes *.npy), so this must be run where they exist:

    coeffs/dim{dim}/deg{deg}/num{nonzero}.npy
        shape (num_tests, dim, n_terms, dim+1); the last axis holds one
        exponent per dimension followed by the coefficient. Tests are indexed
        from 0 here but from 1 in the roots files, hence the t-1 below.

Run it from sparse/ the same way as compare_roots.py.
"""
import os

import numpy as np

import compare_roots as cr

COEFF_DIR = os.path.join(cr.BASE, "coeffs")

# A disputed root at or below this residual is treated as genuine.
REAL_ROOT_TOL = 1e-8

# Show at most this many agreed roots per dimension when establishing the
# baseline residual that the disputed numbers are read against.
BASELINE_SAMPLE = 200


def coeff_path(dim, deg, nonzero=cr.NONZERO):
    return os.path.join(COEFF_DIR, f"dim{dim}", f"deg{deg}", f"num{nonzero}.npy")


def load_coeffs(dim, deg, nonzero=cr.NONZERO):
    """(num_tests, dim, n_terms, dim+1) array, or None if not on disk."""
    path = coeff_path(dim, deg, nonzero)
    if not os.path.isfile(path):
        return None
    return np.load(path)


def evaluate(test_coeffs, root):
    """Evaluate every f_i at root -> array of dim values."""
    root = np.asarray(root, dtype=float)
    # Exponents are stored as floats; np.power(negative, float) is nan, so the
    # cast to int is required, not cosmetic.
    exps = test_coeffs[:, :, :root.size].astype(int)
    vals = test_coeffs[:, :, root.size]
    powers = np.prod(np.power(root[None, None, :], exps), axis=2)
    return np.sum(vals * powers, axis=1)


def residual(test_coeffs, root):
    """max_i |f_i(root)|."""
    return float(np.max(np.abs(evaluate(test_coeffs, root))))


def disputed_roots(ref, others):
    """Roots claimed by some methods but not all.

    Returns [(root, set_of_methods_reporting_it)], where the method set is
    drawn from {"bertini"} | others.keys().
    """
    claims = [("bertini", ref)] + [(name, roots) for name, roots in others.items()]
    pooled = []
    for name, roots in claims:
        for r in roots:
            for entry in pooled:
                if all(abs(x - y) <= cr.TOL for x, y in zip(r, entry[0])):
                    entry[1].add(name)
                    break
            else:
                pooled.append((r, {name}))

    everyone = {name for name, _ in claims}
    return [(r, names) for r, names in pooled if names != everyone]


def main():
    # Residuals are about who is right, not about the search box, so look at
    # bertini's raw output including roots outside [-1, 1]^d.
    cr.DOMAIN_RADIUS = None

    missing_coeffs = []
    baseline = []
    findings = []

    for dim in range(2, cr.MAX_DIM + 1):
        degs = cr.common_degrees(dim)
        if not degs:
            continue

        for deg in degs:
            coeffs = load_coeffs(dim, deg)
            if coeffs is None:
                missing_coeffs.append(coeff_path(dim, deg))
                continue

            bertini, others = cr.load_all(dim, deg)
            all_tests = sorted(set(bertini).union(*(set(o) for o in others.values())))

            for t in all_tests:
                if t - 1 >= len(coeffs):
                    continue
                test_coeffs = coeffs[t - 1]
                ref = bertini.get(t, [])

                agreed = all(
                    cr.roots_match(ref, others[name].get(t, []))[0] for name in others
                )
                if agreed:
                    if len(baseline) < BASELINE_SAMPLE * (dim - 1):
                        for r in ref:
                            baseline.append(residual(test_coeffs, r))
                    continue

                for root, names in disputed_roots(
                    ref, {n: others[n].get(t, []) for n in others}
                ):
                    findings.append({
                        "dim": dim, "deg": deg, "test": t, "root": root,
                        "found_by": sorted(names),
                        "resid": residual(test_coeffs, root),
                        "in_box": all(abs(v) <= 1.0 + 1e-8 for v in root),
                    })

    if missing_coeffs:
        print(f"WARNING: {len(missing_coeffs)} coefficient file(s) not found, "
              f"those degrees were skipped. First few:")
        for p in missing_coeffs[:5]:
            print(f"    {p}")
        print()

    if baseline:
        b = np.array(baseline)
        print(f"Baseline: residuals of {len(b)} roots all methods agree on")
        print(f"    median {np.median(b):.3e}   max {b.max():.3e}\n")

    if not findings:
        print("No disputed roots found.")
        if missing_coeffs and not baseline:
            print("(No coefficients were loaded at all -- check COEFF_DIR.)")
        return

    print(f"{len(findings)} disputed root(s):\n")
    real = spurious = 0
    for f in sorted(findings, key=lambda f: (f["dim"], f["deg"], f["test"])):
        verdict = "REAL" if f["resid"] <= REAL_ROOT_TOL else "SPURIOUS"
        real += verdict == "REAL"
        spurious += verdict == "SPURIOUS"
        missed = [m for m in cr.method_names() if m not in f["found_by"]]
        coords = ", ".join(f"{v: .6f}" for v in f["root"])
        print(f"  [dim{f['dim']} deg{f['deg']} test{f['test']}] ({coords})")
        print(f"      residual {f['resid']:.3e}  -> {verdict}"
              f"{'' if f['in_box'] else '  (outside [-1,1]^d)'}")
        print(f"      found by {', '.join(f['found_by'])}"
              f"{'  |  missed by ' + ', '.join(missed) if missed else ''}")

    print(f"\n===== SUMMARY =====")
    print(f"Disputed roots that are genuine (residual <= {REAL_ROOT_TOL:g}): {real}")
    print(f"Disputed roots that are spurious: {spurious}")


if __name__ == "__main__":
    main()
