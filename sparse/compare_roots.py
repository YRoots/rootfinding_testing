"""
Compare root-finding results across Bertini (v1.6), jroots, yroots, and yroots-optimize.

For every dim/nonzero3/degree combination present in ALL FOUR result sets, this loads
each method's roots for every test case and checks that jroots / yroots / yroots-optimize
agree with Bertini (treated as ground truth) -- same number of roots, and each root
matched within TOL under some pairing. Mismatches are reported to the console.

Coordinate order note: yroots and yroots-optimize store each root as (x1, ..., xd),
matching Bertini's native output order. jroots stores each root's coordinates in
*reverse* order (xd, ..., x1) -- confirmed by cross-checking numeric values in dim2
and dim3 test cases. This script reverses jroots's coordinates before comparing so
all four are compared in a consistent order.
"""
import json
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))  # sparse/
RESULTS = os.path.join(BASE, "results")

BERTINI_VERSION = "v1.6"
NONZERO = 3
TOL = 1e-8
MAX_DIM = 8
MAX_MISMATCHES_SHOWN = 10

# yroots and jroots only search the box [-1, 1]^d, so a bertini root outside it
# is not something they could ever have returned and counting it as a
# disagreement is misleading. Bertini roots outside the closed box (plus
# DOMAIN_TOL, 0 so every solver uses the same rule) are dropped before
# comparing. Set DOMAIN_RADIUS to None to compare
# bertini's raw output instead.
DOMAIN_RADIUS = 1.0
DOMAIN_TOL = 0.0

METHOD_DIRS = {
    "bertini": os.path.join(RESULTS, "bertini_results", BERTINI_VERSION),
    "jroots": os.path.join(RESULTS, "jroots_results"),
    "yroots": os.path.join(RESULTS, "yroots_results")
}

# Methods compared against bertini, and whether their JSON stores coordinates
# in reverse order (xd, ..., x1) rather than bertini's (x1, ..., xd).
COMPARED = (
    ("jroots", True),
    ("yroots", False)
)


def parse_bertini_txt(path, dim):
    """Parse a Bertini roots_deg*.txt file -> {test_idx: [tuple(dim floats), ...]}."""
    results = {}
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f]

    i, n = 0, len(lines)
    while i < n:
        line = lines[i].strip()
        if line.startswith("===== TEST"):
            test_idx = int(line.split()[2])
            i += 1
            while i < n and not lines[i].strip():
                i += 1
            count = int(lines[i].strip().split()[0])
            i += 1
            while i < n and not lines[i].strip():
                i += 1

            roots = []
            for _ in range(count):
                coords = []
                for _ in range(dim):
                    coords.append(float(lines[i].strip().split()[0]))
                    i += 1
                roots.append(tuple(coords))
                while i < n and not lines[i].strip():
                    i += 1
            results[test_idx] = roots
        else:
            i += 1
    return results


def parse_json_roots(path, reverse=False):
    """Parse a roots_deg*.json file -> {test_idx: [tuple(dim floats), ...]}."""
    with open(path) as f:
        data = json.load(f)
    results = {}
    for k, roots in data.items():
        pts = [tuple(reversed(r)) if reverse else tuple(r) for r in roots]
        results[int(k)] = pts
    return results


def in_domain(root):
    """True if a root lies inside [-1, 1]^d, within DOMAIN_TOL."""
    if DOMAIN_RADIUS is None:
        return True
    return all(abs(v) <= DOMAIN_RADIUS + DOMAIN_TOL for v in root)


def roots_match(ref, other, tol=TOL):
    """Greedy unordered match. Returns (ok, detail) where detail explains a failure."""
    if len(ref) != len(other):
        return False, f"count {len(ref)} (bertini) vs {len(other)}"
    used = [False] * len(other)
    for ra in ref:
        found = False
        for j, rb in enumerate(other):
            if used[j]:
                continue
            if all(abs(x - y) <= tol for x, y in zip(ra, rb)):
                used[j] = True
                found = True
                break
        if not found:
            return False, f"unmatched bertini root {ra}"
    return True, None


def degrees_in(base, dim, is_bertini):
    d = os.path.join(base, f"dim{dim}", f"nonzero{NONZERO}")
    if not is_bertini:
        d = os.path.join(d, "roots")
    pattern = re.compile(r"roots_deg(\d+)\.(txt|json)$")
    degs = set()
    if os.path.isdir(d):
        for fn in os.listdir(d):
            m = pattern.match(fn)
            if m:
                degs.add(int(m.group(1)))
    return degs


def method_names():
    return ["bertini"] + [name for name, _ in COMPARED]


def search_dir(name, dim):
    """The directory this method's roots_deg* files are expected in."""
    d = os.path.join(METHOD_DIRS[name], f"dim{dim}", f"nonzero{NONZERO}")
    return d if name == "bertini" else os.path.join(d, "roots")


def degrees_by_method(dim):
    """{method: set of degrees found} for one dimension."""
    return {name: degrees_in(METHOD_DIRS[name], dim, is_bertini=(name == "bertini"))
            for name in method_names()}


def common_degrees(dim, found=None):
    """Degrees present for every method. `found` receives the per-method sets."""
    per_method = degrees_by_method(dim)
    if found is not None:
        found.update(per_method)
    if not all(per_method.values()):
        return []
    return sorted(set.intersection(*per_method.values()))


def diagnose():
    """Explain why nothing was compared, listing every path that was searched."""
    print("Nothing was compared. Checking each configured result directory:\n")
    for name in method_names():
        root = METHOD_DIRS[name]
        mark = "ok     " if os.path.isdir(root) else "MISSING"
        print(f"  [{mark}] {name:11s} {root}")
    print()

    for dim in range(2, MAX_DIM + 1):
        per_method = degrees_by_method(dim)
        if not any(per_method.values()):
            continue
        have = [n for n in method_names() if per_method[n]]
        lack = [n for n in method_names() if not per_method[n]]
        print(f"  dim {dim}: found degrees for {', '.join(have)}; "
              f"nothing for {', '.join(lack)}")
        for name in lack:
            d = search_dir(name, dim)
            if not os.path.isdir(d):
                print(f"      {name}: no such directory {d}")
            else:
                names = sorted(os.listdir(d))[:5]
                print(f"      {name}: {d} exists but holds no roots_deg*.txt/.json"
                      f"{' (contains: ' + ', '.join(names) + ')' if names else ' (empty)'}")
    print("\nAdjust BASE/RESULTS, BERTINI_VERSION, NONZERO or METHOD_DIRS at the "
          "top of this script to match this checkout.")


def load_all(dim, deg):
    """Load bertini plus every compared method -> (bertini, {name: roots})."""
    bertini = parse_bertini_txt(
        os.path.join(METHOD_DIRS["bertini"], f"dim{dim}", f"nonzero{NONZERO}",
                     f"roots_deg{deg}.txt"),
        dim,
    )
    bertini = {t: [r for r in roots if in_domain(r)] for t, roots in bertini.items()}
    others = {
        name: parse_json_roots(
            os.path.join(METHOD_DIRS[name], f"dim{dim}", f"nonzero{NONZERO}",
                         "roots", f"roots_deg{deg}.json"),
            reverse=reverse,
        )
        for name, reverse in COMPARED
    }
    return bertini, others


def main():
    grand_tests = 0
    grand_comparisons = 0
    grand_mismatches = 0

    for dim in range(2, MAX_DIM + 1):
        found = {}
        degs = common_degrees(dim, found)
        if not degs:
            if any(found.values()):
                lack = [n for n in method_names() if not found[n]]
                print(f"=== dim {dim}: skipped -- no roots_deg* files for "
                      f"{', '.join(lack)} ===")
            continue

        dim_tests = 0
        dim_comparisons = 0
        dim_mismatches = []

        for deg in degs:
            bertini, others = load_all(dim, deg)
            all_tests = sorted(set(bertini).union(*(set(o) for o in others.values())))

            for t in all_tests:
                dim_tests += 1
                if t not in bertini:
                    dim_comparisons += len(others)
                    dim_mismatches.append(
                        (deg, t, "-", "test absent from bertini file"))
                    continue
                for name in others:
                    dim_comparisons += 1
                    if t not in others[name]:
                        dim_mismatches.append(
                            (deg, t, name, f"test absent from {name} file"))
                        continue
                    ok, detail = roots_match(bertini[t], others[name][t])
                    if not ok:
                        dim_mismatches.append((deg, t, name, detail))

        grand_tests += dim_tests
        grand_comparisons += dim_comparisons
        grand_mismatches += len(dim_mismatches)

        print(f"=== dim {dim}: degrees {degs[0]}-{degs[-1]} ({len(degs)} degrees), "
              f"{dim_tests} test cases compared, {len(dim_mismatches)} mismatches "
              f"out of {dim_comparisons} comparisons ===")
        for (deg, t, name, detail) in dim_mismatches[:MAX_MISMATCHES_SHOWN]:
            print(f"    [deg {deg}, test {t}] {name} vs bertini: {detail}")
        if len(dim_mismatches) > MAX_MISMATCHES_SHOWN:
            print(f"    ... and {len(dim_mismatches) - MAX_MISMATCHES_SHOWN} more")

    if grand_tests == 0:
        print()
        diagnose()

    print("\n===== GRAND SUMMARY =====")
    print(f"Total test cases compared: {grand_tests}")
    print(f"Total comparisons ({len(COMPARED)} methods x test cases): {grand_comparisons}")
    print(f"Total mismatches vs bertini: {grand_mismatches}")


if __name__ == "__main__":
    main()
