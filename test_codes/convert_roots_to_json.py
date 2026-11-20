"""One-off: convert old NSolveValues / Chebfun root files to roots/roots_deg{deg}.json.

Run from test_codes/. Produces the layout run_yroots_combined.py writes:

    <method dir>/roots/roots_deg{deg}.json   {test number: [roots]}

Sources converted (coordinates are kept exactly as stored, so the
"roots_reversed" flags in graph_all.py still apply):

  dense/results/NSolveValues_results/dim{2,4,5}/deg{n}/roots_test{t}.txt
  dense/results/NSolveValues_results/dim{2,3}_deg{n}_output2.txt
        roots of the "2_cont" run that graph_all.py plots for dims 2-3
  dense/results/chebfun_results/2d_all/deg{n}/test{t}_roots.txt
        -> dense/results/chebfun_results/dim2/roots/
  sparse/results/chebfun_results/dim2/nonzero3/deg{n}/roots_test{t}.txt

The per-test deg{n}/ folders are deleted once their JSON is written and
re-read identically. The Mathematica *_output*.txt files are kept: they also
hold the polynomials.
"""

import json
import math
import os
import re
import shutil

import root_accuracy as ra

PER_TEST = re.compile(r"^(?:roots_test(\d+)|test(\d+)_roots)\.txt$")
FLOAT = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def write_roots_json(path, roots_by_test):
    """Same layout as run_yroots_combined.write_roots_json (copied so this
    script doesn't need yroots installed)."""
    with open(path, "w") as f:
        f.write("{\n")
        items = sorted(roots_by_test.items())
        for i, (test, roots) in enumerate(items):
            rows = ",\n".join("    " + json.dumps(list(root), separators=(",", ":"))
                              for root in roots)
            body = f"[\n{rows}\n  ]" if len(roots) else "[]"
            f.write(f'  "{test}": {body}')
            f.write(",\n" if i < len(items) - 1 else "\n")
        f.write("}\n")


def same_roots(a, b, tol=1e-10):
    """Same tests, and the same roots in each up to tol (any order)."""
    if a.keys() != b.keys():
        return False
    for t in a:
        if len(a[t]) != len(b[t]):
            return False
        left = [list(r) for r in b[t]]
        for r in a[t]:
            hit = next((j for j, q in enumerate(left)
                        if all(abs(x - y) <= tol for x, y in zip(r, q))), None)
            if hit is None:
                return False
            left.pop(hit)
    return True


def parse_root_lines(lines, path):
    """Roots from a per-test file. Repairs Mathematica's two-line display of
    small numbers, e.g.
        -0.0555 0.1336            -6
        9.0083 x 10 0.1253 0.1045
    which means [-0.0555, 0.1336, 9.0083e-6, 0.1253, 0.1045]."""
    lines = [ln.rstrip("\n") for ln in lines
             if ln.strip() and not ln.lstrip().startswith("#")]
    roots, i = [], 0
    while i < len(lines):
        toks = lines[i].split()
        if i + 1 < len(lines) and " x 10" in lines[i + 1]:
            *head, exp = toks
            tail = lines[i + 1].split()
            vals = [float(v) for v in head]
            k = 0
            while k < len(tail):
                if k + 2 < len(tail) and tail[k + 1] == "x" and tail[k + 2] == "10":
                    vals.append(float(tail[k]) * 10.0 ** int(exp))
                    k += 3
                else:
                    vals.append(float(tail[k]))
                    k += 1
            print(f"  repaired two-line root in {path}: {vals}")
            roots.append(vals)
            i += 2
            continue
        if not all(FLOAT.match(t) for t in toks):
            raise ValueError(f"can't parse line {lines[i]!r} in {path}")
        roots.append([float(t) for t in toks])
        i += 1
    return roots


def load_per_test_dir(deg_dir):
    data = {}
    for name in os.listdir(deg_dir):
        m = PER_TEST.match(name)
        if not m:
            raise ValueError(f"unexpected file {name} in {deg_dir}")
        path = os.path.join(deg_dir, name)
        with open(path) as f:
            data[int(m.group(1) or m.group(2))] = parse_root_lines(f, path)
    return data


def check_dims(data, dim, where):
    for t, roots in data.items():
        for r in roots:
            if len(r) != dim or not all(math.isfinite(v) for v in r):
                raise ValueError(f"bad root {r} for test {t} in {where}")


def write_and_verify(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_roots_json(path, data)
    with open(path) as f:
        back = {int(k): v for k, v in json.load(f).items()}
    assert back == {t: [list(r) for r in rs] for t, rs in data.items()}, path


def convert_per_test(deg_parent, out_dir, dim):
    """deg_parent/deg{n}/ per-test files -> out_dir/roots/roots_deg{n}.json."""
    for name in sorted(os.listdir(deg_parent)):
        m = re.fullmatch(r"deg(\d+)", name)
        if not m:
            continue
        deg_dir = os.path.join(deg_parent, name)
        data = load_per_test_dir(deg_dir)
        check_dims(data, dim, deg_dir)
        out = os.path.join(out_dir, "roots", f"roots_deg{m.group(1)}.json")
        if os.path.exists(out):
            raise FileExistsError(out)
        write_and_verify(out, data)
        shutil.rmtree(deg_dir)
        print(f"{deg_dir} ({len(data)} tests) -> {out}")


def convert_mathematica_outputs(results_dir, dim):
    """dim{d}_deg{n}_output2.txt -> dim{d}/roots/roots_deg{n}.json."""
    pat = re.compile(rf"dim{dim}_deg(\d+)_output2\.txt")
    for name in sorted(os.listdir(results_dir)):
        m = pat.fullmatch(name)
        if not m:
            continue
        src = os.path.join(results_dir, name)
        data = {t: [list(r) for r in rs]
                for t, rs in ra._load_mathematica_output(src).items()}
        check_dims(data, dim, src)
        out = os.path.join(results_dir, f"dim{dim}", "roots", f"roots_deg{m.group(1)}.json")
        if os.path.exists(out):
            with open(out) as f:
                existing = {int(k): v for k, v in json.load(f).items()}
            if same_roots(existing, data):
                print(f"{src}: same roots as existing {out} (kept)")
                continue
            # dim2/deg2 also came from a later, aborted NSolve run (its summary
            # files are empty) that found the same roots to ~1e-8. Prefer the
            # output2 ("2_cont") run, whose summaries graph_all.py plots.
            if not same_roots(existing, data, tol=1e-6):
                raise ValueError(f"{out} already exists with different roots than {src}")
            print(f"{src}: replaces {out} (same roots to 1e-6; this is the plotted run)")
        write_and_verify(out, data)
        print(f"{src} ({len(data)} tests) -> {out}")


if __name__ == "__main__":
    nsolve = "../dense/results/NSolveValues_results"
    for dim in (2, 4, 5):
        convert_per_test(f"{nsolve}/dim{dim}", f"{nsolve}/dim{dim}", dim)
    for dim in (2, 3):
        convert_mathematica_outputs(nsolve, dim)

    cheb = "../dense/results/chebfun_results"
    convert_per_test(f"{cheb}/2d_all", f"{cheb}/dim2", 2)
    if not os.listdir(f"{cheb}/2d_all"):
        os.rmdir(f"{cheb}/2d_all")

    sp = "../sparse/results/chebfun_results/dim2/nonzero3"
    convert_per_test(sp, sp, 2)
