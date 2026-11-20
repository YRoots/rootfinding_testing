"""One-off: give the two remaining dense NSolveValues runs their own JSON folders.

Run from test_codes/ (after convert_roots_to_json.py). Same layout as yroots:

  dense/results/NSolveValues_results_run1/dim{d}/roots/roots_deg{n}.json
      the first full run, from dim{d}_deg{n}_output.txt (300 tests per degree,
      dims 2-5); before this its roots existed only in those Mathematica logs.

  dense/results/NSolveValues_results_partial/dim2/roots/roots_deg{2,30}.json
      a later run_NSolve_dense.wls run that stopped early (its summary files
      are empty). Degree 2 is restored from git (convert_roots_to_json.py had
      replaced it with the near-identical "2_cont" roots) and degree 30 is
      moved out of NSolveValues_results/dim2/roots, so that folder holds only
      the "2_cont" run that graph_all.py plots for dim 2.

Coordinates are kept as stored: (xd, ..., x1), i.e. "roots_reversed".
"""

import json
import os
import re
import subprocess

import root_accuracy as ra
from convert_roots_to_json import write_and_verify, check_dims, parse_root_lines

NSOLVE = "../dense/results/NSolveValues_results"
RUN1 = "../dense/results/NSolveValues_results_run1"
PARTIAL = "../dense/results/NSolveValues_results_partial"
# Last commit that still had the per-test roots files of the stopped run
PRE_CONVERSION = "5e26a4103"


def convert_run1():
    pat = re.compile(r"dim(\d+)_deg(\d+)_output\.txt")
    for name in sorted(os.listdir(NSOLVE)):
        m = pat.fullmatch(name)
        if not m:
            continue
        dim, deg = map(int, m.groups())
        src = os.path.join(NSOLVE, name)
        with open(src) as f:
            n_marks = len(re.findall(r'^\d+"end"', f.read(), re.M))
        data = {t: [list(r) for r in rs]
                for t, rs in ra._load_mathematica_output(src).items()}
        if len(data) != n_marks:
            raise ValueError(f"{src}: {n_marks} tests but parsed {len(data)}")
        check_dims(data, dim, src)
        out = os.path.join(RUN1, f"dim{dim}", "roots", f"roots_deg{deg}.json")
        write_and_verify(out, data)
        print(f"{src} ({len(data)} tests) -> {out}")


def git(*args):
    """Run git from the repo root: ls-tree output is limited to the cwd."""
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=True).stdout.strip()
    return root, subprocess.run(["git", *args], cwd=root,
                                capture_output=True, text=True, check=True).stdout


def split_partial_run():
    # degree 2: rebuild from the per-test files as they were before conversion
    root, _ = git("rev-parse", "--show-toplevel")
    rel = os.path.relpath(os.path.realpath(NSOLVE), root) + "/dim2/deg2"
    names = git("ls-tree", "--name-only", f"{PRE_CONVERSION}:{rel}")[1].split()
    if len(names) != 300:
        raise ValueError(f"expected 300 roots files in {rel}, found {len(names)}")
    data = {}
    for name in names:
        t = int(re.fullmatch(r"roots_test(\d+)\.txt", name).group(1))
        text = git("show", f"{PRE_CONVERSION}:{rel}/{name}")[1]
        data[t] = parse_root_lines(text.splitlines(), f"{rel}/{name}")
    check_dims(data, 2, rel)
    out = os.path.join(PARTIAL, "dim2", "roots", "roots_deg2.json")
    write_and_verify(out, data)
    print(f"{rel} @ {PRE_CONVERSION} ({len(data)} tests) -> {out}")

    # degree 30: only this run has it in dim2/roots; move it out
    src = os.path.join(NSOLVE, "dim2", "roots", "roots_deg30.json")
    if os.path.exists(src):
        dst = os.path.join(PARTIAL, "dim2", "roots", "roots_deg30.json")
        os.replace(src, dst)
        print(f"{src} -> {dst}")


if __name__ == "__main__":
    convert_run1()
    split_partial_run()
