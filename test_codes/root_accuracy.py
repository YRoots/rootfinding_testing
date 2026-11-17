"""
Per-degree root accuracy for each method, used by dense/ and sparse/
graph_all.py to choose marker shapes.

The "true" roots of a test are decided by consensus: every method's roots
inside [-1, 1]^d are clustered (two roots are the same if every coordinate
agrees within TOL), and a cluster counts as a real root when at least
CONSENSUS_MIN methods found it. Then for each method and test:

    missed   = real roots this method did not find
    spurious = roots this method returned that are not real (including a
               real root listed more than once)

A test is only judged when at least CONSENSUS_MIN methods have results for
it; otherwise there is nothing to compare against.

Roots are read from the first of these that exists under a roots dir:
    roots/roots_deg{deg}.json     {test number: [roots]}, written by every
                                  run_*_combined script
    roots_deg{deg}.txt            Bertini's or Croots's text output
    deg{deg}/roots_test{N}.txt    one file per test, one root per line
    deg{deg}/test{N}_roots.txt    (NSolveValues and Chebfun)

Methods can be split into groups that only vote among themselves, for runs
that solved different systems (e.g. dense HC, which only has results for
the original systems, is judged against the original Yroots/Jroots runs).
"""
import json
import os
import re

import numpy as np

# Two roots are the same root if every coordinate agrees within TOL. 1e-6
# is loose enough for solvers that return slightly less precise roots
# (Chebfun's are typically ~1e-9 off, up to ~2e-8) and still far tighter
# than the spacing between distinct roots of these random systems.
TOL = 1e-6
CONSENSUS_MIN = 2
DOMAIN_RADIUS = 1.0
DOMAIN_TOL = 1e-8


def _parse_bertini_txt(lines, dim):
    results, i, n = {}, 0, len(lines)
    while i < n:
        if lines[i].startswith("===== TEST"):
            test = int(lines[i].split()[2])
            i += 1
            while not lines[i]:
                i += 1
            count = int(lines[i].split()[0])
            i += 1
            roots = []
            for _ in range(count):
                while not lines[i]:
                    i += 1
                coords = []
                for _ in range(dim):
                    coords.append(float(lines[i].split()[0]))  # real part
                    i += 1
                roots.append(tuple(coords))
            results[test] = roots
        else:
            i += 1
    return results


def _parse_croots_txt(lines):
    results, test = {}, None
    for line in lines:
        if line.startswith("Test"):
            test = int(line.split()[1])
            results[test] = []
        elif line and test is not None:
            results[test].append(tuple(float(v) for v in line.split()))
    return results


_PER_TEST_FILE = re.compile(r"^(?:roots_test(\d+)|test(\d+)_roots)\.txt$")


def _load_per_test_dir(deg_dir):
    """deg{deg}/ folder with one roots file per test. A file that can't be
    parsed (e.g. Mathematica's two-line exponent layout) leaves that test
    out, so it is simply not judged for this method."""
    results = {}
    for name in os.listdir(deg_dir):
        m = _PER_TEST_FILE.match(name)
        if not m:
            continue
        test = int(m.group(1) or m.group(2))
        try:
            with open(os.path.join(deg_dir, name)) as f:
                results[test] = [tuple(float(v) for v in ln.split())
                                 for ln in f if ln.strip() and not ln.startswith("#")]
        except ValueError:
            print(f"  [skip] can't parse {os.path.join(deg_dir, name)}")
    return results


def load_roots(roots_dir, dim, deg, reverse=False):
    """{test number: [root tuples]} for one degree, or None if no file."""
    json_path = os.path.join(roots_dir, "roots", f"roots_deg{deg}.json")
    txt_path = os.path.join(roots_dir, f"roots_deg{deg}.txt")
    deg_dir = os.path.join(roots_dir, f"deg{deg}")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = {int(k): [tuple(r) for r in v] for k, v in json.load(f).items()}
    elif os.path.exists(txt_path):
        with open(txt_path) as f:
            lines = [ln.strip() for ln in f]
        if any(ln.startswith("===== TEST") for ln in lines):
            data = _parse_bertini_txt(lines, dim)
        else:
            data = _parse_croots_txt(lines)
    elif os.path.isdir(deg_dir):
        data = _load_per_test_dir(deg_dir)
        if not data:
            return None
    else:
        return None
    if reverse:
        data = {t: [r[::-1] for r in roots] for t, roots in data.items()}
    return data


def _in_domain(root):
    return all(abs(v) <= DOMAIN_RADIUS + DOMAIN_TOL for v in root)


def _close(a, b):
    return all(abs(x - y) <= TOL for x, y in zip(a, b))


def degree_status(roots_by_method):
    """
    roots_by_method: {method: {test: [roots]}} for one dim/degree.
    Returns {method: {"tests": n, "missed": k, "spurious": k, "bad": k}},
    each count being the number of judged tests with that problem.
    """
    status = {m: {"tests": 0, "missed": 0, "spurious": 0, "bad": 0}
              for m in roots_by_method}
    all_tests = set().union(*(set(d) for d in roots_by_method.values()))
    for test in all_tests:
        present = [m for m, d in roots_by_method.items() if test in d]
        if len(present) < CONSENSUS_MIN:
            continue
        clusters = []  # [representative root, set of methods, count per method]
        extra = {m: 0 for m in present}  # duplicates of a root already listed
        for m in present:
            for root in roots_by_method[m][test]:
                if not _in_domain(root):
                    continue
                for c in clusters:
                    if _close(root, c[0]):
                        if m in c[1]:
                            extra[m] += 1
                        else:
                            c[1].add(m)
                        break
                else:
                    clusters.append([root, {m}])
        for m in present:
            missed = any(len(c[1]) >= CONSENSUS_MIN and m not in c[1] for c in clusters)
            spurious = extra[m] > 0 or any(
                len(c[1]) < CONSENSUS_MIN and m in c[1] for c in clusters)
            s = status[m]
            s["tests"] += 1
            s["missed"] += missed
            s["spurious"] += spurious
            s["bad"] += missed or spurious
    return status


def root_status(roots_dirs, dim, degrees, reversed_methods=(), groups=None):
    """
    roots_dirs: {method: roots directory for this dim}.
    groups: optional {method: group name}; methods only vote with others in
    the same group. Methods not listed share the default group.
    Returns {method: {deg: status dict}} for every degree where the method
    had roots and was judged against at least one other method.
    """
    groups = groups or {}
    out = {m: {} for m in roots_dirs}
    for deg in degrees:
        loaded = {}
        for m, d in roots_dirs.items():
            roots = load_roots(d, dim, deg, reverse=m in reversed_methods)
            if roots is not None:
                loaded.setdefault(groups.get(m), {})[m] = roots
        for group in loaded.values():
            if len(group) < CONSENSUS_MIN:
                continue
            for m, s in degree_status(group).items():
                if s["tests"]:
                    out[m][deg] = s
    return out


# ----------------------------------------------------------------------
# PLOTTING HELPERS (shared by dense/ and sparse/ graph_all.py)
# ----------------------------------------------------------------------

# Marker shape per outcome, and the text used for it in the legend.
MARKERS = {
    "ok": ("o", "all roots found"),
    "missed": ("v", "missed roots"),
    "spurious": ("^", "spurious roots"),
    "both": ("D", "missed + spurious"),
}
MARKER_SIZE = 4
LABEL_FONTSIZE = 5


def outcome(status):
    if status["missed"] and status["spurious"]:
        return "both"
    if status["missed"]:
        return "missed"
    if status["spurious"]:
        return "spurious"
    return "ok"


def draw_markers(ax, degrees, values, statuses, color):
    """
    Draw one marker per degree that has a root status, shaped by outcome,
    with the number of affected tests written beside any non-ok marker.
    degrees/values are the plotted line's points.
    """
    for deg, val in zip(degrees, values):
        s = statuses.get(int(deg))
        if s is None or not np.isfinite(val) or val <= 0:
            continue
        kind = outcome(s)
        ax.plot(deg, val, marker=MARKERS[kind][0], markersize=MARKER_SIZE,
                color=color, markeredgecolor="black", markeredgewidth=0.4,
                linestyle="none", zorder=3)
        if kind != "ok":
            ax.annotate(str(s["bad"]), (deg, val), textcoords="offset points",
                        xytext=(3, 3), fontsize=LABEL_FONTSIZE, color=color)


def marker_legend_handles():
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=m, color="grey", markeredgecolor="black",
                      markeredgewidth=0.4, linestyle="none", label=text)
               for m, text in MARKERS.values()]
    handles.append(Line2D([], [], linestyle="none",
                          label="n = # of tests affected"))
    return handles
