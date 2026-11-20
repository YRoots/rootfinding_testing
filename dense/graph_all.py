import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_codes"))
import root_accuracy  # noqa: E402

plt.rcParams["figure.dpi"] = 400

# ----------------------------------------------------------------------
# CONFIG — edit these to control what gets plotted
# ----------------------------------------------------------------------

DIMS_TO_PLOT = [2, 3, 4, 5]

MIN_DEGREE = 2

DIM_MAX_DEGREES = {
    2: 30,
    3: 20,
    4: 10,
    5: 5,
}

# Which tests to check, per dim/degree:
#   None              every test that has results
#   100               tests 1-100
#   range(1, 51)      tests 1-50 (any list/range of test numbers works)
ROOT_TESTS = 100

# Marker drawn for a method that only has a single degree of data, where
# a line would have nothing to connect and so would not show up at all.
SINGLE_POINT_MARKER = "o"

METRICS = {
    "avg_times": {"title": "Dense Average Times", "ylabel": "Time Log Scale"},
    "avg_resids": {"title": "Dense Average Residuals", "ylabel": "Residual"},
    "max_resids": {"title": "Dense Max Residuals", "ylabel": "Residual"},
    "sum_resids": {"title": "Dense Sum of Residuals", "ylabel": "Residual"},
}

# One entry per method/solver.
#
#   "path"      template filled in with dim/arg; the default for the method
#   "arg_paths" per-metric override, e.g. times stored as .npy
#   "dim_paths" per-dimension override, e.g. NSolveValues used a different
#               layout for the low dimensions
#   "dims"      restricts a method to specific dimensions (e.g. Chebfun only
#               has dim-2 results); omit to include it for every dimension
#
#   "roots"     directory holding this method's roots (roots/roots_deg*.json);
#               methods with one get root-accuracy markers, see ROOT MARKERS
#   "roots_reversed"  True if the roots store coordinates as (xd, ..., x1)
#   "dim_roots" per-dimension override of "roots" (like "dim_paths")
#
# A path ending in .npy is loaded with np.load; anything else is parsed as
# one plain value per line.
#
# Results come from the newer runs (folders named *_new / *2 /
# yroots_results-optimize), which all solve the same zero-constant-term
# systems, so their roots can be compared with each other.
METHODS = {
    "Yroots": {
        "path": "results/yroots_results_new/dim{dim}/{arg}.txt",
        "roots": "results/yroots_results_new/dim{dim}",
        "color": "royalblue",
    },
    "Jroots": {
        "path": "results/jroots_results_new/dim{dim}/{arg}.txt",
        "roots": "results/jroots_results_new/dim{dim}",
        "roots_reversed": True,
        "color": "navy",
    },
    "MTV": {
        "path": "results/telen_results_new/dim{dim}/{arg}.txt",
        "roots": "results/telen_results_new/dim{dim}",
        "color": "tab:purple",
    },
    "Bertini": {
        "path": "results/bertini_results_new/v1.6/dim{dim}/{arg}.txt",
        "roots": "results/bertini_results_new/v1.6/dim{dim}",
        "color": "tab:green",
    },
    "NSolveValues": {
        "path": "results/NSolveValues_results/dim{dim}/{arg}.txt",
        "dim_paths": {
            2: "results/NSolveValues_results/dim{dim}_{arg}2_cont.txt",
            3: "results/NSolveValues_results/dim{dim}_{arg}2_cont.txt",
        },
        # dims 2-3 are plotted from the older "2_cont" run; its roots were
        # extracted from dim{d}_deg{n}_output2.txt into dim{d}/roots/.
        "roots": "results/NSolveValues_results/dim{dim}",
        "roots_reversed": True,
        "color": "tab:red",
    },
    "Reduce": {
        "path": "results/reduce_results/dim{dim}_{arg}.txt",
        # No "roots": the dense Reduce runs never saved their roots
        # (dim{d}_deg{n}_output.txt only holds the polynomials).
        "color": "m",
    },
    "Chebfun": {
        "path": "results/chebfun_results/dim{dim}/{arg}.txt",
        "roots": "results/chebfun_results/dim{dim}",
        "roots_reversed": True,
        "color": "tab:orange",
        "dims": {2},
    },
    "HomotopyContinuation": {
        # The zero-constant-term run (run_hc_combined.jl, suite "dense").
        # The older hc_results/ kept the constant term, so it solved
        # different systems and isn't used.
        "path": "results/hc_results_new/dim{dim}/{arg}.txt",
        "roots": "results/hc_results_new/dim{dim}",
        "color": "tab:brown",
    },
}

# Extra roots that vote on which roots are real but are never plotted.
# Methods only vote within their "roots_group" (default: the newer runs).
ROOT_VOTERS = {}

# Flip a method to False to leave it out of every graph without
# deleting its config above. Keys must match METHODS.
METHODS_ENABLED = {
    "Yroots": True,
    "Jroots": True,
    "MTV": True,
    "Bertini": True,
    "NSolveValues": True,
    "Reduce": True,
    "Chebfun": True,
    "HomotopyContinuation": True,
}

# ----------------------------------------------------------------------
# ROOT MARKERS
# ----------------------------------------------------------------------
# For every method with a "roots" entry, each degree's point gets a marker
# shaped by how its roots compare with the other methods' (consensus: a root
# is real if at least 2 methods found it; see test_codes/root_accuracy.py):
#   o  all roots found     v  missed roots
#   ^  spurious roots      D  missed + spurious
# Non-o markers are labelled with the number of tests at that degree that
# had the problem. Every solver with saved roots gets markers; Reduce has
# none saved for dense.
SHOW_ROOT_MARKERS = True

# ----------------------------------------------------------------------
# CORE LOGIC
# ----------------------------------------------------------------------

def parse_values(raw_text):
    """Turn a results file's raw text into a list of absolute float values."""
    raw_text = raw_text.replace(" ", "").replace("*^", "e")
    return [abs(float(v)) for v in raw_text.split("\n") if v]


def resolve_path(config, dim, arg):
    """Pick the path template for this method/dim/arg and fill it in."""
    template = config["path"]
    if dim in config.get("dim_paths", {}):
        template = config["dim_paths"][dim]
    if arg in config.get("arg_paths", {}):
        template = config["arg_paths"][arg]
    return template.format(dim=dim, arg=arg)


def load_method_series(config, dim, arg, max_deg):
    """Load one method's series for a given dim/arg, truncated to max_deg."""
    path = resolve_path(config, dim, arg)
    try:
        if path.endswith(".npy"):
            values = np.abs(np.load(path)).ravel().tolist()
        else:
            with open(path) as f:
                values = parse_values(f.read())
    except FileNotFoundError:
        print(f"  [skip] no data at {path}")
        return None
    return values[:max_deg - MIN_DEGREE + 1]


def compute_root_status(dim, max_deg):
    """{method label: {deg: status}} for every method with roots. Disabled
    methods still vote on which roots are real, so hiding a line does not
    change the other lines' markers."""
    configs = {**METHODS, **ROOT_VOTERS}
    roots_dirs = {
        label: config.get("dim_roots", {}).get(dim, config["roots"]).format(dim=dim)
        for label, config in configs.items()
        if "roots" in config and ("dims" not in config or dim in config["dims"])
    }
    reversed_methods = {label for label in roots_dirs
                        if configs[label].get("roots_reversed")}
    groups = {label: configs[label].get("roots_group") for label in roots_dirs}
    return root_accuracy.root_status(
        roots_dirs, dim, range(MIN_DEGREE, max_deg + 1), reversed_methods, groups,
        tests=ROOT_TESTS)


def plot_metric(dim, arg, max_deg, root_status=None):
    info = METRICS[arg]
    fig, ax = plt.subplots()
    ax.set_yscale("log")

    for label, config in METHODS.items():
        if not METHODS_ENABLED.get(label, True):
            continue
        if "dims" in config and dim not in config["dims"]:
            continue
        series = load_method_series(config, dim, arg, max_deg)
        if not series:
            continue
        x = np.arange(MIN_DEGREE, MIN_DEGREE + len(series))
        # A one-degree series has no segment to draw, so mark the point
        # instead of drawing an invisible line.
        statuses = (root_status or {}).get(label)
        marker = SINGLE_POINT_MARKER if len(series) == 1 and not statuses else None
        ax.plot(x, series, color=config["color"], marker=marker, label=label)
        if statuses:
            root_accuracy.draw_markers(ax, x, series, statuses, config["color"])

    ax.set_ylabel(info["ylabel"])
    ax.set_xlabel("Degree")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    # Both legends sit outside the axes so they never cover data.
    method_legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1),
                              fontsize=6, title="Method", title_fontsize=6)
    if root_status and any(root_status.values()):
        ax.add_artist(method_legend)
        ax.legend(handles=root_accuracy.marker_legend_handles(),
                  loc="lower left", bbox_to_anchor=(1.02, 0),
                  fontsize=6, title="Roots", title_fontsize=6)
    plt.title(f"Dim {dim} {info['title']}")

    out_dir = f"graphs/dim{dim}"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/dim{dim}_{arg}.png", bbox_inches="tight")
    plt.close(fig)


def main():
    missing = set(METHODS) - set(METHODS_ENABLED)
    if missing:
        raise ValueError(
            f"METHODS_ENABLED is missing entries for: {sorted(missing)}"
        )

    for dim in DIMS_TO_PLOT:
        if dim not in DIM_MAX_DEGREES:
            raise ValueError(
                f"No max-degree entry for dim {dim} — add one to DIM_MAX_DEGREES."
            )
        max_deg = DIM_MAX_DEGREES[dim]
        root_status = compute_root_status(dim, max_deg) if SHOW_ROOT_MARKERS else None
        for arg in METRICS:
            plot_metric(dim, arg, max_deg, root_status)


if __name__ == "__main__":
    main()
