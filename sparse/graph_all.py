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

# Which dimensions to generate graphs for. Add/remove freely; the max
# degree for each one is looked up from DIM_MAX_DEGREES below.
DIMS_TO_PLOT = [2, 3, 4, 5, 6, 7, 8]

# Max degree to plot, per dimension. Add an entry here before adding a
# new dimension to DIMS_TO_PLOT.
DIM_MAX_DEGREES = {
    2: 30,
    3: 10,
    4: 10,
    5: 10,
    6: 10,
    7: 10,
    8: 10,
}

NONZERO = 3

# Metric files to read, and the titles/axis labels to use for each.
METRICS = {
    "avg_times": {"title": "Sparse Average Times", "ylabel": "Time Log Scale"},
    "avg_resids": {"title": "Sparse Average Residuals", "ylabel": "Residual"},
    "max_resids": {"title": "Sparse Max Residuals", "ylabel": "Residual"},
    "sum_resids": {"title": "Sparse Sum of Residuals", "ylabel": "Residual"},
}

# One entry per method/solver. "path" is a template filled in with
# dim/nonzero/arg. "dims" restricts a method to specific dimensions
# (e.g. Chebfun only has dim-2 results); omit "dims" to include a
# method for every dimension in DIMS_TO_PLOT.
#
# "roots" is the directory holding the method's roots (roots/roots_deg*.json,
# or roots_deg*.txt); methods with one get root-accuracy markers, see ROOT
# MARKERS. "roots_reversed" marks roots stored as (xd, ..., x1).
METHODS = {
    "Yroots": {
        "path": "results/yroots_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "roots": "results/yroots_results/dim{dim}/nonzero{nonzero}",
        "color": "royalblue",
    },
    "Jroots": {
        "path": "results/jroots_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "roots": "results/jroots_results/dim{dim}/nonzero{nonzero}",
        "roots_reversed": True,
        "color": "navy",
    },
    "MTV": {
        "path": "results/telen_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "roots": "results/telen_results/dim{dim}/nonzero{nonzero}",
        "color": "tab:purple",
    },
    "Bertini": {
        "path": "results/bertini_results_new/v1.6/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "roots": "results/bertini_results_new/v1.6/dim{dim}/nonzero{nonzero}",
        "color": "tab:green",
    },
    "NSolveValues": {
        "path": "results/NSolveValues_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "roots": "results/NSolveValues_results/dim{dim}/nonzero{nonzero}",
        "roots_reversed": True,
        "color": "tab:red",
    },
    "Reduce": {
        "path": "results/reduce_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "roots": "results/reduce_results/dim{dim}/nonzero{nonzero}",
        "roots_reversed": True,
        "color": "m",
    },
    "Chebfun": {
        "path": "results/chebfun_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "roots": "results/chebfun_results/dim{dim}/nonzero{nonzero}",
        "color": "tab:orange",
        "dims": {2},
    },
    "Yroots-Lambda": {
        "path": "results/yroots_lambda_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "roots": "results/yroots_lambda_results/dim{dim}/nonzero{nonzero}",
        "color": "teal",
    },
    "Jroots-Lambda": {
        "path": "results/jroots_lambda_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "roots": "results/jroots_lambda_results/dim{dim}/nonzero{nonzero}",
        "color": "darkgoldenrod",
    },
    "Croots": {
        "path": "results/croot_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "roots": "results/croot_results/dim{dim}/nonzero{nonzero}",
        "color": "slategrey",
    },
}

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
    "Yroots-Lambda": False,
    "Jroots-Lambda": False,
    "Croots": False,
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
# had the problem. Disabled methods still vote on which roots are real.
SHOW_ROOT_MARKERS = True

# Which tests to check, per dim/degree:
#   None              every test that has results
#   100               tests 1-100
#   range(1, 51)      tests 1-50 (any list/range of test numbers works)
ROOT_TESTS = None


# ----------------------------------------------------------------------
# CORE LOGIC
# ----------------------------------------------------------------------

def parse_values(raw_text):
    """Turn a results file's raw text into a list of absolute float values."""
    raw_text = raw_text.replace(" ", "").replace("*^", "e")
    return [abs(float(v)) for v in raw_text.split("\n") if v]


def load_method_series(config, dim, nonzero, arg, max_deg):
    """Load one method's series for a given dim/arg, truncated to max_deg."""
    path = config["path"].format(dim=dim, nonzero=nonzero, arg=arg)
    try:
        with open(path) as f:
            values = parse_values(f.read())
    except FileNotFoundError:
        print(f"  [skip] no data at {path}")
        return None
    # Values start at degree 2, so degrees 2..max_deg are the first
    # max_deg - 1 entries.
    return values[:max_deg - 1]


def compute_root_status(dim, max_deg, nonzero):
    """{method label: {deg: status}} for every method with roots."""
    roots_dirs = {
        label: config.get("dim_roots", {}).get(dim, config["roots"]).format(dim=dim, nonzero=nonzero)
        for label, config in METHODS.items()
        if "roots" in config and ("dims" not in config or dim in config["dims"])
    }
    reversed_methods = {label for label in roots_dirs
                        if METHODS[label].get("roots_reversed")}
    return root_accuracy.root_status(
        roots_dirs, dim, range(2, max_deg + 1), reversed_methods,
        tests=ROOT_TESTS)


def plot_metric(dim, arg, max_deg, nonzero, root_status=None):
    info = METRICS[arg]
    fig, ax = plt.subplots()
    ax.set_yscale("log")

    for label, config in METHODS.items():
        if not METHODS_ENABLED.get(label, True):
            continue
        if "dims" in config and dim not in config["dims"]:
            continue
        series = load_method_series(config, dim, nonzero, arg, max_deg)
        if not series:
            continue
        x = np.arange(2, min(len(series), 30) + 2)
        ax.plot(x, series, config["color"], label=label)
        statuses = (root_status or {}).get(label)
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
        root_status = (compute_root_status(dim, max_deg, NONZERO)
                       if SHOW_ROOT_MARKERS else None)
        for arg in METRICS:
            plot_metric(dim, arg, max_deg, NONZERO, root_status)


if __name__ == "__main__":
    main()