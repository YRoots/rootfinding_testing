import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams["figure.dpi"] = 400

# ----------------------------------------------------------------------
# CONFIG — edit these to control what gets plotted
# ----------------------------------------------------------------------

# Which dimensions to generate graphs for. Add/remove freely; the max
# degree for each one is looked up from DIM_MAX_DEGREES below.
DIMS_TO_PLOT = [2, 3, 4, 5]

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
METHODS = {
    "Yroots": {
        "path": "results/yroots_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "color": "royalblue",
    },
    "Yroots2": {
            "path": "results/yroots_results-optimize/dim{dim}/nonzero{nonzero}/{arg}.txt",
            "color": "teal",
        },
    "Jroots": {
        "path": "results/jroots_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "color": "navy",
    },
    "MTV": {
        "path": "results/telen_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "color": "tab:purple",
    },
    "Bertini": {
        "path": "results/bertini_results/v1.6/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "color": "tab:green",
    },
    "NSolveValues": {
        "path": "results/NSolveValues_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "color": "tab:red",
    },
    "Reduce": {
        "path": "results/reduce_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "color": "m",
    },
    "Chebfun": {
        "path": "results/chebfun_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "color": "tab:orange",
        "dims": {2},
    },
    "Yroots-Lambda": {
        "path": "results/yroots_lambda_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "color": "teal",
    },
    "Jroots-Lambda": {
        "path": "results/jroots_lambda_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "color": "darkgoldenrod",
    },
    "Croots": {
        "path": "results/croot_results/dim{dim}/nonzero{nonzero}/{arg}.txt",
        "color": "slategrey",
    },
}

# Flip a method to False to leave it out of every graph without
# deleting its config above. Keys must match METHODS.
METHODS_ENABLED = {
    "Yroots": True,
    "Yroots2": True,
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
    return values[:min(len(values), max_deg)]


def plot_metric(dim, arg, max_deg, nonzero):
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

    ax.set_ylabel(info["ylabel"])
    ax.set_xlabel("Degree")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend()
    plt.title(f"Dim {dim} {info['title']}")

    out_dir = f"graphs/dim{dim}"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/dim{dim}_{arg}.png")
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
        for arg in METRICS:
            plot_metric(dim, arg, max_deg, NONZERO)


if __name__ == "__main__":
    main()