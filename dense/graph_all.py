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

# Every results file holds one value per degree, starting at this degree.
MIN_DEGREE = 2

# Highest degree to plot, per dimension. Methods with shorter files just
# stop early. Add an entry here before adding a new dimension to
# DIMS_TO_PLOT.
DIM_MAX_DEGREES = {
    2: 30,
    3: 20,
    4: 10,
    5: 5,
}

# Marker drawn for a method that only has a single degree of data, where
# a line would have nothing to connect and so would not show up at all.
SINGLE_POINT_MARKER = "o"

# Metric files to read, and the titles/axis labels to use for each.
METRICS = {
    "avg_times": {"title": "Dense Average Times", "ylabel": "Time Log Scale"},
    "avg_resids": {"title": "Dense Average Residuals", "ylabel": "Residual"},
    "max_resids": {"title": "Dense Max Residuals", "ylabel": "Residual"},
    "sum_resids": {"title": "Dense Sum of Residuals", "ylabel": "Residual"},
}

# One entry per method/solver.
#
#   "path"      template filled in with dim/arg; the default for the method
#   "arg_paths" per-metric override, e.g. Bertini stores times as .npy
#   "dim_paths" per-dimension override, e.g. NSolveValues used a different
#               layout for the low dimensions
#   "dims"      restricts a method to specific dimensions (e.g. Chebfun only
#               has dim-2 results); omit to include it for every dimension
#
# A path ending in .npy is loaded with np.load; anything else is parsed as
# one plain value per line.
METHODS = {
    "Yroots": {
        "path": "results/yroots_results/dim{dim}/{arg}.txt",
        "color": "royalblue",
    },
    "Jroots": {
        "path": "results/jroots_results/dim{dim}/{arg}.txt",
        "color": "navy",
    },
    "MTV": {
        "path": "results/telen_results/dim{dim}_{arg}.txt",
        "color": "tab:purple",
    },
    "Bertini": {
        "path": "results/bertini_results/dim{dim}_{arg}.txt",
        "arg_paths": {
            "avg_times": "results/bertini_results/dim{dim}_{arg}.npy",
        },
        "color": "tab:green",
    },
    "NSolveValues": {
        "path": "results/NSolveValues_results/dim{dim}/{arg}.txt",
        "dim_paths": {
            2: "results/NSolveValues_results/dim{dim}_{arg}2_cont.txt",
            3: "results/NSolveValues_results/dim{dim}_{arg}2_cont.txt",
        },
        "color": "tab:red",
    },
    "Reduce": {
        "path": "results/reduce_results/dim{dim}_{arg}.txt",
        "color": "m",
    },
    "Chebfun": {
        "path": "results/chebfun_results/dim{dim}_{arg}.txt",
        "color": "tab:orange",
        "dims": {2},
    },
    "HomotopyContinuation": {
        "path": "results/hc_results/dim{dim}/{arg}.txt",
        "color": "tab:brown",
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
    "HomotopyContinuation": True,
}


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


def plot_metric(dim, arg, max_deg):
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
        marker = SINGLE_POINT_MARKER if len(series) == 1 else None
        ax.plot(x, series, color=config["color"], marker=marker, label=label)

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
            plot_metric(dim, arg, max_deg)


if __name__ == "__main__":
    main()
