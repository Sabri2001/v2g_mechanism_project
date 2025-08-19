import os
import json
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
import numpy as np
import re


def extract_metadata(path):
    """
    Extract disconnection_time and alpha_factor from path.
    Example: vcg_time_0.25_af_1.0_20250801_094342
    """
    match = re.search(r"vcg_time_(\d+\.?\d*)_af_(\d+\.?\d*)", path)
    if match:
        disconnection_time = float(match.group(1))
        alpha_factor = float(match.group(2))
        return disconnection_time, alpha_factor
    else:
        return None, None

def load_vcg(log_path):
    with open(log_path, 'r') as f:
        log = json.load(f)
    return log['results']['vcg_tax']['0']

def collect_data(base_dir="../outputs/tsg/xp_5"):
    data = defaultdict(list)

    for path in glob(os.path.join(base_dir, "vcg_time_*_af_*")):
        disconnection_time, alpha_factor = extract_metadata(path)
        if disconnection_time is None:
            print(f"Skipping path {path} - could not extract metadata.")
            continue

        run_dirs = glob(os.path.join(path, "*", "centralized", "run_*"))
        print(f"{path} | centralized: Found {len(run_dirs)} runs")

        for run_dir in run_dirs:
            log_file = os.path.join(run_dir, "log.json")
            if os.path.exists(log_file):
                vcg = load_vcg(log_file)
                data[(disconnection_time, alpha_factor)].append(vcg)

    return data

def compute_mean_vcg(data):
    """
    For each (disconnection_time, alpha_factor), compute mean vcg tax for EV 0.
    Returns: dict[(disconnection_time, alpha_factor)] = mean vcg_tax
    """
    result = {}
    for (disconnection_time, alpha_factor), vcgs in data.items():
        mean_vcg = np.mean(vcgs)
        result[(disconnection_time, alpha_factor)] = mean_vcg
    return result

def plot_vcg(mean_vcg):
    fig, ax = plt.subplots(figsize=(11, 6))

    # Extract sorted unique values for grid
    disconnection_times = sorted(set(k[0] for k in mean_vcg.keys()))
    alpha_factors = sorted(set(k[1] for k in mean_vcg.keys()))

    # Prepare matrix for color values
    grid = np.full((len(alpha_factors), len(disconnection_times)), np.nan)

    for i, alpha in enumerate(alpha_factors):
        for j, dtime in enumerate(disconnection_times):
            grid[i, j] = mean_vcg.get((dtime, alpha), np.nan)

    # Plot heatmap in index coordinates
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="viridis"
    )

    # Label x-axis with disconnection times
    ax.set_xticks(range(len(disconnection_times)))
    ax.set_xticklabels(disconnection_times)
    ax.set_xlabel("Desired disconnection time (h)")

    # Remove y-axis ticks (we’ll add alpha labels manually)
    ax.set_yticks([])

    # Add alpha labels next to rows
    for i, alpha in enumerate(alpha_factors):
        ax.text(
            -0.6, i, f"{round(alpha*32)}",
            va="center", ha="right", fontsize=10
        )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Mean VCG tax ($)")

    # Add a "legend-like" label for alpha values
    # (using axes coordinates so it won't be clipped)
    ax.text(
        -0.1, 0.5, "Inflexibility (cost of 1-hour delay in $)",
        rotation=90, va="center", ha="center", fontsize=12,
        transform=ax.transAxes   # <-- anchor to axes, not data
    )

    # Increase left margin so there's room for labels
    plt.subplots_adjust(left=0.3)

    plt.savefig("../outputs/tsg/xp_5/vcg.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    data = collect_data()
    print(f"Collected data for {len(data)} configurations.")
    mean_vcg = compute_mean_vcg(data)
    plot_vcg(mean_vcg)
