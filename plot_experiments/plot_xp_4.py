import os
import json
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
import numpy as np
import re


def extract_metadata(path):
    """
    Extract limit_factor and alpha_factor from path.
    Example: cost_savings_limit_0.25_af_1.0_20250801_094342
    """
    match = re.search(r"cost_savings_limit_(\d+\.?\d*)_af_(\d+\.?\d*)", path)
    if match:
        limit_factor = float(match.group(1))
        alpha_factor = float(match.group(2))
        return limit_factor, alpha_factor
    else:
        return None, None

def load_costs(log_path):
    with open(log_path, 'r') as f:
        log = json.load(f)
    return log['results']['total_cost']

def collect_data(base_dir="../outputs/tsg/xp_4"):
    data = defaultdict(lambda: {"centralized": [], "inflexible_centralized": []})

    for path in glob(os.path.join(base_dir, "cost_savings_limit_*_af_*")):
        limit_factor, alpha_factor = extract_metadata(path)
        if limit_factor is None:
            print(f"Skipping path {path} - could not extract metadata.")
            continue

        for exp_type in ["centralized", "inflexible_centralized"]:
            run_dirs = glob(os.path.join(path, "*", exp_type, "run_*"))
            print(f"{path} | {exp_type}: Found {len(run_dirs)} runs")

            for run_dir in run_dirs:
                log_file = os.path.join(run_dir, "log.json")
                if os.path.exists(log_file):
                    cost = load_costs(log_file)
                    data[(limit_factor, alpha_factor)][exp_type].append(cost)

    return data

def compute_savings(data):
    """
    For each (limit_factor, alpha_factor), compute mean and std of cost savings.
    Returns: dict[limit_factor] = list of (alpha_reduction%, mean_saving, std_saving)
    """
    result = defaultdict(list)
    for (limit_factor, alpha_factor), costs in data.items():
        coord = costs["centralized"]
        inflex = costs["inflexible_centralized"]
        if len(coord) != len(inflex):
            continue  # Skip mismatched runs

        savings = [(u - c) / u * 100 for c, u in zip(coord, inflex)]
        mean_saving = np.mean(savings)
        std_saving = np.std(savings)
        alpha = round(alpha_factor * 31.2)
        result[limit_factor].append((alpha, mean_saving, std_saving))
    
    # Sort x-axis values
    for limit in result:
        result[limit] = sorted(result[limit], key=lambda x: x[0])

    return result

# def plot_savings(savings_by_limit):
#     plt.figure(figsize=(10, 6))

#     for limit, points in sorted(savings_by_limit.items()):
#         x = [p[0] for p in points]
#         y = [p[1] for p in points]
#         yerr = [p[2] for p in points]
#         plt.errorbar(x, y, yerr=yerr, label=f"EVCS power limit: {round(limit)} kW", marker='o', capsize=4)

#     plt.xlabel("Inflexibility (cost of 1-hour delay in $)")
#     plt.ylabel("Cost Savings (%)")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig("../outputs/tsg/xp_4/summary_cost_savings_vs_inflexibility.png")

def plot_savings(savings_by_limit):
    plt.figure(figsize=(10, 6))

    # Collect all distinct inflexibility values
    all_inflex = sorted({p[0] for points in savings_by_limit.values() for p in points})

    for inflex in all_inflex:
        x = []
        y = []
        yerr = []
        for limit, points in sorted(savings_by_limit.items()):
            # find point with this inflexibility
            for p in points:
                if p[0] == inflex:
                    x.append(limit)
                    y.append(p[1])
                    yerr.append(p[2])
                    break
        plt.errorbar(
            x, y, yerr=yerr,
            label=f"Inflexibility: {inflex} $/h^2",
            marker='o', capsize=4
        )

    plt.xlabel("EVCS power limit (kW)")
    plt.ylabel("Cost Savings (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../outputs/tsg/xp_4/summary_cost_savings_vs_inflexibility.pdf")


if __name__ == "__main__":
    data = collect_data()
    print(f"Collected data for {len(data)} configurations.")
    savings_by_limit = compute_savings(data)
    plot_savings(savings_by_limit)
