import os
import json
import matplotlib.pyplot as plt
from glob import glob
from collections import defaultdict
import numpy as np
import re


def extract_metadata(path):
    """
    Extract battery_factor and alpha_factor from path.
    Example: cost_savings_soc_bw_0.25_af_1.0_20250801_094342
    """
    match = re.search(r"cost_savings_soc_bw_(\d+\.?\d*)_af_(\d+\.?\d*)", path)
    if match:
        battery_factor = float(match.group(1))
        alpha_factor = float(match.group(2))
        return battery_factor, alpha_factor
    else:
        return None, None

def load_costs(log_path):
    with open(log_path, 'r') as f:
        log = json.load(f)
    return log['results']['total_cost']

def collect_data(base_dir="../outputs/tsg/xp_2"):
    data = defaultdict(lambda: {"centralized": [], "unidirectional_centralized": []})

    for path in glob(os.path.join(base_dir, "cost_savings_soc_bw_*_af_*")):
        battery_factor, alpha_factor = extract_metadata(path)
        if battery_factor is None:
            print(f"Skipping path {path} - could not extract metadata.")
            continue

        for exp_type in ["centralized", "unidirectional_centralized"]:
            run_dirs = glob(os.path.join(path, "*", exp_type, "run_*"))
            print(f"{path} | {exp_type}: Found {len(run_dirs)} runs")

            for run_dir in run_dirs:
                log_file = os.path.join(run_dir, "log.json")
                if os.path.exists(log_file):
                    cost = load_costs(log_file)
                    data[(battery_factor, alpha_factor)][exp_type].append(cost)

    return data

def compute_savings(data):
    """
    For each (battery_factor, alpha_factor), compute mean and std of cost savings.
    Returns: dict[alpha_factor] = list of (bw_reduction%, mean_saving, std_saving)
    """
    result = defaultdict(list)
    for (battery_factor, alpha_factor), costs in data.items():
        coord = costs["centralized"]
        unidir = costs["unidirectional_centralized"]
        if len(coord) != len(unidir):
            continue  # Skip mismatched runs

        savings = [(u - c) / u * 100 for c, u in zip(coord, unidir)]
        mean_saving = np.mean(savings)
        std_saving = np.std(savings)
        bw = battery_factor * 0.13
        result[alpha_factor].append((bw, mean_saving, std_saving))
    
    # Sort x-axis values
    for alpha in result:
        result[alpha] = sorted(result[alpha], key=lambda x: x[0])

    return result

def plot_savings(savings_by_alpha):
    plt.figure(figsize=(10, 6))
    for alpha, points in savings_by_alpha.items():
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        yerr = [p[2] for p in points]
        plt.errorbar(x, y, yerr=yerr, label=f"Inflexibility (cost of 1-hour delay): {round(alpha * 31.2)} $", marker='o', capsize=4)

    plt.xlabel("Battery Wear Cost ($/kWh)")
    plt.ylabel("Cost Savings (%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../outputs/tsg/xp_2/summary_cost_savings_vs_battery_wear.pdf")


if __name__ == "__main__":
    data = collect_data()
    print(f"Collected data for {len(data)} configurations.")
    savings_by_alpha = compute_savings(data)
    plot_savings(savings_by_alpha)
