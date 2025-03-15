#!/usr/bin/env python3

import os
import re
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def gather_computation_times(input_folder):
    """
    Recursively go through each subfolder in `input_folder`, look for 
    'centralized/run_*' and 'coordinated/run_*' logs, read 'computation_time',
    and return a DataFrame with columns: ['num_evs', 'method', 'time'].
    """
    all_data = []

    # Pattern to extract the number of EVs from folder names like "5_EVs_2_2_20250315_120353"
    # This regex looks for something like: "5_EVs" or "15_EVs" or "30_EVs"
    evs_pattern = re.compile(r"(\d+)_EVs")

    # List top-level experiment folders, e.g. "5_EVs_...", "15_EVs_...", "30_EVs_..."
    for top_dir in os.listdir(input_folder):
        top_dir_path = os.path.join(input_folder, top_dir)
        if not os.path.isdir(top_dir_path):
            continue

        # Find number of EVs from folder name
        match = evs_pattern.search(top_dir)
        if not match:
            # Skip folders that don't match the pattern
            continue
        num_evs = int(match.group(1))

        # For each method: 'centralized' and 'coordinated'
        for method in ["centralized", "coordinated"]:
            method_dir = os.path.join(top_dir_path, method)
            if not os.path.isdir(method_dir):
                continue

            # Each run is in run_1, run_2, etc.
            for run_dir_name in os.listdir(method_dir):
                run_dir = os.path.join(method_dir, run_dir_name)
                if not os.path.isdir(run_dir):
                    continue

                log_path = os.path.join(run_dir, "log.json")
                if not os.path.isfile(log_path):
                    continue

                # Read log.json and extract computation_time
                with open(log_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Typical structure: data["results"]["computation_time"]
                # Adjust this to your actual JSON structure if it differs.
                try:
                    comp_time = data["results"]["computation_time"]
                except (KeyError, TypeError):
                    continue

                # Append to our collection
                all_data.append({
                    "num_evs": num_evs,
                    "method": method,
                    "time": comp_time
                })

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(all_data)
    return df

def plot_computation_times(df, output_path="computation_times.png"):
    """
    Given a DataFrame with ['num_evs', 'method', 'time'],
    make a boxplot with x-axis=[5,15,30] (sorted), hue=method,
    and log-scale y-axis. Adds extra padding between labels and axes.
    """
    sns.set(style="whitegrid")

    unique_evs = sorted(df["num_evs"].unique())

    plt.figure(figsize=(8, 5))

    ax = sns.boxplot(
        x="num_evs",
        y="time",
        hue="method",
        data=df,
        order=unique_evs,
        palette=["#1f77b4", "#ff7f0e"],
        width=0.6
    )

    # Increase padding with 'labelpad'
    ax.set_xlabel("Number of EVs", fontsize=12, labelpad=15)
    ax.set_ylabel("Computation Time (s)", fontsize=12, labelpad=15)
    ax.set_title("Computation Times: Centralized vs. Coordinated", fontsize=13)

    ax.set_yscale("log")

    ax.legend(title="Method", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    # Example usage
    input_folder = "outputs/final/computation"
    output_plot  = "outputs/final/computation/computation_times.png"

    df_times = gather_computation_times(input_folder)
    if df_times.empty:
        print("No computation times found.")
    else:
        plot_computation_times(df_times, output_plot)
