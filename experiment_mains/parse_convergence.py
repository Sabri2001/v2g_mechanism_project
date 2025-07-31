#!/usr/bin/env python
# Filename: plot_admm_iterations.py

import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

logging.basicConfig(level=logging.INFO)

def gather_admm_data(base_path):
    """
    Scans subfolders of the form run_{index}_nm{nb} in base_path,
    reads 'log.json', and collects 'admm_iterations' and 'nu_multiplier'
    from each run.

    Returns:
        dict: {
            "coordinated": [
                {
                    "admm_iterations": int,
                    "nu_multiplier": float
                },
                ...
            ]
        }
    """
    results_by_experiment = {"coordinated": []}

    # Traverse all subfolders under base_path
    for item in os.listdir(base_path):
        subfolder = os.path.join(base_path, item)
        if os.path.isdir(subfolder):
            # Look for log.json
            log_file = os.path.join(subfolder, "log.json")
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r") as f:
                        data = json.load(f)
                    # Extract results
                    admm_iterations = data.get("results", {}).get("admm_iterations", None)
                    nu_multiplier = data.get("results", {}).get("nu_multiplier", None)
                    
                    # If both exist, add to our data structure
                    if admm_iterations is not None and nu_multiplier is not None:
                        results_by_experiment["coordinated"].append({
                            "admm_iterations": admm_iterations,
                            "nu_multiplier": nu_multiplier
                        })
                except Exception as e:
                    logging.error(f"Error reading {log_file}: {e}")
            else:
                logging.warning(f"'{log_file}' not found.")
    
    return results_by_experiment


def plot_admm_iterations_violin(results_by_experiment, output_path):
    """
    Produces a vertical violin plot of the ADMM iteration counts
    for the 'coordinated' experiment, separated by nu_multiplier.
    nu_multiplier = 1.0 is renamed "Constant" (blue),
    and nu_multiplier = 1.1 is renamed "Adaptive" (orange).

    Args:
        results_by_experiment (dict): Dict of lists of run results.
        output_path (str): Path to save the output plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import logging

    sns.set(style="whitegrid")

    coordinated_runs = results_by_experiment.get("coordinated", [])
    data_rows = []
    for run_results in coordinated_runs:
        iters = run_results.get("admm_iterations", None)
        nu_mult = run_results.get("nu_multiplier", None)
        if iters is not None and nu_mult is not None and nu_mult in [1.0, 1.1]:
            data_rows.append({
                "nu_multiplier": nu_mult,
                "admm_iterations": iters
            })

    if not data_rows:
        logging.warning("No ADMM iteration data (with nu_multiplier=1.0 or 1.1) found. Skipping plot.")
        return

    df = pd.DataFrame(data_rows)
    
    # Map numeric nu_multiplier values to descriptive strings.
    mapping = {1.0: "Constant", 1.1: "Adaptive"}
    df["nu_multiplier"] = df["nu_multiplier"].replace(mapping)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Define custom palette: blue for Constant, orange for Adaptive.
    palette = {"Constant": "blue", "Adaptive": "orange"}
    
    sns.violinplot(
        data=df,
        x="nu_multiplier",
        y="admm_iterations",
        cut=0,
        palette=palette,
        ax=ax
    )
    
    # Set x and y axis labels with fontsize 15 and adjusted label padding.
    ax.set_xlabel("", fontsize=15, labelpad=15)
    ax.set_ylabel("ADMM Iterations", fontsize=15, labelpad=15)
    
    # Set tick labels for both axes to fontsize 15.
    ax.tick_params(axis='both', which='major', labelsize=15)
    
    # If a legend were added, ensure its fonts are also set to 15.
    # For example:
    # legend = ax.legend(fontsize=15)
    # legend.get_title().set_fontsize(15)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Vertical violin plot of ADMM iterations saved to {output_path}")


if __name__ == "__main__":
    # Base directory containing run_{index}_nm{nb} subfolders
    base_path = "outputs/final/convergence/convergence/coordinated"
    
    # Output path for the violin plot
    output_plot_path = os.path.join(base_path, "admm_iterations_violin.png")

    # 1) Gather data from the subfolders
    results_by_experiment = gather_admm_data(base_path)

    # 2) Plot and save the violin plot
    plot_admm_iterations_violin(results_by_experiment, output_plot_path)
