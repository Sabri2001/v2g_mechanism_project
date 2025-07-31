import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)


def collect_vcg_tax_values(base_path):
    """
    Iterates over subfolders in base_path, opens the log.json file in each folder,
    and returns the total VCG tax (in $) for each file.
    """
    tax_values = []
    
    for item in os.listdir(base_path):
        subfolder = os.path.join(base_path, item)
        if os.path.isdir(subfolder):
            log_file = os.path.join(subfolder, "log.json")
            if os.path.exists(log_file):
                try:
                    with open(log_file, "r") as f:
                        data = json.load(f)
                    results = data.get("results", {})
                    vcg_tax_dict = results.get("vcg_tax", {})
                    total_vcg_tax = sum(vcg_tax_dict.values())
                    tax_values.append(total_vcg_tax)
                except Exception as e:
                    logging.error(f"Error reading {log_file}: {e}")
            else:
                logging.warning(f"log.json not found in {subfolder}.")
    return tax_values


def plot_vcg_tax_distribution(vcg_tax_values, output_path, nbins=30):
    """
    Creates a normal histogram of VCG tax values (in $), displayed on a 
    standard  x-axis. 

    Args:
        vcg_tax_values (list): List of raw VCG tax values (in $).
        output_path (str): Path to save the output plot.
        nbins (int): Number of bins for the histogram.
    """
    # Filter out non-positive values, as they cannot be shown on a log scale
    data_to_plot = vcg_tax_values

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot a normal histogram with a chosen number of bins
    ax.hist(data_to_plot, bins=nbins, edgecolor='black')

    # Compute and add vertical line for the mean of the positive values
    mean_val = np.mean(data_to_plot)
    ax.axvline(x=mean_val, color="orange", linestyle="-", linewidth=2, label=f"Mean = {mean_val:.2f}")

    # Create legend, set fonts
    ax.legend(fontsize=15)
    ax.set_xlabel("VCG Tax ($)", fontsize=15, labelpad=10)
    ax.set_ylabel("Count", fontsize=15, labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=15)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Histogram saved to {output_path}")


if __name__ == "__main__":
    # Example usage:
    base_folder = "outputs/final/vcg_budget/2_2_oversubscription/coordinated"
    output_plot_path = "outputs/final/vcg_budget/2_2_oversubscription/vcg_tax_distribution.png"

    # Collect raw VCG tax values (in $)
    vcg_tax_values = collect_vcg_tax_values(base_folder)

    # Plot them on a normal histogram with a log (base 10) x-axis
    plot_vcg_tax_distribution(vcg_tax_values, output_plot_path, nbins=50)
