#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging


logging.basicConfig(level=logging.INFO)

def plot_fake_alpha_tau_bars(chart_data, output_path):
    """
    Bar chart with bars for each scenario,
    plus std error bars from the run-to-run variation.
    """
    sns.set(style="whitegrid")
    
    xp_labels = list(chart_data.keys()) 
    means = [320, 320] # [325, 840]
    stds = [285, 285] #  [290, 560]

    x_positions = np.arange(len(xp_labels))
    width = 0.5  # single bar width

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot bars with error bars
    ax.bar(
        x_positions,
        means,
        width,
        yerr=stds,
        capsize=5,
        color=["blue", "orange", "green"][:len(xp_labels)],
        alpha=0.7
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xp_labels, fontsize=25)
    ax.set_xlabel("Scenario", fontsize=25, labelpad=15)
    ax.set_ylabel("Total Cost ($)", fontsize=25, labelpad=20)
    
    # Set tick label sizes for both axes
    ax.tick_params(axis='both', which='major', labelsize=25)

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logging.info(f"Plot saved to {output_path}")

def main():
    chart_data = {
        "Bid": 0,
        "No bid": 0
    }

    output_path = "bar.png"
    plot_fake_alpha_tau_bars(chart_data, output_path)


if __name__ == "__main__":
    main()
