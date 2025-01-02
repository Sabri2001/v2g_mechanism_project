import os
import json
import re
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_FOLDER = '../outputs/economic_feasibility/base_case'


def extract_factors(folder_name):
    """
    Extract alpha and price factors from folder name.
    Expected format: alpha_{alpha}_price_{price}_{timestamp}
    """
    pattern = r'alpha_(?P<alpha>[0-9.]+)_price_(?P<price>[0-9.]+)_\d+'
    match = re.match(pattern, folder_name)
    if match:
        alpha = float(match.group('alpha'))
        price = float(match.group('price'))
        return alpha, price
    else:
        raise ValueError(f"Folder name '{folder_name}' does not match the expected pattern.")


def compute_mean_v2g_percentage(root_folder):
    """
    Traverse the directory structure starting from root_folder,
    compute mean v2g_percentage for each alpha-price combination.

    Returns:
        DataFrame with alpha as rows, price as columns, and mean v2g_percentage as values.
    """
    root = Path(root_folder)
    data = []

    # Iterate over all subfolders in root
    for subfolder in root.iterdir():
        if subfolder.is_dir():
            try:
                alpha, price = extract_factors(subfolder.name)
            except ValueError as e:
                logging.warning(e)
                continue

            coordinated_scheduling_path = subfolder / 'coordinated_scheduling'
            if not coordinated_scheduling_path.exists():
                logging.warning(f"'coordinated_scheduling' not found in {subfolder}")
                continue

            v2g_fractions = []
            # Iterate over all run folders
            for run_dir in coordinated_scheduling_path.iterdir():
                if run_dir.is_dir():
                    log_file = run_dir / 'log.json'
                    if log_file.exists():
                        try:
                            with open(log_file, 'r', encoding='utf-8') as f:
                                log_data = json.load(f)
                            v2g_fraction = log_data.get('results', {}).get('v2g_fraction')
                            if v2g_fraction is not None:
                                v2g_fractions.append(v2g_fraction)
                            else:
                                logging.warning(f"'v2g_fraction' not found in {log_file}")
                        except json.JSONDecodeError:
                            logging.error(f"Invalid JSON in {log_file}")
                    else:
                        logging.warning(f"'log.json' not found in {run_dir}")

            if v2g_fractions:
                mean_v2g = np.mean(v2g_fractions)
                data.append({'alpha': alpha, 'price': price, 'mean_v2g_percentage': mean_v2g})
            else:
                logging.warning(f"No valid 'v2g_fraction' data found in {subfolder}")

    if not data:
        raise ValueError("No data collected. Please check the directory structure and log.json files.")

    df = pd.DataFrame(data)
    pivot_df = df.pivot_table(index='alpha', columns='price', values='mean_v2g_percentage')
    return pivot_df


def plot_v2g_grid(pivot_df, root_folder):
    """
    Plot a grid of squares where each square's green shade represents the mean v2g_percentage.
    """
    alpha_values = sorted(pivot_df.index)
    price_values = sorted(pivot_df.columns)

    alpha_idx = {alpha: i for i, alpha in enumerate(alpha_values)}
    price_idx = {price: i for i, price in enumerate(price_values)}

    grid = np.zeros((len(alpha_values), len(price_values)))

    # Populate the grid with v2g_percentage values
    for alpha in alpha_values:
        for price in price_values:
            value = pivot_df.at[alpha, price]
            # Ensure no negative values
            value = max(value, 0)
            grid[alpha_idx[alpha], price_idx[price]] = value

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.Greens
    max_value = grid.max()
    norm = plt.Normalize(vmin=0, vmax=max_value)

    ax.set_aspect('equal', 'box')

    # Draw rectangles
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            color = cmap(norm(grid[i, j]))
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            # Annotate the cell
            ax.text(j + 0.5, i + 0.5, f"{grid[i, j]:.1f}%", ha='center', va='center', fontsize=12)

    ax.set_xticks(np.arange(len(price_values)) + 0.5)
    ax.set_yticks(np.arange(len(alpha_values)) + 0.5)
    ax.set_xticklabels(price_values, fontsize=12)
    ax.set_yticklabels(alpha_values, fontsize=12)
    ax.set_xlabel('Price Factor', fontsize=14)
    ax.set_ylabel('Alpha Factor', fontsize=14)
    ax.set_title('V2G as a Percentage of Energy Needed for EV Charging', fontsize=16, pad=20)

    ax.set_xlim(0, len(price_values))
    ax.set_ylim(0, len(alpha_values))
    ax.set_frame_on(False)
    ax.tick_params(length=0)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.08)
    cbar.set_label('Mean V2G Percentage (%)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    cbar_ticks = np.linspace(0, max_value, num=6)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{tick:.1f}%" for tick in cbar_ticks])

    output_path = os.path.join(root_folder, 'v2g_percentage_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Plot saved to {output_path}")


def main(root_folder):
    # Configure logging for this script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    pivot_df = compute_mean_v2g_percentage(root_folder)
    logging.info("Pivot Table of Mean V2G Percentages:\n%s", pivot_df)
    plot_v2g_grid(pivot_df, root_folder)


if __name__ == "__main__":
    main(ROOT_FOLDER)
