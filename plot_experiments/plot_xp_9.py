#!/usr/bin/env python3
"""
Standalone script to generate incentive compatibility grid plot.

Usage:
    python plot_xp_9.py <results_pattern> <target_ev_id> <true_tau> <true_alpha> <tau1> <tau2> ... --- <alpha1> <alpha2> ...

Example:
    python plot_xp_9.py ../outputs/tsg/xp_9/tau_17_alpha_50_20251003_102819 0 19 60 17 18 19 20 21 --- 40 50 60 70 80
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import logging


def find_result_dir(base_pattern, tau, alpha):
    """Find the results directory for a specific tau/alpha bid combination."""
    base_dir = os.path.dirname(base_pattern)
    pattern = os.path.join(base_dir, f"tau_{tau}_alpha_{int(alpha)}_*")
    matches = glob.glob(pattern)
    
    if not matches:
        return None
    
    return max(matches, key=os.path.getctime)

def load_all_results(result_dir):
    """Load log.json files from ALL runs in a results directory."""
    results = []
    centralized_dir = os.path.join(result_dir, "centralized")
    
    if not os.path.exists(centralized_dir):
        logging.warning(f"Centralized directory not found: {centralized_dir}")
        return results
    
    run_dirs = glob.glob(os.path.join(centralized_dir, "run_*"))
    
    for run_dir in sorted(run_dirs):
        log_path = os.path.join(run_dir, "log.json")
        
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                data = json.load(f)
            results.append(data)
    
    if not results:
        logging.warning(f"No log files found in: {result_dir}")
    
    return results

def compute_utility(results, config, target_ev_id, true_tau, true_alpha):
    """
    Compute the true utility for the target EV.
    
    The EV reports a BID (disconnection_time, disconnection_time_flexibility) to the mechanism,
    but its TRUE preferences are (true_tau, true_alpha).
    
    Utility = energy_cost + wear_cost + adaptability_cost + soc_cost
    
    The adaptability cost uses the TRUE values:
        adaptability_cost = 0.5 * true_alpha * (true_tau - actual_tau)²
    
    Args:
        results: Results dictionary from the experiment
        config: Config dictionary from the experiment
        target_ev_id: ID of the target EV
        true_tau: True desired disconnection time (hours) - fixed across all runs
        true_alpha: True flexibility coefficient ($/h²) - fixed across all runs
    
    Returns:
        float: Total utility (cost) for the target EV
    """
    target_ev = None
    for ev in config["evs"]:
        if ev["id"] == target_ev_id:
            target_ev = ev
            break
    
    if target_ev is None:
        raise ValueError(f"EV with id {target_ev_id} not found in config")
    
    # Get actual disconnection time assigned by the mechanism
    print(target_ev_id, type(target_ev_id))
    actual_tau = results["actual_disconnection_time"][f"{target_ev_id}"]
    
    # Compute adaptability cost using TRUE preferences (not the bid)
    adaptability_cost = 0.5 * true_alpha * ((true_tau - actual_tau) ** 2)
    
    # Extract parameters
    granularity = config["granularity"]
    dt = config["dt"]
    nb_time_steps = config["nb_time_steps"]
    market_prices = config["market_prices"]
    
    # Get SoC evolution to back-calculate charging rates
    soc_over_time = results["soc_over_time"][f"{target_ev_id}"]
    energy_efficiency = target_ev["energy_efficiency"]
    battery_wear_coef = target_ev["battery_wear_cost_coefficient"]
    
    # Compute energy and wear costs
    energy_cost = 0.0
    wear_cost = 0.0
    
    for step in range(nb_time_steps):
        # Back out u[step] from SoC dynamics: soc[t+1] = soc[t] + u[t] * dt * eff
        u_step = (soc_over_time[step + 1] - soc_over_time[step]) / (dt * energy_efficiency)
        
        # Energy cost
        price = market_prices[step // granularity]
        energy_cost += price * u_step * dt
        
        # Wear cost
        wear_cost += battery_wear_coef * abs(u_step) * energy_efficiency * dt
    
    # SoC flexibility cost
    soc_cost = results.get("soc_cost_per_ev", {}).get(f"{target_ev_id}", "0")
    
    # Total utility
    total_utility = energy_cost + wear_cost + adaptability_cost + soc_cost
    
    return total_utility

def plot_incentive_lie_grid(tau_values, alpha_values, cost_grid, output_path, 
                             true_tau, true_alpha):
    """
    Plot a heatmap showing utility as a function of bid parameters.
    Highlight the truthful bid cell.
    """
    # sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 20})
    plt.figure(figsize=(10, 8))
    
    cmap = "RdYlGn_r"  # Green = low cost (good), Red = high cost (bad)
    
    ax = sns.heatmap(
        cost_grid,
        xticklabels=tau_values,
        yticklabels=alpha_values,
        cmap=cmap,
        annot=False,
        fmt=".2f",
        cbar_kws={'label': 'Utility ($)'}
    )

    ax.invert_yaxis()
    
    plt.xlabel("Bid: disconnection time (h)", labelpad=15, fontsize=20)
    plt.ylabel("Bid: inflexibility coefficient ($/h²)", labelpad=15, fontsize=20)
    
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Total cost ($)", fontsize=20, labelpad=15)

    # Highlight the truthful bid cell
    if true_tau in tau_values and true_alpha in alpha_values:
        col_index = tau_values.index(true_tau)
        row_index = alpha_values.index(true_alpha)
        rect = Rectangle(
            (col_index, row_index), 1, 1, 
            fill=False, edgecolor='blue', linewidth=4
        )
        ax.add_patch(rect)
        
        plt.text(
            0.03, 0.07, 
            f'Blue box = truthful bid ({true_tau}, {true_alpha})',
            transform=ax.transAxes,
            fontsize=20,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {output_path}")

def main():
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 7:
        print("Usage: python plot_xp_9.py <results_pattern> <target_ev_id> <true_tau> <true_alpha> <tau1> <tau2> ... --- <alpha1> <alpha2> ...")
        print("Example: python plot_xp_9.py ../outputs/tsg/xp_9/tau_17_alpha_50_20251003_102819 0 19 60 17 18 19 20 21 --- 50 55 60 65 70")
        sys.exit(1)
    
    base_pattern = sys.argv[1]
    target_ev_id = int(sys.argv[2])
    true_tau = int(sys.argv[3])
    true_alpha = float(sys.argv[4])
    
    # Find separator
    try:
        sep_idx = sys.argv.index("---")
    except ValueError:
        print("ERROR: Missing '---' separator between tau and alpha values")
        sys.exit(1)
    
    tau_values = [int(x) for x in sys.argv[5:sep_idx]]
    alpha_values = [float(x) for x in sys.argv[sep_idx+1:]]
    
    print(f"{'='*60}")
    print(f"Incentive Compatibility Analysis")
    print(f"{'='*60}")
    print(f"Base pattern: {base_pattern}")
    print(f"Target EV ID: {target_ev_id}")
    print(f"TRUE preferences: tau={true_tau}h, alpha={true_alpha}$/h²")
    print(f"Testing bid values:")
    print(f"  Tau: {tau_values}")
    print(f"  Alpha: {alpha_values}")
    print(f"{'='*60}\n")
    
    # Initialize cost grid
    cost_grid = np.zeros((len(alpha_values), len(tau_values)))
    
    # Process each bid combination
    for i, tau in enumerate(tau_values):
        for j, alpha in enumerate(alpha_values):
            print(f"Processing bid: tau={tau}, alpha={alpha}...")
            
            result_dir = find_result_dir(base_pattern, tau, alpha)
            
            if result_dir is None:
                print(f"  WARNING: Directory not found")
                cost_grid[j, i] = np.nan
                continue
            
            all_data = load_all_results(result_dir)
            
            if not all_data:
                print(f"  WARNING: No results loaded")
                cost_grid[j, i] = np.nan
                continue
            
            print(f"  Loaded {len(all_data)} runs")
            
            # Compute utility for each run using TRUE preferences
            utilities = []
            for run_idx, data in enumerate(all_data):
                utility = compute_utility(
                    data["results"], 
                    data["config"], 
                    target_ev_id, 
                    true_tau, 
                    true_alpha
                )
                utilities.append(utility)
            
            # Average across runs
            avg_utility = np.mean(utilities)
            std_utility = np.std(utilities)
            cost_grid[j, i] = avg_utility
            
            print(f"  Average: ${avg_utility:.2f} (±${std_utility:.2f})\n")
    
    # Analysis
    print(f"{'='*60}")
    print("Analysis Results")
    print(f"{'='*60}")
    
    min_cost = np.nanmin(cost_grid)
    min_idx = np.unravel_index(np.nanargmin(cost_grid), cost_grid.shape)
    best_tau = tau_values[min_idx[1]]
    best_alpha = alpha_values[min_idx[0]]
    
    print(f"Best bid strategy: tau={best_tau}, alpha={best_alpha}")
    print(f"  Cost: ${min_cost:.2f}")
    
    if true_tau in tau_values and true_alpha in alpha_values:
        truthful_idx = (alpha_values.index(true_alpha), tau_values.index(true_tau))
        truthful_cost = cost_grid[truthful_idx]
        
        print(f"\nTruthful bid: tau={true_tau}, alpha={true_alpha}")
        print(f"  Cost: ${truthful_cost:.2f}")
        
        if best_tau == true_tau and best_alpha == true_alpha:
            print("\n✓ INCENTIVE COMPATIBLE: Truthful bidding is optimal!")
        else:
            savings = truthful_cost - min_cost
            pct_savings = (savings / truthful_cost) * 100
            print(f"\n✗ NOT INCENTIVE COMPATIBLE: EV benefits from lying")
            print(f"  Savings from lying: ${savings:.2f} ({pct_savings:.2f}%)")
    else:
        print(f"\nWarning: Truthful bid not in tested grid")
    
    print(f"{'='*60}\n")
    
    # Generate plots
    output_base = os.path.dirname(base_pattern)
    output_png = os.path.join(output_base, "incentive_lie_grid.png")
    output_pdf = os.path.join(output_base, "incentive_lie_grid.pdf")
    
    plot_incentive_lie_grid(tau_values, alpha_values, cost_grid, output_png, true_tau, true_alpha)
    plot_incentive_lie_grid(tau_values, alpha_values, cost_grid, output_pdf, true_tau, true_alpha)
    
    print("Done!")


if __name__ == "__main__":
    main()
