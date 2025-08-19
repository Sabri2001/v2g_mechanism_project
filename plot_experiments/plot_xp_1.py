import argparse
import json
import matplotlib.pyplot as plt
import os
import numpy as np


def load_log(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def main(log_list_file, ev_index):
    with open(log_list_file, 'r') as f:
        log_paths = [line.strip() for line in f if line.strip()]

    if len(log_paths) != 2:
        print("Expected exactly 2 log files: one for regular and one for low battery wear.")
        return

    logs = []
    labels = ["SoC - low battery wear", "SoC - regular battery wear"]

    for path in log_paths:
        logs.append(load_log(path))

    # Extract market prices (should be same for both)
    market_prices = logs[0]['config']['market_prices']
    market_hours = [h + 0.5 for h in range(10, 22)] 

    # Extract SOC data (15-min granularity = 4 points per hour)
    socs_percent = []
    time_range = logs[0]['config']['time_range']  # [10, 22]
    start, end = time_range
    n_steps = (end - start) * 4 + 1 # 4 per hour

    time_15min = np.linspace(start, end, n_steps)

    for log in logs:
        evs = log['config']['evs']
        battery_capacity = next(ev for ev in evs if ev['id'] == ev_index)['battery_capacity']
        soc = log['results']['soc_over_time'][str(ev_index)]
        soc_percent = [(v / battery_capacity) * 100 for v in soc]
        socs_percent.append(soc_percent)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot market prices (hourly)
    ax1.plot(market_hours, market_prices, color='tab:blue', label='Market prices')
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Price ($/kWh)", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_xticks(range(10, 23))
    ax1.set_xlim(10, 22)

    # Plot SOC (as percent) on same plot with second y-axis
    ax2 = ax1.twinx()
    ax2.plot(time_15min, socs_percent[0], label=labels[0], color='tab:orange')
    ax2.plot(time_15min, socs_percent[1], label=labels[1], color='tab:green')
    ax2.set_ylabel("SOC (%)", color='tab:purple')
    ax2.tick_params(axis='y', labelcolor='tab:purple')
    ax2.set_xlim(10, 22)

    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'../outputs/tsg/xp_1/market_prices_and_soc_ev_{ev_index}.pdf')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_list', type=str, required=True, help='Text file with paths to log.json files')
    parser.add_argument('--ev_index', type=int, required=True, help='Index of EV to plot')
    args = parser.parse_args()
    main(args.log_list, args.ev_index)
