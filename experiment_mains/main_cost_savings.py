import os
import sys
import logging
import copy
import numpy as np
from datetime import datetime

from config.config_handler import ConfigHandler
from experiments.coordinated_scheduling_experiment import CoordinatedSchedulingExperiment
from experiments.unidirectional_coordinated_scheduling_experiment import UnidirectionalCoordinatedSchedulingExperiment
from results.plot_handler import PlotHandler


def run_single_experiment(experiment_class, config):
    experiment = experiment_class(config)
    return experiment.run()

def main(config_path):
    # 1) Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 2) Load base config to see how many runs we want
    base_handler = ConfigHandler(config_path)
    base_config = base_handler.load_config()
    num_runs = base_config.get("num_runs", 20)

    # We also precompute T and dt for any run_config
    start_time, end_time = base_config["time_range"]
    granularity = base_config["granularity"]
    T = (end_time - start_time) * granularity
    dt = 1.0 / granularity

    # 3) Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(
        "outputs",
        base_config.get("folder", "default_folder"),
        f"{base_config.get('name_xp', 'custom_bar_chart')}_{timestamp}"
    )
    os.makedirs(base_output_dir, exist_ok=True)

    # 4) Define the 3 experiments we want.
    #   1) "unidirectional_coordinated" w/ alpha=1.0
    #   2) "coordinated" (alpha=1.0)
    #   3) "coordinated" (alpha=0.25)
    # We store each as (label, experiment_class, alpha_factor).
    xp_definitions = [
        ("Unidirectional", UnidirectionalCoordinatedSchedulingExperiment, 1.0),
        ("Bidirectional", CoordinatedSchedulingExperiment, 1.0),
        ("Bidirectional - high flexibility", CoordinatedSchedulingExperiment, 0.25),
    ]

    # We want battery_factor in [1.0, 0.01].
    battery_factors = [1.0, 0.25]

    # 5) Data structure to accumulate results:
    #    costs_data[(xp_label, battery_factor)] = list of total operator costs across runs
    costs_data = {}

    # 6) Loop over each xp type & battery_factor combination
    for (xp_label, xp_class, alpha_override) in xp_definitions:
        for bf in battery_factors:
            combo_key = (xp_label, bf)
            costs_data[combo_key] = []

            run_config_handler = ConfigHandler(config_path)
            # Force the alpha_factor and battery_factor for this run
            run_config = run_config_handler.load_config(
                alpha_factor_override=alpha_override,
                battery_factor_override=bf
            )
            run_config["T"] = T
            run_config["dt"] = dt
            sampled_data_per_run = run_config_handler.get_sampled_data_per_run()

            # For each combination, we do num_runs
            for run_idx in range(num_runs):
                sampled_data = sampled_data_per_run[run_idx % len(sampled_data_per_run)]

                # Update run_config with the sample
                run_config["market_prices"]    = sampled_data["sampled_day_data"]["prices"]
                run_config["sampled_day_date"] = sampled_data["sampled_day_data"]["date"]
                run_config["evs"]             = sampled_data["evs"]

                # 7) Run the experiment
                results = run_single_experiment(xp_class, run_config)

                # 8) Extract total operator cost
                total_cost = results.get("sum_operator_costs", np.nan)
                costs_data[combo_key].append(total_cost)

    # 9) Now we build a bar chart from `costs_data`.
    # Plot function will expect something like:
    #   { 
    #       xp_label: {
    #           bf_value: [list of runs' costs]
    #       },
    #       ...
    #   }
    chart_data = {}
    for (xp_label, bf) in costs_data:
        if xp_label not in chart_data:
            chart_data[xp_label] = {}
        chart_data[xp_label][bf] = costs_data[(xp_label, bf)]

    # 10) Call a new plot method on PlotHandler
    plot_output_path = os.path.join(base_output_dir, "bar_chart_summary.png")
    PlotHandler.plot_cost_savings_bars(chart_data, plot_output_path)
    logging.info("Bar chart analysis completed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_bar_chart.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])
