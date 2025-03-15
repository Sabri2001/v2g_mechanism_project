import os
import sys
import logging
import copy
import numpy as np
from datetime import datetime

# Import necessary modules
from config.config_handler import ConfigHandler
from experiments.coordinated_scheduling_experiment import CoordinatedSchedulingExperiment
from experiments.unidirectional_coordinated_scheduling_experiment import UnidirectionalCoordinatedSchedulingExperiment
from results.plot_handler import PlotHandler


def run_single_experiment(experiment_class, config):
    """Runs an experiment and returns its results dictionary."""
    experiment = experiment_class(config)
    return experiment.run()

def main(config_path):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Load base configuration once to get analysis parameters
    config_handler = ConfigHandler(config_path)
    config = config_handler.load_config()

    start_time, end_time = config["time_range"]
    granularity = config["granularity"]
    T = (end_time - start_time) * granularity
    dt = 1.0 / granularity
    config["T"] = T
    config["dt"] = dt
    
    # Read the ranges and number of runs per combination from the config.
    battery_factor_range = config.get("battery_factor_range", [1.0])
    alpha_factor_range = config.get("alpha_factor_range", [1.0])
    num_runs = config.get("num_runs", 20)
    
    # Create an output directory for the analysis results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(
        "outputs",
        config.get("folder", "analysis"),
        f"{config.get('name_xp', 'cost_savings_analysis')}_{timestamp}"
    )
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Data structure to hold cost savings for each combination
    # Structure: { battery_factor: { alpha_factor: [list of cost savings (%)] } }
    results_data = {}
    
    for battery_factor in battery_factor_range:
        results_data[battery_factor] = {}
        for alpha_factor in alpha_factor_range:
            # Prep config for run
            run_config_handler = ConfigHandler(config_path)
            run_config = run_config_handler.load_config(
                alpha_factor_override=alpha_factor,
                battery_factor_override=battery_factor
            )
            run_data = run_config_handler.get_sampled_data_per_run()
            run_config["T"] = T
            run_config["dt"] = dt
            # Prep baseline config
            baseline_config_handler = ConfigHandler(config_path)
            baseline_config = baseline_config_handler.load_config(
                alpha_factor_override=1000.0,
                battery_factor_override=battery_factor
            )
            baseline_data = baseline_config_handler.get_sampled_data_per_run()
            baseline_config["T"] = T
            baseline_config["dt"] = dt

            savings_list = []
            for run in range(num_runs):                
                # Use one sampled run from the available list (cycle if needed)
                run_data_current_run = run_data[run % len(run_data)]
                baseline_data_current_run = baseline_data[run % len(baseline_data)]
                
                # Update config with sampled day data and EV parameters
                run_config["market_prices"] = run_data_current_run["sampled_day_data"]["prices"]
                run_config["sampled_day_date"] = run_data_current_run["sampled_day_data"]["date"]
                run_config["evs"] = run_data_current_run["evs"]
                baseline_config["market_prices"] = baseline_data_current_run["sampled_day_data"]["prices"]
                baseline_config["sampled_day_date"] = baseline_data_current_run["sampled_day_data"]["date"]
                baseline_config["evs"] = baseline_data_current_run["evs"]
                
                # Run coordinated experiment
                coordinated_results = run_single_experiment(CoordinatedSchedulingExperiment, run_config)
                baseline_results = run_single_experiment(CoordinatedSchedulingExperiment, baseline_config)

                # Extract the cost metric
                coord_cost = coordinated_results.get("sum_operator_costs", np.nan)
                baseline_cost = baseline_results.get("sum_operator_costs", np.nan)
                
                # Compute cost savings (%)
                if baseline_cost != 0:
                    cost_savings = ((baseline_cost-coord_cost) / baseline_cost) * 100
                    print(f"Cost savings: {cost_savings:.2f}%")
                else:
                    cost_savings = 0.0
                savings_list.append(cost_savings)
            
            # Compute mean and standard deviation for this combination
            mean_savings = np.mean(savings_list)
            std_savings = np.std(savings_list)
            results_data[battery_factor][alpha_factor] = {"mean": mean_savings, "std": std_savings}
            logging.info(
                f"Battery Factor: {battery_factor}, Time Flexibility Factor: {alpha_factor}, "
                f"Mean Savings: {mean_savings:.2f}%, Std: {std_savings:.2f}%"
            )
    
    # Prepare data for plotting: for each battery_factor, sort the alpha_factor values
    plot_data = {}
    for bf, alpha_dict in results_data.items():
        sorted_alphas = sorted(alpha_dict.keys())
        means = [alpha_dict[a]["mean"] for a in sorted_alphas]
        stds = [alpha_dict[a]["std"] for a in sorted_alphas]
        plot_data[bf] = {"alpha": sorted_alphas, "mean": means, "std": stds}
    
    # Use PlotHandler to create the line plot with error (std) bars.
    plot_output_path = os.path.join(base_output_dir, "cost_savings_vs_alpha.png")
    PlotHandler.plot_cost_savings_vs_alpha(plot_data, plot_output_path)
    logging.info("Cost savings analysis completed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_cost_savings_analysis.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])
