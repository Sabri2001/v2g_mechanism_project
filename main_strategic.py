import os
import sys
import logging
import copy
import numpy as np
from datetime import datetime

# Experiments
from config.config_handler import ConfigHandler
from experiments.uncoordinated_charging_experiment import UncoordinatedChargingExperiment
from experiments.coordinated_scheduling_experiment import CoordinatedSchedulingExperiment
from experiments.unidirectional_coordinated_scheduling_experiment import UnidirectionalCoordinatedSchedulingExperiment

# Results and plotting
from results.result_handler import ResultHandler
from results.summary_result_handler import SummaryResultHandler
from results.plot_handler import PlotHandler


def run_experiment(experiment_class, config, output_dir, experiment_type, run_id, sampled_day_data):
    """
    Initializes and runs a single experiment, then saves the results.
    """
    logging.info(f"Running {experiment_type}, Run ID: {run_id}")

    # Instantiate and run the experiment
    experiment = experiment_class(config)
    results = experiment.run()

    # Save results (plots, logs) for this run
    result_handler = ResultHandler(config, results, output_dir, experiment_type, run_id)
    result_handler.save()

    logging.info(f"Completed {experiment_type}, Run ID: {run_id}")

def run_strategy_check(config, base_output_dir):
    """
    For each combination of candidate bids for the EV indicated by 'ev_nash' (from tau_nash and alpha_nash),
    modify that EV's bid in the configuration, run the ADMM (coordinated) experiment, and compute
    its actual utility (as energy_cost + adaptability_cost, where
    adaptability_cost is computed using the true parameters). Finally, produce a grid plot.
    """
    tau_values = config.get("tau_nash", [])
    alpha_values = config.get("alpha_nash", [])
    
    if not tau_values or not alpha_values:
        logging.error("tau_nash or alpha_nash lists are empty in config!")
        return

    # The EV id on which to run the Nash check.
    ev_nash_id = config.get("ev_nash", 0)
    logging.info(f"Running Nash check for EV id {ev_nash_id}")

    # Cost grid: rows correspond to alpha values, columns to tau values.
    cost_grid = np.zeros((len(alpha_values), len(tau_values)))

    # We assume that the true private type values for the target EV are provided
    # under keys "true_disconnection_time" and "true_disconnection_time_flexibility".
    true_tau_global = None
    true_alpha_global = None
    for ev in config["evs"]:
        if ev["id"] == ev_nash_id:
            true_tau_global = ev.get("true_disconnection_time", ev["disconnection_time"])
            true_alpha_global = ev.get("true_disconnection_time_flexibility",
                                       ev["disconnection_time_flexibility"])
            break

    # Loop over each candidate bid combination.
    for i, tau_bid in enumerate(tau_values):
        for j, alpha_bid in enumerate(alpha_values):
            # Create a deep copy of config to modify the target EV's bid.
            config_nash = copy.deepcopy(config)
            ev_found = False
            for ev in config_nash["evs"]:
                if ev["id"] == ev_nash_id:
                    ev["disconnection_time"] = tau_bid
                    ev["disconnection_time_flexibility"] = alpha_bid
                    ev_found = True
                    break
            if not ev_found:
                logging.error(f"EV with id {ev_nash_id} not found in config during Nash check.")
                continue

            # Run the coordinated ADMM experiment with the modified config.
            experiment = CoordinatedSchedulingExperiment(config_nash)
            results = experiment.run()

            # Find the index of the target EV in the results.
            target_index = next((idx for idx, ev in enumerate(config_nash["evs"]) if ev["id"] == ev_nash_id), None)
            if target_index is None:
                logging.error(f"EV with id {ev_nash_id} not found in results during Nash check.")
                continue

            # Extract the assignment outcome for the target EV.
            t_actual = results["actual_disconnection_time"][target_index]
            # Use the true private type values (true_tau_global, true_alpha_global) if available.
            current_true_tau = true_tau_global if true_tau_global is not None else tau_bid
            current_true_alpha = true_alpha_global if true_alpha_global is not None else alpha_bid

            # Compute adaptability cost using the true values.
            adaptability_cost = 0.5 * current_true_alpha * ((current_true_tau - t_actual) ** 2)
            # Get the energy_cost and congestion_cost for the target EV from the results.
            energy_cost = results.get("energy_cost", {}).get(ev_nash_id, 0)
            # Total cost is the sum of these two components.
            cost = energy_cost + adaptability_cost

            cost_grid[j, i] = cost # row: alpha index, col: tau index
            logging.info(f"Nash check: tau {tau_bid}, alpha {alpha_bid}, cost {cost}")

    # Plot the grid (heatmap) of costs for the target EV.
    nash_plot_path = os.path.join(base_output_dir, "nash_grid.png")
    # Pass the true bid values to the plotting routine.
    PlotHandler.plot_nash_grid(tau_values, alpha_values, cost_grid, nash_plot_path,
                                 true_tau=current_true_tau, true_alpha=current_true_alpha)
    logging.info(f"Nash check grid plot saved to {nash_plot_path}")

def main(config_path):
    """
    Main entry point for running EV charging experiments.
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1) Load Configuration
    config_handler = ConfigHandler(config_path)
    config = config_handler.load_config()
    sampled_data_per_run = config_handler.get_sampled_data_per_run()
    logging.info("Configuration loaded and validated.")

    start_time, end_time = config["time_range"]
    granularity = config["granularity"]
    T = (end_time - start_time) * granularity
    dt = 1.0 / granularity
    config["T"] = T
    config["dt"] = dt

    # 2) Create Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(
        "outputs",
        config.get("folder", "default_folder"),
        f"{config.get('name_xp', 'default_xp')}_{timestamp}"
    )
    os.makedirs(base_output_dir, exist_ok=True)

    # 3) Map Experiment Types to Classes
    experiment_classes = {
        "uncoordinated": UncoordinatedChargingExperiment,
        "coordinated": CoordinatedSchedulingExperiment,
        "unidirectional_coordinated": UnidirectionalCoordinatedSchedulingExperiment,
    }

    experiment_types = config.get("experiment_types", [])
    if not experiment_types:
        raise ValueError("No experiment types specified in the configuration.")

    # 4) Run Each Experiment for Each Sampled Run
    for run_data in sampled_data_per_run:
        run_id = run_data["run_id"]
        sampled_day_data = run_data["sampled_day_data"]
        evs = run_data["evs"]

        # Update config with sampled data
        config["market_prices"] = sampled_day_data["prices"]
        config["sampled_day_date"] = sampled_day_data["date"]
        config["evs"] = evs

        # Run each requested experiment
        for experiment_type in experiment_types:
            if experiment_type not in experiment_classes:
                logging.warning(f"Skipping invalid experiment type: {experiment_type}")
                continue

            experiment_class = experiment_classes[experiment_type]
            run_experiment(
                experiment_class=experiment_class,
                config=config,
                output_dir=base_output_dir,
                experiment_type=experiment_type,
                run_id=run_id,
                sampled_day_data=sampled_day_data
            )

    logging.info("All experiments completed.")

    # 5) If nash_check is enabled, run the Nash check branch.
    if config.get("nash_check", False):
        logging.info("Starting Nash check grid experiment.")
        run_strategy_check(config, base_output_dir)

    # 6) Generate Summary Plots (Across All Runs)
    summary_handler = SummaryResultHandler(base_output_dir, experiment_types)
    summary_handler.aggregate_results()
    summary_handler.generate_summary_plots(config.get("plots", []))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_path>")
        sys.exit(1)

    config_file = sys.argv[1]
    main(config_file)
