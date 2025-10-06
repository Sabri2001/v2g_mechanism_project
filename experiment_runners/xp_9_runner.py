import os
import sys
import logging
import argparse
import copy
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_handler import ConfigHandler
# from experiments.uncoordinated_charging_experiment import UncoordinatedChargingExperiment
from experiments.centralized_scheduling_experiment import CentralizedSchedulingExperiment
from results.result_handler import ResultHandler


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

def run_nash_experiments(config, base_output_dir):
    """
    For each combination of candidate bids for the EV indicated by 'ev_nash' (from tau_nash and alpha_nash),
    modify that EV's bid in the configuration, run the ADMM (coordinated) experiment, and save results.
    """
    tau_values = config.get("tau_nash", [])
    alpha_values = config.get("alpha_nash", [])
    
    if not tau_values or not alpha_values:
        logging.error("tau_nash or alpha_nash lists are empty in config!")
        return

    # The EV id on which to run the Nash check.
    ev_nash_id = config.get("ev_nash", 0)
    logging.info(f"Running Nash check for EV id {ev_nash_id}")

    # Get true private type values for the target EV
    true_tau_global = None
    true_alpha_global = None
    for ev in config["evs"]:
        if ev["id"] == ev_nash_id:
            true_tau_global = ev.get("true_disconnection_time", ev["disconnection_time"])
            true_alpha_global = ev.get("true_disconnection_time_flexibility",
                                       ev["disconnection_time_flexibility"])
            break

    # Create directory for Nash experiment results
    nash_output_dir = os.path.join(base_output_dir, "nash_experiments")
    os.makedirs(nash_output_dir, exist_ok=True)

    # Store results for plotting later
    nash_results = {
        "tau_values": tau_values,
        "alpha_values": alpha_values,
        "ev_nash_id": ev_nash_id,
        "true_tau": true_tau_global,
        "true_alpha": true_alpha_global,
        "results": []
    }

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
            current_true_tau = true_tau_global if true_tau_global is not None else tau_bid
            current_true_alpha = true_alpha_global if true_alpha_global is not None else alpha_bid

            # Compute adaptability cost using the true values.
            adaptability_cost = 0.5 * current_true_alpha * ((current_true_tau - t_actual) ** 2)
            energy_cost = results.get("energy_cost", {}).get(ev_nash_id, 0)
            total_cost = energy_cost + adaptability_cost

            # Store result
            nash_results["results"].append({
                "tau_bid": tau_bid,
                "alpha_bid": alpha_bid,
                "t_actual": t_actual,
                "energy_cost": energy_cost,
                "adaptability_cost": adaptability_cost,
                "total_cost": total_cost
            })

            logging.info(f"Nash check: tau {tau_bid}, alpha {alpha_bid}, cost {total_cost}")

    # Save nash results to file for plotting script
    import json
    nash_results_file = os.path.join(base_output_dir, "nash_results.json")
    with open(nash_results_file, 'w') as f:
        json.dump(nash_results, f, indent=2)
    
    logging.info(f"Nash check results saved to {nash_results_file}")

def main():
    """
    Main entry point for running EV charging experiments.
    """
    parser = argparse.ArgumentParser(description="Run EV Charging Experiment 9")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--timestamp", required=True, help="Timestamp for output directory")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1) Load Configuration
    config_handler = ConfigHandler(args.config)
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
    base_output_dir = os.path.join(
        "outputs",
        config.get("folder", "default_folder"),
        f"{config.get('name_xp', 'default_xp')}_{args.timestamp}"
    )
    os.makedirs(base_output_dir, exist_ok=True)

    # 3) Map Experiment Types to Classes
    experiment_classes = {
        # "uncoordinated": UncoordinatedChargingExperiment,
        "centralized": CentralizedSchedulingExperiment,
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

    logging.info("All standard experiments completed.")

    # 5) If nash_check is enabled, run the Nash check experiments.
    if config.get("nash_check", False):
        logging.info("Starting Nash check grid experiment.")
        run_nash_experiments(config, base_output_dir)

    logging.info("Experiment 9 runner completed successfully.")


if __name__ == "__main__":
    main()
