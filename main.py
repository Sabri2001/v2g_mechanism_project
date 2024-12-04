# main.py
from datetime import datetime
import os

from config.config_handler import ConfigHandler
from experiments.uncoordinated_charging_experiment import UncoordinatedChargingExperiment
from experiments.centralized_scheduling_experiment import CentralizedSchedulingExperiment
from experiments.inelastic_centralized_scheduling_experiment import InelasticCentralizedSchedulingExperiment
from results.result_handler import ResultHandler
from results.summary_result_handler import SummaryResultHandler  # Import the new class


def run_experiment(experiment_class, config, timestamp, experiment_type, run_id, sampled_day_data):
    """
    Helper function to initialize and run a single experiment.
    """
    print(f"Running {experiment_type}, Run ID: {run_id}")

    # Add the sampled day's data to the config for logging
    config["sampled_day_date"] = sampled_day_data["date"]

    experiment = experiment_class(config)
    results = experiment.run()

    # Pass the sampled day data to the result handler for logging
    result_handler = ResultHandler(config, results, timestamp, experiment_type, run_id)
    result_handler.save()

    print(f"Completed {experiment_type}, Run ID: {run_id}")

def main(config_path):
    # Load configuration
    config_handler = ConfigHandler(config_path)
    config = config_handler.load_config()
    sampled_data_per_run = config_handler.get_sampled_data_per_run()
    print("Config loaded and validated.")

    # Get timestamp for unique output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Map experiment types to classes
    experiment_classes = {
        "uncoordinated": UncoordinatedChargingExperiment,
        "centralized_scheduling": CentralizedSchedulingExperiment,
        "inelastic_centralized_scheduling": InelasticCentralizedSchedulingExperiment
    }

    # Get experiment types
    experiment_types = config.get("experiment_types", [])

    if not experiment_types:
        raise ValueError("No experiment types specified in the configuration.")

    # Base output directory
    base_output_dir = os.path.join("outputs", f"{config.get('name_xp', 'default_xp')}_{timestamp}")

    # Loop over runs
    for run_data in sampled_data_per_run:
        run_id = run_data["run_id"]
        sampled_day_data = run_data["sampled_day_data"]
        evs = run_data["evs"]
        # Update the config with the sampled data for this run
        config["market_prices"] = sampled_day_data["prices"]
        config["evs"] = evs

        for experiment_type in experiment_types:
            if experiment_type not in experiment_classes:
                print(f"Skipping invalid experiment type: {experiment_type}")
                continue

            experiment_class = experiment_classes[experiment_type]
            run_experiment(experiment_class, config, timestamp, experiment_type, run_id, sampled_day_data)

    print("All experiments completed.")

    # Generate summary plots using SummaryResultHandler
    if "objective_bars" in config.get("plots", []):
        summary_handler = SummaryResultHandler(base_output_dir, experiment_types)
        summary_handler.aggregate_results()
        summary_handler.generate_summary_plots(config.get("plots", []))


if __name__ == "__main__":
    config_file = "config/config.yaml"
    main(config_file)
