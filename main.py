import os
import sys
import logging
from datetime import datetime

# Experiments
from config.config_handler import ConfigHandler
from experiments.uncoordinated_charging_experiment import UncoordinatedChargingExperiment
from experiments.coordinated_scheduling_experiment import CoordinatedSchedulingExperiment
from experiments.unidirectional_coordinated_scheduling_experiment import UnidirectionalCoordinatedSchedulingExperiment
from experiments.centralized_scheduling_experiment import CentralizedSchedulingExperiment

# Results
from results.result_handler import ResultHandler
from results.summary_result_handler import SummaryResultHandler


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

def main(config_path):
    """
    Main entry point for running EV charging experiments.

    Args:
        config_path (str): Path to the YAML configuration file.
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

    logging.info("All experiments completed.")

    # 5) Generate Summary Plots (Across All Runs)
    summary_handler = SummaryResultHandler(base_output_dir, experiment_types)
    summary_handler.aggregate_results()
    summary_handler.generate_summary_plots(config.get("plots", []))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_path>")
        sys.exit(1)

    config_file = sys.argv[1]
    main(config_file)    
