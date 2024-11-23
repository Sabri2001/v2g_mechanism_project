from config.config_handler import ConfigHandler
from experiments.uncoordinated_charging_experiment import UncoordinatedChargingExperiment
from experiments.centralized_scheduling_experiment import CentralizedSchedulingExperiment
from results.result_handler import ResultHandler


def main(config_path):
    # Load configuration
    config_handler = ConfigHandler(config_path)
    config = config_handler.load_config()
    print("Config loaded and validated.")

    # Map experiment types to classes
    experiment_classes = {
        "uncoordinated": UncoordinatedChargingExperiment,
        "centralized_scheduling": CentralizedSchedulingExperiment
    }

    # Determine experiment type from the config
    experiment_type = config.get("experiment_type")
    if experiment_type not in experiment_classes:
        raise ValueError(f"Invalid experiment type: {experiment_type}. Valid types are: {list(experiment_classes.keys())}")
    else:
        print(f"Experiment type: {experiment_type}")

    # Run the selected experiment
    experiment_class = experiment_classes[experiment_type]
    experiment = experiment_class(config)
    results = experiment.run()

    # Handle and save results
    result_handler = ResultHandler(config, results)
    result_handler.save()


if __name__ == "__main__":
    config_file = "config/config.yaml"
    main(config_file)
 