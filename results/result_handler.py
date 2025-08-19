import os
import json
import logging
from typing import Dict, Any, List, Union

from results.plot_handler import PlotHandler


class ResultHandler:
    """
    Handles the saving of logs (in JSON) and the generation of plots for experiment results.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        results: Dict[str, Any],
        base_output_dir: str,
        experiment_type: str,
        run_id: Union[int, str],
        ):
        """
        Initializes the ResultHandler.

        Args:
            config (dict): Experiment configuration dictionary.
            results (dict): Dictionary containing the results of the experiment.
            base_output_dir (str): Base directory where results will be saved.
            experiment_type (str): Identifier for the current experiment type.
            run_id (int or str): Identifier for the run (useful in stochastic runs).
        """
        self.config = config
        self.results = results
        self.base_output_dir = base_output_dir
        self.experiment_type = experiment_type
        self.run_id = run_id
        self.output_dir = self._get_output_dir()

    def _get_output_dir(self) -> str:
        """
        Builds and returns the output directory path for this experiment and run,
        ensuring that the directory is created.

        Returns:
            str: The path to the output directory.
        """
        experiment_dir = os.path.join(self.base_output_dir, self.experiment_type)
        run_dir = os.path.join(experiment_dir, f"run_{self.run_id}")

        # Ensure the directories exist
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def save_log(self) -> None:
        """
        Saves experiment configuration and results to a JSON log file.
        """
        log_data = {
            "config": self.config,
            "results": self.results,  # dump everything in results
        }

        log_path = os.path.join(self.output_dir, "log.json")
        with open(log_path, "w", encoding="utf-8") as file:
            json.dump(log_data, file, indent=4)

        logging.info(f"Log saved to {log_path}")

    def save_plots(self) -> None:
        """
        Saves plots by calling the corresponding PlotHandler methods only if those
        plot names are requested in config['plots'].
        """
        requested_plots: List[str] = self.config.get("plots", [])

        # Set up the time axis
        start_time, end_time = self.config["time_range"]
        nb_time_steps = self.config["nb_time_steps"]
        time_axis = [start_time + step*self.config["dt"] for step in range(nb_time_steps + 1)]

        # Mapping from plot_name to the corresponding PlotHandler method
        plot_methods = {
            "soc_evolution": PlotHandler.plot_soc_evolution,
        }

        # Loop over all plot_methods, only generate plots if they are requested
        for plot_name, plot_method in plot_methods.items():
            if plot_name not in requested_plots:
                logging.debug(f"Plot '{plot_name}' not requested. Skipping.")
                continue

            output_path = os.path.join(self.output_dir, f"{plot_name}.pdf")
                    
            if plot_name == "soc_evolution":
                # This plot function requires `results`, `time_axis`, `config`, and `output_path`.
                plot_method(self.results, time_axis, output_path, self.config)

            else:
                # For all other plot types in plot_methods,
                # assume it requires (results, time_axis, output_path).
                plot_method(self.results, time_axis, output_path)

            logging.info(f"Plot '{plot_name}' saved to {output_path}")

    def save(self) -> None:
        """
        The main entry point for saving logs and plots.
        """
        self.save_log()
        self.save_plots()
        logging.info(f"All results saved in {self.output_dir}")
