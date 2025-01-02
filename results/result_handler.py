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

        Instead of hardcoding specific keys, we dump everything in self.results
        under "results". If needed, you can keep filtering certain keys.
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

        Some plots are managed by SummaryResultHandler (e.g., total_cost_bars),
        so we skip them here to avoid duplication.
        """
        requested_plots: List[str] = self.config.get("plots", [])

        # Set up the time axis
        start_time, end_time = self.config["time_range"]
        T = end_time - start_time  # total time steps
        time_axis = [start_time + t for t in range(T + 1)]

        # Mapping from plot_name to the corresponding PlotHandler method
        # Only plots that are actually handled here
        plot_methods = {
            "total_cost_vs_energy_cost": PlotHandler.plot_total_cost_vs_energy_cost,
            "soc_evolution": PlotHandler.plot_soc_evolution,
            "market_prices": PlotHandler.plot_market_prices,
            "evcs_power": PlotHandler.plot_evcs_power,
        }

        # Loop over all plot_methods, only generate plots if they are requested
        for plot_name, plot_method in plot_methods.items():
            if plot_name not in requested_plots:
                logging.debug(f"Plot '{plot_name}' not requested. Skipping.")
                continue

            output_path = os.path.join(self.output_dir, f"{plot_name}.png")

            # Special-case handling
            if plot_name == "market_prices":
                market_prices = self.config.get("market_prices")
                if market_prices:
                    plot_method(market_prices, [start_time, end_time], output_path)
                else:
                    logging.warning(
                        "No 'market_prices' found in config. Skipping market_prices plot."
                    )

            elif plot_name == "soc_evolution":
                # This plot function requires `results`, `time_axis`, `config`, and `output_path`.
                plot_method(self.results, time_axis, output_path, self.config)

            elif plot_name == "evcs_power":
                # Check if 'evcs_power' data is available in results
                evcs_power_data = self.results.get("evcs_power")
                if evcs_power_data is not None:
                    # time_axis[:-1] might be used if evcs_power has length T
                    plot_method(evcs_power_data, time_axis[:-1], output_path)
                else:
                    logging.warning(
                        "No 'evcs_power' found in results. Skipping evcs_power plot."
                    )

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
