import os
import json
import logging
from typing import List, Dict, Any

from results.plot_handler import PlotHandler


class SummaryResultHandler:
    """
    A handler that aggregates multiple experiment runs from the output directory
    and generates summary plots combining their results.
    """

    def __init__(self, base_output_dir: str, experiment_types: List[str]) -> None:
        """
        Initializes the SummaryResultHandler.

        Args:
            base_output_dir (str): The base directory where all experiment outputs are stored.
            experiment_types (List[str]): A list of experiment type identifiers (e.g. ["uncoordinated", "coordinated"]).
        """
        self.base_output_dir = base_output_dir
        self.experiment_types = experiment_types
        self.results_by_experiment: Dict[str, List[Dict[str, Any]]] = {}

    def aggregate_results(self) -> None:
        """
        Aggregates (loads) results for each experiment type from JSON log files.
        Populates self.results_by_experiment[xp_type] with a list of result dicts
        loaded from each run's `log.json` file.
        """
        for xp_type in self.experiment_types:
            xp_path = os.path.join(self.base_output_dir, xp_type)
            if not os.path.isdir(xp_path):
                logging.warning(
                    f"Directory {xp_path} does not exist. "
                    f"Skipping experiment type '{xp_type}'."
                )
                continue

            # Initialize an empty list for this experiment type
            self.results_by_experiment[xp_type] = []

            # Iterate over each run directory
            for run_dir in os.listdir(xp_path):
                run_path = os.path.join(xp_path, run_dir)
                if not os.path.isdir(run_path):
                    continue

                log_path = os.path.join(run_path, "log.json")
                if not os.path.isfile(log_path):
                    logging.warning(f"No log.json found in {run_path}. Skipping.")
                    continue

                try:
                    with open(log_path, "r", encoding="utf-8") as log_file:
                        log_data = json.load(log_file)
                        # Safely extract "results" if present
                        results = log_data.get("results")
                        if results is None:
                            logging.warning(
                                f"'results' key not found in log.json at {log_path}. Skipping."
                            )
                            continue
                        self.results_by_experiment[xp_type].append(results)
                except (json.JSONDecodeError, OSError) as e:
                    logging.error(f"Error reading {log_path}: {e}")

    def generate_summary_plots(self, plots: List[str]) -> None:
        """
        Generates summary plots (e.g. total cost bars, cost benchmarking, V2G fraction)
        across all runs in self.results_by_experiment.

        Args:
            plots (List[str]): List of requested plot identifiers, e.g. ["total_cost_bars", "v2g_fraction"].
        """
        # total_cost_bars
        if "total_cost_bars" in plots:
            output_path = os.path.join(self.base_output_dir, "total_cost_bars.png")
            PlotHandler.plot_summary_bars(self.results_by_experiment, output_path)
            logging.info(f"Total cost bars plot saved to {output_path}")

        # total_cost_benchmarking
        if "total_cost_benchmarking" in plots:
            if "uncoordinated" not in self.experiment_types:
                logging.warning(
                    "Cannot generate total cost benchmarking plot because "
                    "'uncoordinated' is not present in experiment types."
                )
            else:
                output_path = os.path.join(self.base_output_dir, "total_cost_benchmarking.png")
                PlotHandler.plot_cost_benchmarking_bars(self.results_by_experiment, output_path)
                logging.info(f"Percentage change plot saved to {output_path}")

        # v2g_fraction
        if "v2g_fraction" in plots:
            output_path = os.path.join(self.base_output_dir, "v2g_fraction.png")
            PlotHandler.plot_v2g_fraction_bars(self.results_by_experiment, output_path)
            logging.info(f"Energy transferred fraction plot saved to {output_path}")
