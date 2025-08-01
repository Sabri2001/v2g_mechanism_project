import os
import json
import logging
from typing import List, Dict, Any
import numpy as np

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
        if "vcg_tax_bar" in plots:
            output_path = os.path.join(self.base_output_dir, "vcg_tax_bar.png")
            PlotHandler.plot_vcg_tax_bar(self.results_by_experiment, output_path)
            logging.info(f"VCG tax bar plot saved to {output_path}")

        if "vcg_vs_disconnection_time" in plots:
            output_path = os.path.join(self.base_output_dir, "vcg_vs_disconnection_time.png")
            PlotHandler.plot_vcg_vs_disconnection_time(self.results_by_experiment, output_path)
            logging.info(f"VCG vs disconnection time plot saved to {output_path}")

        if "vcg_vs_flexibility" in plots:
            output_path = os.path.join(self.base_output_dir, "vcg_vs_flexibility.png")
            PlotHandler.plot_vcg_vs_flexibility(self.results_by_experiment, output_path)
            logging.info(f"VCG vs flexibility plot saved to {output_path}")

        if "ev_total_cost_comparison" in plots:
            output_path = os.path.join(self.base_output_dir, "ev_total_cost_comparison.png")
            PlotHandler.plot_ev_total_cost_comparison(self.results_by_experiment, output_path)
            logging.info(f"EV total cost comparison plot saved to {output_path}")

        if "admm_iterations_violin" in plots:
            output_path = os.path.join(self.base_output_dir, "admm_iterations_violin.png")
            PlotHandler.plot_admm_iterations_violin(self.results_by_experiment, output_path)
            logging.info(f"ADMM iterations violin plot saved to {output_path}")

        if "vcg_tax_distribution" in plots:
            output_path = os.path.join(self.base_output_dir, "vcg_tax_distribution.png")
            PlotHandler.plot_vcg_tax_distribution(self.results_by_experiment, output_path)
            logging.info(f"VCG tax distribution plot saved to {output_path}")

        if "vcg_tax_violin" in plots:
            output_path = os.path.join(self.base_output_dir, "vcg_tax_violin.png")
            PlotHandler.plot_vcg_tax_violin(self.results_by_experiment, output_path)
            logging.info(f"VCG tax violin plot saved to {output_path}")
        
        if "gap_violin" in plots:
            output_path = os.path.join(self.base_output_dir, "gap_violin.png")
            PlotHandler.plot_gap_violin(self.results_by_experiment, output_path)
            logging.info(f"Gap violin saved to {output_path}") 

        if "gap_distribution" in plots:
            output_path = os.path.join(self.base_output_dir, "gap_distribution.png")
            PlotHandler.plot_gap_distribution(self.results_by_experiment, output_path)
            logging.info(f"Gap distribution saved to {output_path}") 

        if "gap_swarm" in plots:
            output_path = os.path.join(self.base_output_dir, "gap_swarm.png")
            PlotHandler.plot_gap_swarm(self.results_by_experiment, output_path)
            logging.info(f"Gap swarm saved to {output_path}") 
        
        if "total_cost_bars" in plots:
            output_path = os.path.join(self.base_output_dir, "total_cost_bars.png")
            PlotHandler.plot_total_cost(self.results_by_experiment, output_path)
            logging.info(f"Total cost bars plot saved to {output_path}")

        if "total_cost_and_energy_bars" in plots:
            output_path = os.path.join(self.base_output_dir, "total_cost_bars.png")
            PlotHandler.plot_total_cost_and_energy(self.results_by_experiment, output_path)
            logging.info(f"Total cost bars plot saved to {output_path}")

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

        if "v2g_fraction" in plots:
            output_path = os.path.join(self.base_output_dir, "v2g_fraction.png")
            PlotHandler.plot_v2g_fraction_bars(self.results_by_experiment, output_path)
            logging.info(f"Energy transferred fraction plot saved to {output_path}")

    def save_computation_time_stats(self, filename: str = "computation_time_stats.json"):
        """
        Saves robust summary statistics (median, quartiles, min, max)
        of computation times for each experiment type into a JSON file.
        """
        import json
        import logging
        import numpy as np
        import os

        comp_time_stats = {}

        for xp_type in self.experiment_types:
            # Skip if we have no results for that experiment type
            if xp_type not in self.results_by_experiment:
                continue

            times = []
            for run_result in self.results_by_experiment[xp_type]:
                if "computation_time" in run_result:
                    times.append(run_result["computation_time"])

            if times:
                arr = np.array(times)
                median_time = float(np.median(arr))
                q1_time = float(np.percentile(arr, 25))
                q3_time = float(np.percentile(arr, 75))
                min_time = float(np.min(arr))
                max_time = float(np.max(arr))

                comp_time_stats[xp_type] = {
                    "median_computation_time_s": median_time,
                    "q1_computation_time_s": q1_time,
                    "q3_computation_time_s": q3_time,
                    "min_computation_time_s": min_time,
                    "max_computation_time_s": max_time
                }

                logging.info(
                    f"[{xp_type}] Computation Time -> "
                    f"median={median_time:.4f}s, q1={q1_time:.4f}s, q3={q3_time:.4f}s, "
                    f"min={min_time:.4f}s, max={max_time:.4f}s"
                )
            else:
                comp_time_stats[xp_type] = {
                    "median_computation_time_s": None,
                    "q1_computation_time_s": None,
                    "q3_computation_time_s": None,
                    "min_computation_time_s": None,
                    "max_computation_time_s": None
                }

        # Save to JSON in the base output directory
        output_path = os.path.join(self.base_output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(comp_time_stats, f, indent=2)

        logging.info(f"Computation time stats written to {output_path}.")
