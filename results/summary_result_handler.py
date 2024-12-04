import os
import json
from results.plot_handler import PlotHandler


class SummaryResultHandler:
    def __init__(self, base_output_dir, experiment_types):
        self.base_output_dir = base_output_dir
        self.experiment_types = experiment_types
        self.results_by_experiment = {}

    def aggregate_results(self):
        """
        Aggregate results for each experiment type across runs.
        """
        for xp_type in self.experiment_types:
            xp_path = os.path.join(self.base_output_dir, xp_type)
            if os.path.isdir(xp_path):
                self.results_by_experiment[xp_type] = []
    
                for run_dir in os.listdir(xp_path):
                    run_path = os.path.join(xp_path, run_dir)
                    if os.path.isdir(run_path):
                        log_path = os.path.join(run_path, "log.json")
                        if os.path.isfile(log_path):
                            with open(log_path, "r") as log_file:
                                log_data = json.load(log_file)
                                self.results_by_experiment[xp_type].append(log_data["results"])

    def generate_summary_plots(self, plots):
        """
        Generate summary plots as specified in the configuration.
        """
        if "objective_bars" in plots:
            output_path = os.path.join(self.base_output_dir, "objective_bars.png")
            PlotHandler.plot_summary_bars(self.results_by_experiment, output_path)
            print(f"Objective bars plot saved to {output_path}")
