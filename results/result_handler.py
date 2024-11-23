import os
import json
from datetime import datetime
from results.plot_handler import PlotHandler


class ResultHandler:
    def __init__(self, config, results):
        self.config = config
        self.results = results
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.get_output_dir()

    def get_output_dir(self):
        experiment_type = self.config["experiment_type"]
        output_dir = os.path.join("outputs", f"{experiment_type}_{self.timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def save_log(self):
        log_data = {
            "config": self.config,
            "results": {
                "operator_objective_vector": self.results["operator_objective_vector"],
                "energy_cost_vector": self.results["energy_cost_vector"],  # Include energy cost vector
                "sum_operator_objective": self.results["sum_operator_objective"],
                "sum_energy_costs": self.results["sum_energy_costs"],  # Include total energy costs
                "soc_over_time": self.results["soc_over_time"],  # Include SoC data for logging
            },
        }
        log_path = os.path.join(self.output_dir, "log.json")
        with open(log_path, "w") as file:
            json.dump(log_data, file, indent=4)

    def save_plots(self):
        plots = self.config.get("plots", [])
        start_time, end_time = self.config["time_range"]
        T = end_time - start_time  # Total number of time slots
        time_axis = [start_time + t for t in range(T)]  # Actual time points for plotting

        plot_methods = {
            "objective_vs_cost": PlotHandler.plot_objective_vs_cost,
            "soc_evolution": PlotHandler.plot_soc_evolution,
        }

        for plot_name in plots:
            plot_method = plot_methods.get(plot_name)
            if plot_method:
                output_path = os.path.join(self.output_dir, f"{plot_name}.png")
                if plot_name == "soc_evolution":
                    # Pass SoC data and time axis for plotting evolution
                    plot_method(self.results["soc_over_time"], time_axis, output_path, self.config)
                else:
                    # Pass results and time axis for plotting
                    plot_method(self.results, time_axis, output_path)

    def save(self):
        self.save_log()
        self.save_plots()
        print(f"Results saved in {self.output_dir}")
