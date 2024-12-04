import os
import json
from results.plot_handler import PlotHandler


class ResultHandler:
    def __init__(self, config, results, timestamp, experiment_type, run_id):
        self.config = config
        self.results = results
        self.timestamp = timestamp
        self.experiment_type = experiment_type
        self.run_id = run_id
        self.output_dir = self.get_output_dir()

    def get_output_dir(self):
        # Extract the experiment name and timestamp from the configuration
        xp_name = self.config.get("name_xp", "default_xp")
        
        # Construct the base directory: outputs/{xp name}_{timestamp}
        base_dir = os.path.join("outputs", f"{xp_name}_{self.timestamp}")
        
        # Add experiment type and run ID to the directory structure
        experiment_dir = os.path.join(base_dir, self.experiment_type)
        run_dir = os.path.join(experiment_dir, f"run_{self.run_id}")
        
        # Ensure the directories exist
        os.makedirs(run_dir, exist_ok=True)
        
        return run_dir

    def save_log(self):
        log_data = {
            "config": self.config,
            "results": {
                "operator_objective_vector": self.results["operator_objective_vector"],
                "energy_cost_vector": self.results["energy_cost_vector"],
                "sum_operator_objective": self.results["sum_operator_objective"],
                "sum_energy_costs": self.results["sum_energy_costs"],
                "soc_over_time": self.results["soc_over_time"],
                "desired_disconnect_time": self.results["desired_disconnect_time"],
                "actual_disconnect_time": self.results["actual_disconnect_time"],
            },
        }
        log_path = os.path.join(self.output_dir, "log.json")
        with open(log_path, "w") as file:
            json.dump(log_data, file, indent=4)

    def save_plots(self):
        plots = self.config.get("plots", [])
        start_time, end_time = self.config["time_range"]
        T = end_time - start_time  # Total number of time slots
        time_axis = [start_time + t for t in range(T + 1)]

        # Define the plot methods
        plot_methods = {
            "objective_vs_cost": PlotHandler.plot_objective_vs_cost,
            "soc_evolution": PlotHandler.plot_soc_evolution,
            "market_prices": PlotHandler.plot_market_prices,
        }

        for plot_name in plots:
            plot_method = plot_methods.get(plot_name)
            if plot_method:
                output_path = os.path.join(self.output_dir, f"{plot_name}.png")
                
                # Special handling for market_prices
                if plot_name == "market_prices":
                    if "market_prices" in self.config:
                        market_prices = self.config["market_prices"]
                        plot_method(market_prices, [start_time, end_time], output_path)
                # Special handling for soc_evolution
                elif plot_name == "soc_evolution":
                    plot_method(self.results, time_axis, output_path, self.config)
                else:
                    plot_method(self.results, time_axis, output_path)

    def save(self):
        self.save_log()
        self.save_plots()
        print(f"Results saved in {self.output_dir}")
