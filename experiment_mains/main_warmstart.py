import logging
import time
import numpy as np
import os

from config.config_handler import ConfigHandler
from experiments.coordinated_scheduling_experiment import CoordinatedSchedulingExperiment
from experiments.centralized_scheduling_experiment import CentralizedSchedulingExperiment
from results.plot_handler import PlotHandler

def main_warmstart_experiment(config_path):
    """
    Runs multiple days/runs, measuring partial scheduling times for
    ADMM (coordinated) and Gurobi (centralized) with/without warmstart,
    then plots the results on a 2-category bar chart (ADMM vs Gurobi).
    """
    logging.basicConfig(level=logging.INFO)
    
    # Load config
    config_handler = ConfigHandler(config_path)
    config = config_handler.load_config()
    sampled_data_per_run = config_handler.get_sampled_data_per_run()

    # Set up time discretization
    start_time, end_time = config["time_range"]
    granularity = config["granularity"]
    T = (end_time - start_time) * granularity
    dt = 1.0 / granularity
    config["T"] = T
    config["dt"] = dt

    # Read where to save the plot
    plot_dir = config.get("folder", "warmstart_outputs")
    plot_filename = config.get("name_xp", "warmstart_partial_bar.png")
    os.makedirs(plot_dir, exist_ok=True)

    # Times for partial runs only
    times = {
        "coord_no_ws": [],
        "coord_ws": [],
        "cent_no_ws": [],
        "cent_ws": []
    }

    for run_idx, run_data in enumerate(sampled_data_per_run):
        logging.info(f"=== Run {run_data['run_id']} ({run_idx+1}/{len(sampled_data_per_run)}) ===")

        config["market_prices"] = run_data["sampled_day_data"]["prices"]
        config["sampled_day_date"] = run_data["sampled_day_data"]["date"]
        config["evs"] = run_data["evs"]
        evs = config["evs"]

        # Full runs (with EV 0) to get the initial schedules
        # (We skip details here, but you'd measure time or store partial solutions.)
        coord_full_exp = CoordinatedSchedulingExperiment(config)
        coord_full_res = coord_full_exp.run()

        cent_full_exp = CentralizedSchedulingExperiment(config)
        cent_full_res = cent_full_exp.run()

        # Build partial config (exclude EV 0)
        partial_config = config.copy()
        partial_config["evs"] = [ev for ev in evs if ev["id"] != 0]

        # Warmstart data if present
        admm_warmstart_dict = {}
        if "u" in coord_full_res:  # shape [num_evs, T]
            for i, ev in enumerate(evs):
                if ev["id"] != 0:
                    admm_warmstart_dict[ev["id"]] = coord_full_res["u"][i]

        cent_warmstart_dict = {}
        if "u" in cent_full_res:  # shape: {ev_id -> list}
            for ev in evs:
                if ev["id"] != 0:
                    cent_warmstart_dict[ev["id"]] = cent_full_res["u"][ev["id"]]

        # ADMM Partial (No Warmstart)
        admm_no_ws_exp = CoordinatedSchedulingExperiment(partial_config)
        start_t = time.time()
        _ = admm_no_ws_exp.run()
        times["coord_no_ws"].append(time.time() - start_t)

        # ADMM Partial (Warmstart)
        admm_ws_exp = CoordinatedSchedulingExperiment(partial_config)
        admm_ws_exp.apply_warmstart(admm_warmstart_dict)
        start_t = time.time()
        _ = admm_ws_exp.run()
        times["coord_ws"].append(time.time() - start_t)

        # Gurobi Partial (No Warmstart)
        cent_no_ws_exp = CentralizedSchedulingExperiment(partial_config)
        start_t = time.time()
        _ = cent_no_ws_exp.run()
        times["cent_no_ws"].append(time.time() - start_t)

        # Gurobi Partial (Warmstart)
        cent_ws_exp = CentralizedSchedulingExperiment(partial_config)
        if hasattr(cent_ws_exp, "apply_warmstart"):
            cent_ws_exp.apply_warmstart(cent_warmstart_dict)
        start_t = time.time()
        _ = cent_ws_exp.run()
        times["cent_ws"].append(time.time() - start_t)

    # Create the partial warmstart bar chart
    plot_path = os.path.join(plot_dir, plot_filename)
    PlotHandler.plot_warmstart(times, plot_path)
    logging.info(f"Bar chart saved to {plot_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main_warmstart_experiment.py <config_path>")
        sys.exit(1)

    config_file = sys.argv[1]
    main_warmstart_experiment(config_file)
