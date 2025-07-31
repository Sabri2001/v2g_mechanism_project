import os
import sys
import logging
import copy
import numpy as np
from datetime import datetime

from config.config_handler import ConfigHandler
from experiments.coordinated_scheduling_experiment import CoordinatedSchedulingExperiment
from results.plot_handler import PlotHandler


def run_coordinated_with_fake_alpha_tau(
    original_config,
    fake_alpha=None,
    fake_tau=None
    ):
    """
    1) Makes a deep copy of original_config.
    2) Overrides each EV's alpha (disconnection_time_flexibility) with `fake_alpha`
       and/or disconnection_time with `fake_tau` if provided.
    3) Runs the CoordinatedSchedulingExperiment to get an assignment schedule (u[t], t_actual, etc.).
    4) Then re-computes the operator cost using the *real* alpha/tau from the original config
       (that is, we do a custom cost calculation) and returns that cost.
    """

    import copy

    # 1) Copy config
    run_config = copy.deepcopy(original_config)

    # 2) Override each EV for scheduling
    for i, ev in enumerate(run_config["evs"]):
        if fake_alpha is not None:
            ev["disconnection_time_flexibility"] = fake_alpha
        if fake_tau is not None:
            ev["disconnection_time"] = fake_tau

    # 3) Run CoordinatedSchedulingExperiment to get a final schedule
    experiment = CoordinatedSchedulingExperiment(run_config)
    fake_results = experiment.run()  # results based on the overridden alpha/tau

    # 4) Re-compute the cost with the *actual* alpha/tau from original_config
    #    We'll write a helper function below to do this cost calculation from the final schedule.
    #    The function does not re-run optimization; it simply calculates cost from the schedule.
    real_cost = compute_real_cost(
        schedule=fake_results,
        schedule_config=run_config,
        original_config=original_config
    )

    return real_cost

def compute_real_cost(schedule, schedule_config, original_config):
    """
    Given:
     - 'schedule': the dictionary returned by CoordinatedSchedulingExperiment (u[t], t_actual, etc.)
       that was produced using possibly 'fake' alpha/tau.
     - 'schedule_config': the config used for the scheduling. (We need T, dt, market prices, etc.)
     - 'original_config': the config with each EV's *real* alpha, disconnection_time, etc.
    We do a custom operator cost calculation as if each EV had its real alpha and real desired disconnection_time,
    but using the same schedule (u[t], t_actual).

    Returns: float (sum_operator_costs) under the real parameters.
    """
    import numpy as np

    evs_fake = schedule_config["evs"]  # these were used in scheduling
    evs_real = original_config["evs"]  # real data
    T = schedule_config["T"]
    dt = schedule_config["dt"]
    granularity = schedule_config["granularity"]
    market_prices = schedule_config["market_prices"]

    # We have schedule["u"] and schedule["t_actual"], both shape (#EV, T)
    final_u = np.array(schedule["u"]) if "u" in schedule else None
    if final_u is None:
        raise ValueError("No 'u' array found in schedule. Did you store it in 'results'?")

    final_t_actual = schedule.get("t_actual", None)
    if final_t_actual is None:
        raise ValueError("No 't_actual' in schedule. Did you store it in 'results'?")

    # We'll compute sum_operator_costs = sum of (energy cost + battery wear + adaptability penalty).
    # The difference now is we want to use the *real* alpha and real 'disconnection_time' in the penalty.
    operator_cost_vector = np.zeros(T)

    # Map ev_ids to indices
    # We'll assume evs are in the same order, but let's confirm by ID. We'll create a dictionary:
    id_to_idx_fake = {ev["id"]: i for i, ev in enumerate(evs_fake)}
    id_to_idx_real = {ev["id"]: i for i, ev in enumerate(evs_real)}

    # We'll accumulate total cost
    total_cost = 0.0

    # 1) Energy + battery wear cost
    for t in range(T):
        hour_idx = t // granularity
        price_t = market_prices[hour_idx]
        sum_cost_t = 0.0
        for ev_fake in evs_fake:
            ev_id = ev_fake["id"]
            i_fake = id_to_idx_fake[ev_id]
            u_it = final_u[i_fake, t]
            # We want the battery wear from the real data? Actually, the battery_wear_coefficient is presumably the same
            # so let's keep the original or your call. Usually that doesn't change, so it might be in original_config as well.
            # We'll do: battery_wear_coefficient from the real config for consistency.
            i_real = id_to_idx_real[ev_id]
            ev_real = evs_real[i_real]

            battery_wear = ev_real["battery_wear_cost_coefficient"]
            eff = ev_real.get("energy_efficiency", 1.0)

            cost_energy = price_t * u_it * dt
            cost_wear   = battery_wear * abs(u_it) * eff * dt
            sum_cost_t += (cost_energy + cost_wear)
        operator_cost_vector[t] = sum_cost_t
        total_cost += sum_cost_t

    # 2) Adaptability penalty using *real* alpha and real desired_disconnection_time
    #    The scheduling gave us t_actual. We stored it in final_t_actual[i]. We want:
    #       cost_i += 0.5 * alpha_real * ( ( tau_desired_real - t_actual[i] )^2 )
    #    in hours, not sub-steps
    for ev_fake in evs_fake:
        ev_id = ev_fake["id"]
        i_fake = id_to_idx_fake[ev_id]
        i_real = id_to_idx_real[ev_id]

        alpha_real = evs_real[i_real]["disconnection_time_flexibility"]
        tau_real   = evs_real[i_real]["disconnection_time"]
        # t_actual we have from final_t_actual[i_fake], which is in "hours" presumably
        # (the post-process in your code divides by granularity).
        t_act = final_t_actual[i_fake]
        penalty = 0.5 * alpha_real * (tau_real - t_act)**2
        total_cost += penalty

    return float(total_cost)

def main(config_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1) Load base config to get number of runs etc.
    base_handler = ConfigHandler(config_path)
    base_config = base_handler.load_config()
    num_runs = base_config.get("num_runs", 20)

    start_time, end_time = base_config["time_range"]
    granularity = base_config["granularity"]
    T = (end_time - start_time) * granularity
    dt = 1.0 / granularity

    # 2) Create output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(
        "outputs",
        base_config.get("folder", "default_folder"),
        f"{base_config.get('name_xp', 'fake_alpha_tau')}_{timestamp}"
    )
    os.makedirs(base_output_dir, exist_ok=True)

    # 3) We'll define three experiment “scenarios”:
    # XP1: normal scheduling with real alpha/tau
    # XP2: scheduling with alpha=40, cost computed with real alpha
    xp_definitions = [
        ("Bid",  None,   None),   # no overrides
        ("No bid",       40.0,   None),   # override alpha
        # ("No time",      40.0,   17),   # override tau
    ]

    # We'll collect a list of cost values for each scenario
    chart_data = {
        "Bid":    [],
        "No bid":         [],
        # "No time":        []
    }

    # 4) Because we want random sampling, for each run we can re-instantiate a ConfigHandler
    #    or we can use the pre-sampled data from base_handler. Let's do the “instantiate each run” approach:
    for xp_label, fake_alpha, fake_tau in xp_definitions:
        handler = ConfigHandler(config_path)
        run_config = handler.load_config()  # real config
        sampled_data_per_run = handler.get_sampled_data_per_run()
        run_config["T"] = T
        run_config["dt"] = dt
        for run_idx in range(num_runs):
            # Select day
            sampled_data = sampled_data_per_run[run_idx % len(sampled_data_per_run)]

            # Insert the day data / EV parameters into run_config
            run_config["market_prices"]    = sampled_data["sampled_day_data"]["prices"]
            run_config["sampled_day_date"] = sampled_data["sampled_day_data"]["date"]
            run_config["evs"]             = sampled_data["evs"]
            # We have the *real* alpha/tau in run_config now

            # 5) Do the scheduling with (fake_alpha, fake_tau) if provided, then compute real cost
            real_cost = run_coordinated_with_fake_alpha_tau(
                original_config=run_config,
                fake_alpha=fake_alpha,
                fake_tau=fake_tau
            )
            chart_data[xp_label].append(real_cost)

    # 6) We can now plot the data
    plot_output_path = os.path.join(base_output_dir, "fake_alpha_tau_bar_chart.png")
    PlotHandler.plot_fake_alpha_tau_bars(chart_data, plot_output_path)

    logging.info("All runs completed. Bar chart saved.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_coordinated_fake_alpha_tau.py <config_path>")
        sys.exit(1)
    main(sys.argv[1])
