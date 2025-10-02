import logging
from copy import copy
import numpy as np
import gurobipy as gp
from gurobipy import GRB, Model
import time

# Import our base experiment & generic ADMM solver
from experiments.base_experiment import BaseExperiment
from experiments.admm_solver import ADMM


class CoordinatedSchedulingExperiment(BaseExperiment):
    """
    Implements a Coupled Coordinated Scheduling approach with ADMM,
    enforcing a global constraint on the sum of absolute (charge/discharge).
    """
    def __init__(self, config):
        super().__init__(config)
        self.warmstart_solutions = None

    def apply_warmstart(self, warmstart_dict):
        """
        warmstart_dict: { ev_id -> list of length T with power values }
        """
        self.warmstart_solutions = warmstart_dict

    def run(self):
        """
        High-level method orchestrating the ADMM process and building final results.
        """
        # 1) Extract data from config
        start_time, end_time = self.config["time_range"]
        nb_time_steps = self.config["nb_time_steps"]
        dt = self.config["dt"]
        granularity = self.config["granularity"]
        evs = self.config["evs"]
        market_prices = self.config["market_prices"]
        evcs_power_limit = self.config["evcs_power_limit"]

        nu = self.config["nu"]
        nu_multiplier = self.config["nu_multiplier"]
        max_iter = self.config["max_iter"]

        # 2) Prepare iteration_state for ADMM
        iteration_state = {
            "u": np.zeros((len(evs), nb_time_steps)),
            "soc": np.zeros((len(evs), nb_time_steps + 1)),
            "t_actual": np.zeros(len(evs), dtype=float),
            "dual": np.zeros(nb_time_steps),  # dual variable for the global constraint
        }
        if self.warmstart_solutions is not None:
            for i, ev in enumerate(evs):
                ev_id = ev["id"]
                # Only fill for that EV if present
                if ev_id in self.warmstart_solutions:
                    iteration_state["u"][i,:] = self.warmstart_solutions[ev_id]

        # 3) Build an ADMM instance
        self.admm_solver = ADMM(
            num_agents=len(evs),
            nb_time_steps=nb_time_steps,
            nu=nu,
            nu_multiplier=nu_multiplier,
            max_iter=max_iter,
            tol=1e-3,
            local_subproblem_fn=lambda ev_idx, st, old_st: self._solve_local_subproblem(evs, ev_idx, st, old_st, market_prices, start_time, end_time, nb_time_steps, dt, granularity),
            global_step_fn=lambda st: self._global_step(st, evcs_power_limit)
        )
        self.admm_solver.iteration_state = iteration_state
        self.admm_solver.old_iteration_state = copy(iteration_state) # shallow copy

        # 4) Solve
        start_t = time.time()
        final_state = self.admm_solver.solve()
        solve_time = time.time() - start_t
        logging.info(f"Coordinated scheduling took {solve_time:.2f} seconds.")

        # 5) Aggregate results (costs, SoC, etc.)
        self._postprocess_results(evs, nb_time_steps, market_prices, final_state, start_time, dt, granularity)
        self.results["admm_iterations"] = self.admm_solver.iter_count
        self.results["nu_multiplier"] = self.config["nu_multiplier"]
        self.results["admm_solve_time"] = solve_time

        return self.results

    # -------------------------------------------------------------------------
    # Local Subproblem
    # -------------------------------------------------------------------------

    def _solve_local_subproblem(self, evs, ev_idx, state, old_state, market_prices, start_time, end_time, nb_time_steps, dt, granularity):
        """
        Builds and solves the local Gurobi subproblem for a single EV
        under the ADMM penalty for the absolute value of the sum of u.
        """
        ev = evs[ev_idx]
        alpha = ev["disconnection_time_flexibility"]
        beta = ev["soc_flexibility"]
        battery_wear = ev["battery_wear_cost_coefficient"]
        eff = ev["energy_efficiency"]
        M = ev["battery_capacity"] + ev["max_charge_rate"] * nb_time_steps
        start_step = start_time * granularity
        end_step = end_time * granularity

        # ADMM references from iteration_state
        dual = state["dual"]
        nu = self.admm_solver.nu

        # Set up a Gurobi model
        model = Model(f"EV_{ev['id']}")
        model.setParam("OutputFlag", 0)

        # Decision variables
        u = model.addVars(nb_time_steps, lb=-ev["max_discharge_rate"], ub=ev["max_charge_rate"], name="u")
        soc = model.addVars(nb_time_steps + 1, lb=0, ub=ev["battery_capacity"], name="soc")
        b = model.addVars(nb_time_steps, vtype=GRB.BINARY, name="b")  # disconnection indicator
        actual_disconnection_step = model.addVar(vtype=GRB.INTEGER, lb= (start_step+1), ub=(end_step), name="actual_disconnection_step")
        abs_u = model.addVars(nb_time_steps, lb=0, name="abs_u")
        delta = model.addVars(nb_time_steps, vtype=GRB.BINARY, name="delta") # step < actual_disconnection_step indicator

        # Build objective
        operator_cost = 0
        for step in range(nb_time_steps):
            hour_idx = step // granularity
            operator_cost += market_prices[hour_idx] * u[step] * dt
            operator_cost += battery_wear * abs_u[step] * dt * eff

        # Quadratic penalty for disconnection time
        desired_substep = ev["disconnection_time"] * granularity
        operator_cost += 0.5 * alpha * ((desired_substep - actual_disconnection_step)/granularity)**2

        # Quadratic penalty for SoC deviation
        delta_soc = model.addVar(lb=0, name="delta_soc")
        model.addConstr(delta_soc >= ev["desired_soc"] - soc[nb_time_steps], "delta_soc_constraint")
        operator_cost += 0.5 * beta * delta_soc * delta_soc

        # ADMM penalty: we add variables w[t]
        admm_cost = 0
        w = model.addVars(nb_time_steps, lb=0, name="w")
        # Define a variable to capture the absolute value of (u[t] + sum of other agents' u[t])
        local_abs = model.addVars(nb_time_steps, lb=0, name="local_abs")
        for step in range(nb_time_steps):
            # Get the contribution from other agents using their u values from the previous iteration.
            sum_other = np.sum([old_state["u"][j, step] for j in range(len(evs)) if j != ev_idx])
            # Enforce local_abs[step] to capture the absolute value of that sum.
            model.addConstr(local_abs[step] >= u[step] + sum_other, name=f"local_abs_pos_{step}")
            model.addConstr(local_abs[step] >= -(u[step] + sum_other), name=f"local_abs_neg_{step}")

            admm_cost += 0.5 / nu * (w[step]*w[step] - dual[step]*dual[step])
            model.addConstr(
                w[step] >= dual[step] + nu * (local_abs[step] - self.config["evcs_power_limit"]),
                name=f"wPos_{step}"
            )
            model.addConstr(w[step] >= 0, name=f"wNonNeg_{step}")

        model.setObjective(operator_cost + admm_cost, GRB.MINIMIZE)

        # Constraints
        # 1) Initial SoC
        model.addConstr(soc[0] == ev["initial_soc"], "InitialSoC")

        # 2) Exactly one disconnection time, that tracks t_actual
        model.addConstr(sum(b[step] for step in range(nb_time_steps)) == 1, "OneDisconnectionTime")
        model.addConstr(actual_disconnection_step == sum((step + start_step + 1) * b[step] for step in range(nb_time_steps)), "stepActualLink")

        # 3) SoC dynamics
        for step in range(nb_time_steps):
            model.addConstr(soc[step + 1] == soc[step] + dt * u[step] * eff, f"SoCDyn_{step}")

        # 4) abs_u, SoC threshold and charging constraints
        for step in range(nb_time_steps):
            model.addConstr(abs_u[step] >= u[step], f"AbsUPos_{step}")
            model.addConstr(abs_u[step] >= -u[step], f"AbsUNeg_{step}")

            # Define delta[step] to indicate if step < actual_disconnection_step
            model.addConstr(actual_disconnection_step - (step + start_step) >= 1 - M * (1 - delta[step]), f"Delta_Definition1_{step}")
            model.addConstr(actual_disconnection_step - (step + start_step) <= M * delta[step], f"Delta_Definition2_{step}")

            # Charging/discharging limits dependent on delta[step]
            model.addConstr(u[step] <= ev["max_charge_rate"] * delta[step], f"MaxChargeRate_{step}")
            model.addConstr(u[step] >= -ev["max_discharge_rate"] * delta[step], f"MinDischargeRate_{step}")

        # 5) Min SoC constraints
        for step in range(nb_time_steps + 1):
            model.addConstr(soc[step] >= ev["min_soc"], f"MinSoC_{step}")

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            # Retrieve solutions
            sol_u = np.array([u[step].X for step in range(nb_time_steps)])
            sol_soc = np.array([soc[step].X for step in range(nb_time_steps + 1)])
            sol_actual_disconnection_step = actual_disconnection_step.X
        else:
            logging.warning(f"EV {ev['id']} subproblem not optimal, status {model.status}")
            # fallback to old iteration values
            sol_u = state["u"][ev_idx, :]
            sol_soc = state["soc"][ev_idx, :]
            sol_actual_disconnection_step = state["t_actual"][ev_idx] * granularity

        # Store solutions in iteration_state
        state["u"][ev_idx, :] = sol_u
        state["soc"][ev_idx, :] = sol_soc
        state["t_actual"][ev_idx] = sol_actual_disconnection_step/granularity

    # -------------------------------------------------------------------------
    # Global Step
    # -------------------------------------------------------------------------

    def _global_step(self, state, evcs_power_limit):
        """
        Updates the dual variable based on the absolute value of the sum of u_n across all EVs.
        """
        sum_u = np.sum(state["u"], axis=0)
        dual = state["dual"]
        nu = self.admm_solver.nu

        new_dual = np.maximum(0, dual + nu * (np.abs(sum_u) - evcs_power_limit))
        state["dual"] = new_dual

    # -------------------------------------------------------------------------
    # Post-processing
    # -------------------------------------------------------------------------

    def _postprocess_results(self, evs, nb_time_steps, market_prices, final_state, start_time, dt, granularity):
        """
        Builds aggregator-level cost vectors and SoC tracking from the final ADMM state.
        """
        wear_cost_vector = np.zeros(nb_time_steps)
        energy_cost_vector = np.zeros(nb_time_steps)

        soc_over_time = {ev["id"]: [0.0]*(nb_time_steps+1) for ev in evs}
        desired_disconnection_time = []
        actual_disconnection_time = []

        # 1) SoC results + disconnection times
        for i, ev in enumerate(evs):
            desired_disconnection_time.append(ev["disconnection_time"])
            actual_disconnection_time.append(final_state["t_actual"][i]) # already converted to hours

            # SoC
            for step in range(nb_time_steps+1):
                soc_val = final_state["soc"][i, step]
                soc_over_time[ev["id"]][step] = float(soc_val)

        # 2) Per-time-step energy and battery wear costs
        for step in range(nb_time_steps):
            sum_wear_cost_t = 0.0
            sum_energy_cost_t = 0.0

            for i, ev in enumerate(evs):
                u_it = final_state["u"][i, step]
                cost_energy = market_prices[step//granularity] * u_it * dt
                cost_wear = (ev["battery_wear_cost_coefficient"] *
                             abs(u_it) *
                             ev["energy_efficiency"]) * dt

                sum_wear_cost_t += cost_wear
                sum_energy_cost_t += cost_energy

            wear_cost_vector[step] = sum_wear_cost_t
            energy_cost_vector[step] = sum_energy_cost_t


        # 3) Per-EV soc and delay costs
        soc_cost_vector = np.zeros(len(evs))
        delay_cost_vector = np.zeros(len(evs))
        for i, ev in enumerate(evs):
            # SoC cost: quadratic penalty on final SoC deviation
            soc_cost_vector[i] = 0.5 * ev["soc_flexibility"] * (ev["desired_soc"] - final_state["soc"][i, nb_time_steps]) ** 2

            # Delay cost: quadratic penalty on disconnection time deviation
            delay_cost_vector[i] = 0.5 * ev["disconnection_time_flexibility"] * ((ev["disconnection_time"] - final_state["t_actual"][i]) ** 2)

        # 4) Save final dictionary
        total_cost = float(wear_cost_vector.sum() + energy_cost_vector.sum() +
                           soc_cost_vector.sum() + delay_cost_vector.sum())
        
        self.results = {
            "wear_cost_over_time": wear_cost_vector.tolist(),
            "energy_cost_over_time": energy_cost_vector.tolist(),
            "total_cost": total_cost,
            "sum_energy_cost": float(energy_cost_vector.sum()),
            "soc_over_time": soc_over_time,
            "desired_disconnection_time": desired_disconnection_time,
            "actual_disconnection_time": actual_disconnection_time,
            "u": final_state["u"].tolist(),
            "t_actual": final_state["t_actual"].tolist()
        }
