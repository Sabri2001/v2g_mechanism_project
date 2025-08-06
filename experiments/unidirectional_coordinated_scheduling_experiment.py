import logging
from copy import copy
import numpy as np
import gurobipy as gp
from gurobipy import GRB, Model
import time

# Import our base experiment & generic ADMM solver
from experiments.base_experiment import BaseExperiment
from experiments.admm_solver import ADMM


class UnidirectionalCoordinatedSchedulingExperiment(BaseExperiment):
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
        T = self.config["T"]
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
            "u": np.zeros((len(evs), T)),
            "soc": np.zeros((len(evs), T + 1)),
            "t_actual": np.zeros(len(evs), dtype=float),
            "dual": np.zeros(T),  # dual variable for the global constraint
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
            T=T,
            nu=nu,
            nu_multiplier=nu_multiplier,
            max_iter=max_iter,
            tol=1e-3,
            local_subproblem_fn=lambda ev_idx, st, old_st: self._solve_local_subproblem(evs, ev_idx, st, old_st, market_prices, start_time, end_time, T, dt, granularity),
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
        self._postprocess_results(evs, T, market_prices, final_state, start_time, dt, granularity)
        self.results["admm_iterations"] = self.admm_solver.iter_count
        self.results["nu_multiplier"] = self.config["nu_multiplier"]
        self.results["admm_solve_time"] = solve_time

        return self.results

    # -------------------------------------------------------------------------
    # Local Subproblem
    # -------------------------------------------------------------------------

    def _solve_local_subproblem(self, evs, ev_idx, state, old_state, market_prices, start_time, end_time, T, dt, granularity):
        """
        Builds and solves the local Gurobi subproblem for a single EV
        under the ADMM penalty for the absolute value of the sum of u.
        """
        ev = evs[ev_idx]
        alpha = ev["disconnection_time_flexibility"]
        beta = ev["soc_flexibility"]
        battery_wear = ev["battery_wear_cost_coefficient"]
        eff = ev["energy_efficiency"]
        M = ev["battery_capacity"] + ev["max_charge_rate"] * T
        start_step = start_time * granularity
        end_step = end_time * granularity

        # ADMM references from iteration_state
        dual = state["dual"]
        nu = self.admm_solver.nu

        # Set up a Gurobi model
        model = Model(f"EV_{ev['id']}")
        model.setParam("OutputFlag", 0)

        # Decision variables
        u = model.addVars(T, lb=0, ub=ev["max_charge_rate"], name="u")
        soc = model.addVars(T + 1, lb=0, ub=ev["battery_capacity"], name="soc")
        b = model.addVars(T, vtype=GRB.BINARY, name="b")  # disconnection indicator
        t_actual = model.addVar(vtype=GRB.INTEGER, lb= (start_step+1), ub=(end_step), name="t_actual")
        abs_u = model.addVars(T, lb=0, name="abs_u")
        delta = model.addVars(T, vtype=GRB.BINARY, name="delta") # t < t_actual indicator

        # Build objective
        operator_cost = 0
        for t in range(T):
            hour_idx = t // granularity
            operator_cost += market_prices[hour_idx] * u[t] * dt
            operator_cost += battery_wear * abs_u[t] * dt * eff

        # Quadratic penalty for disconnection time
        desired_substep = ev["disconnection_time"] * granularity
        operator_cost += 0.5 * alpha * ((desired_substep - t_actual)/granularity)**2

        # Quadratic penalty for SoC deviation
        delta_soc = model.addVar(lb=0, name="delta_soc")
        model.addConstr(delta_soc >= ev["desired_soc"] - soc[T], "delta_soc_constraint")
        operator_cost += 0.5 * beta * delta_soc * delta_soc

        # ADMM penalty: we add variables w[t]
        admm_cost = 0
        w = model.addVars(T, lb=0, name="w")
        # Define a variable to capture the absolute value of (u[t] + sum of other agents' u[t])
        local_abs = model.addVars(T, lb=0, name="local_abs")
        for t in range(T):
            # Get the contribution from other agents using their u values from the previous iteration.
            sum_other = np.sum([old_state["u"][j, t] for j in range(len(evs)) if j != ev_idx])
            # Enforce local_abs[t] to capture the absolute value of that sum.
            model.addConstr(local_abs[t] >= u[t] + sum_other, name=f"local_abs_pos_{t}")
            model.addConstr(local_abs[t] >= -(u[t] + sum_other), name=f"local_abs_neg_{t}")

            admm_cost += 0.5 / nu * (w[t]*w[t] - dual[t]*dual[t])
            model.addConstr(
                w[t] >= dual[t] + nu * (local_abs[t] - self.config["evcs_power_limit"]),
                name=f"wPos_{t}"
            )
            model.addConstr(w[t] >= 0, name=f"wNonNeg_{t}")

        model.setObjective(operator_cost + admm_cost, GRB.MINIMIZE)

        # Constraints
        # 1) Initial SoC
        model.addConstr(soc[0] == ev["initial_soc"], "InitialSoC")

        # 2) Exactly one disconnection time, that tracks t_actual
        model.addConstr(sum(b[t] for t in range(T)) == 1, "OneDisconnectionTime")
        model.addConstr(t_actual == sum((t + start_step + 1) * b[t] for t in range(T)), "tActualLink")

        # 3) SoC dynamics
        for t in range(T):
            model.addConstr(soc[t + 1] == soc[t] + dt * u[t] * eff, f"SoCDyn_{t}")

        # 4) abs_u, SoC threshold and charging constraints
        for t in range(T):
            model.addConstr(abs_u[t] >= u[t], f"AbsUPos_{t}")
            model.addConstr(abs_u[t] >= -u[t], f"AbsUNeg_{t}")

            # Define delta[t] to indicate if t < t_actual
            model.addConstr(t_actual - (t + start_step) >= 1 - M * (1 - delta[t]), f"Delta_Definition1_{t}")
            model.addConstr(t_actual - (t + start_step) <= M * delta[t], f"Delta_Definition2_{t}")

            # Charging/discharging limits dependent on delta[t]
            model.addConstr(u[t] <= ev["max_charge_rate"] * delta[t], f"MaxChargeRate_{t}")
            model.addConstr(u[t] >= -ev["max_discharge_rate"] * delta[t], f"MinDischargeRate_{t}")

        # 5) Min SoC constraints
        for t in range(T + 1):
            model.addConstr(soc[t] >= ev["min_soc"], f"MinSoC_{t}")

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            # Retrieve solutions
            sol_u = np.array([u[t].X for t in range(T)])
            sol_soc = np.array([soc[t].X for t in range(T+1)])
            sol_t_actual = t_actual.X
        else:
            logging.warning(f"EV {ev['id']} subproblem not optimal, status {model.status}")
            # fallback to old iteration values
            sol_u = state["u"][ev_idx, :]
            sol_soc = state["soc"][ev_idx, :]
            sol_t_actual = state["t_actual"][ev_idx]

        # Store solutions in iteration_state
        state["u"][ev_idx, :] = sol_u
        state["soc"][ev_idx, :] = sol_soc
        state["t_actual"][ev_idx] = sol_t_actual/granularity

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

    def _postprocess_results(self, evs, T, market_prices, final_state, start_time, dt, granularity):
        """
        Builds aggregator-level cost vectors and SoC tracking from the final ADMM state.
        """
        operator_cost_vector = np.zeros(T)
        energy_cost_vector = np.zeros(T)

        soc_over_time = {ev["id"]: [0.0]*(T+1) for ev in evs}
        desired_disconnection_time = []
        actual_disconnection_time = []

        # We'll also track total energy needed vs. total V2G
        total_energy_needed = 0.0
        total_energy_transferred = 0.0

        # 1) SoC results + disconnection times
        for i, ev in enumerate(evs):
            desired_disconnection_time.append(ev["disconnection_time"])
            actual_disconnection_time.append(final_state["t_actual"][i]) # already convert to hours

            # SoC
            for t in range(T+1):
                soc_val = final_state["soc"][i, t]
                soc_over_time[ev["id"]][t] = float(soc_val)

            # compute needed energy
            # needed_i = ev["desired_soc"] - ev["initial_soc"]
            # if needed_i > 0:
            #     total_energy_needed += needed_i

        # 2) Per-time-step energy and battery wear costs
        for t in range(T):
            sum_op_cost_t = 0.0
            sum_energy_cost_t = 0.0

            for i, ev in enumerate(evs):
                u_it = final_state["u"][i, t]
                cost_energy = market_prices[t//granularity] * u_it * dt
                cost_wear = (ev["battery_wear_cost_coefficient"] *
                             abs(u_it) *
                             ev["energy_efficiency"]) * dt

                sum_op_cost_t += (cost_energy + cost_wear)
                sum_energy_cost_t += cost_energy

                # if u_it < 0, we have V2G
                # if u_it < 0:
                #     total_energy_transferred += -u_it * dt

            operator_cost_vector[t] = sum_op_cost_t
            energy_cost_vector[t] = sum_energy_cost_t


        # 3) Per-EV soc and delay costs
        soc_cost_vector = np.zeros(len(evs))
        delay_cost_vector = np.zeros(len(evs))
        for i, ev in enumerate(evs):
            # SoC cost: quadratic penalty on final SoC deviation
            soc_cost_vector[i] = 0.5 * ev["soc_flexibility"] * (ev["desired_soc"] - final_state["soc"][i, T]) ** 2

            # Delay cost: quadratic penalty on disconnection time deviation
            delay_cost_vector[i] = 0.5 * ev["disconnection_time_flexibility"] * ((ev["disconnection_time"] - final_state["t_actual"][i]) ** 2)

        # 3) V2G fraction
        # v2g_fraction = 0.0
        # if total_energy_needed > 0:
        #     v2g_fraction = (total_energy_transferred / total_energy_needed)*100.0

        # 4) Save final dictionary
        total_cost = float(operator_cost_vector.sum() + energy_cost_vector.sum() +
                           soc_cost_vector.sum() + delay_cost_vector.sum())
        
        self.results = {
            "operator_cost_over_time": operator_cost_vector.tolist(),
            "energy_cost_over_time": energy_cost_vector.tolist(),
            "total_cost": total_cost,
            "sum_energy_cost": float(energy_cost_vector.sum()),
            "soc_over_time": soc_over_time,
            "desired_disconnection_time": desired_disconnection_time,
            "actual_disconnection_time": actual_disconnection_time,
            # "v2g_fraction": v2g_fraction,
            "u": final_state["u"].tolist(),
            "t_actual": final_state["t_actual"].tolist()
        }

        # 5) If walras_tax is enabled, compute additional metrics
        # if self.config.get("walras_tax", False):
        #     energy_cost_dict = {}
        #     adaptability_cost_dict = {}
        #     congestion_cost_dict = {}
        #     dual = final_state["dual"]  # Global dual variable array (length T)
        #     for i, ev in enumerate(evs):
        #         ev_id = ev["id"]

        #         # Energy cost: sum over t of market_prices[t] * u[i,t]
        #         energy_cost_ev = sum(market_prices[t//granularity] * final_state["u"][i, t] for t in range(T)) * dt
        #         energy_cost_dict[ev_id] = energy_cost_ev

        #         # Adaptability cost: quadratic penalty on disconnect time deviation
        #         adaptability_cost_ev = 0.5 * ev["disconnection_time_flexibility"] * ((ev["disconnection_time"] - final_state["t_actual"][i]) ** 2)
        #         adaptability_cost_dict[ev_id] = adaptability_cost_ev    

        #         # Congestion cost: extra cost due to dual variables ("Walras tax")
        #         congestion_cost_ev = sum(dual[t] * final_state["u"][i, t] for t in range(T)) * dt
        #         congestion_cost_dict[ev_id] = congestion_cost_ev

        #     # Add the new metrics to the results dictionary
        #     self.results["energy_cost"] = energy_cost_dict
        #     self.results["adaptability_cost"] = adaptability_cost_dict
        #     self.results["congestion_cost"] = congestion_cost_dict

        # 6) Compute individual costs (without payments)
        individual_cost = {}
        for i, ev in enumerate(evs):
            cost_i = 0.0
            for t in range(T):
                # Energy cost and battery wear cost
                cost_i +=  ev["battery_wear_cost_coefficient"] * abs(final_state["u"][i, t]) * ev["energy_efficiency"] * dt
            # Quadratic penalty on disconnection time deviation
            cost_i += 0.5 * ev["disconnection_time_flexibility"] * ((ev["disconnection_time"] - final_state["t_actual"][i]) ** 2)
            cost_i += 0.5 * ev["soc_flexibility"] * (ev["desired_soc"] - final_state["soc"][i, T]) ** 2
            individual_cost[ev["id"]] = cost_i
        # Store the per-EV cost breakdown in the results
        self.results["individual_cost"] = individual_cost

        # 7) Compute individual payments based on market prices
        self.results["individual_payment"] = {}
        for i, ev in enumerate(evs):
            # Initialize the payment for this EV
            individual_payment = 0.0
            # Compute the payment as the sum of market prices multiplied by u[i,t]
            for t in range(T):
                individual_payment += market_prices[t//granularity] * final_state["u"][i, t] * dt
            # Store the payment in the results dictionary
            self.results["individual_payment"][ev["id"]] = individual_payment

        # 8) If vcg is enabled, compute additional metrics
        if self.config.get("vcg", False):
            vcg_tax_dict = {}
            # Compute total cost for the "others" in the full run for each EV n
            for ev in evs:
                ev_id = ev["id"]
                # Sum cost for all EVs except the one with id ev_id
                original_others_cost = sum(
                    individual_cost[other_ev["id"]] + self.results["individual_payment"][other_ev["id"]]
                    for other_ev in evs if other_ev["id"] != ev_id
                )

                # Create a copy of the config and remove EV n
                config_without_ev = self.config.copy()
                # (Assuming a shallow copy is acceptable here; otherwise, use copy.deepcopy)
                config_without_ev["evs"] = [other_ev for other_ev in evs if other_ev["id"] != ev_id]
                config_without_ev["vcg"] = False  # Disable VCG tax calculation in the new run

                # Instantiate and run a new CoordinatedSchedulingExperiment for the remaining EVs
                from experiments.coordinated_scheduling_experiment import CoordinatedSchedulingExperiment
                experiment_without_ev = CoordinatedSchedulingExperiment(config_without_ev)
                results_without_ev = experiment_without_ev.run()

                # Extract individual cost breakdown from the run without EV n
                individual_cost_without_ev = results_without_ev.get("individual_cost", {})
                new_individual_payments = results_without_ev.get("individual_payment", {})
                new_others_cost = sum(
                    individual_cost_without_ev[ev_id] + new_individual_payments.get(ev_id, 0.0)
                    for ev_id in individual_cost_without_ev
                )

                # The VCG tax for EV n is the difference in the cost incurred by the others:
                vcg_tax_dict[ev_id] = original_others_cost - new_others_cost

            # Store the VCG tax breakdown in the results dictionary
            self.results["vcg_tax"] = vcg_tax_dict

        # Store time flexibility coefficient and desired disconnectino time for each EV
        self.results["disconnection_time_flexibilities"] = {
            ev["id"]: ev.get("disconnection_time_flexibility", 0.0)/2 for ev in evs
        }
        self.results["desired_disconnection_time"] = {
            ev["id"]: ev.get("disconnection_time", 0.0) for ev in evs
        }
