import logging
from copy import copy
import numpy as np
import gurobipy as gp
from gurobipy import GRB, Model

# Import our base experiment & generic ADMM solver
from experiments.base_experiment import BaseExperiment
from experiments.admm_solver import ADMM


class InelasticCoordinatedSchedulingExperiment(BaseExperiment):
    """
    Demonstrates a Coupled Coordinated Scheduling approach with ADMM,
    enforcing a global constraint on the sum of absolute (charge/discharge).
    """

    def run(self):
        """
        High-level method orchestrating the ADMM process and building final results.
        """
        # 1) Extract data from config
        start_time, end_time = self.config["time_range"]
        T = end_time - start_time
        evs = self.config["evs"]
        market_prices = self.config["market_prices"]
        evcs_power_limit = self.config["evcs_power_limit"]

        # 2) Prepare iteration_state for ADMM
        iteration_state = {
            "u": np.zeros((len(evs), T)),
            "abs_u": np.zeros((len(evs), T)),
            "soc": np.zeros((len(evs), T + 1)),
            "t_actual": np.zeros(len(evs), dtype=int),
            "mu": np.zeros(T),  # dual variable for the global constraint
        }

        # 3) Build an ADMM instance
        self.admm_solver = ADMM(
            num_agents=len(evs),
            T=T,
            beta=0.1,
            max_iter=100,
            tol=1e-3,
            local_subproblem_fn=lambda ev_idx, st, old_st: self._solve_local_subproblem(evs, ev_idx, st, old_st, market_prices, start_time, end_time),
            global_step_fn=lambda st: self._global_step(st, evcs_power_limit)
        )
        self.admm_solver.iteration_state = iteration_state
        self.admm_solver.old_iteration_state = copy(iteration_state) # shallow copy

        # 4) Solve
        final_state = self.admm_solver.solve()

        # 5) Aggregate results (costs, SoC, etc.)
        self._postprocess_results(evs, T, market_prices, final_state, start_time)

        return self.results

    # -------------------------------------------------------------------------
    # Local Subproblem
    # -------------------------------------------------------------------------

    def _solve_local_subproblem(self, evs, ev_idx, state, old_state, market_prices, start_time, end_time):
        """
        Builds and solves the local Gurobi subproblem for a single EV
        under the ADMM penalty for the sum of |u|.
        """
        ev = evs[ev_idx]
        alpha = ev["disconnection_time_preference_coefficient"]
        beta_wear = ev["battery_wear_cost_coefficient"]
        eff = ev["energy_efficiency"]
        T = end_time - start_time
        M = ev["battery_capacity"] + ev["max_charge_rate"] * T

        # ADMM references from iteration_state
        old_abs_u = old_state["abs_u"]
        mu = state["mu"]
        beta = self.admm_solver.beta

        # Set up a Gurobi model
        model = Model(f"EV_{ev['id']}")
        model.setParam("OutputFlag", 0)

        # Decision variables
        u = model.addVars(T, lb=-ev["max_discharge_rate"], ub=ev["max_charge_rate"], name="u")
        soc = model.addVars(T + 1, lb=0, ub=ev["battery_capacity"], name="soc")
        b = model.addVars(T, vtype=GRB.BINARY, name="b")  # disconnection indicator
        # t_actual = model.addVar(vtype=GRB.INTEGER, lb=start_time + 1, ub=end_time, name="t_actual")
        abs_u = model.addVars(T, lb=0, name="abs_u")
        z = model.addVars(T, vtype=GRB.BINARY, name="z") # SoC threshold indicator
        delta = model.addVars(T, vtype=GRB.BINARY, name="delta") # t < t_actual indicator

        t_actual = ev["disconnection_time"] # not a decision variable anymore

        # Build objective
        operator_cost = 0
        for t in range(T):
            operator_cost += market_prices[t] * u[t]
            operator_cost += beta_wear * abs_u[t] * eff

        # ADMM penalty: we add variables w[t]
        admm_cost = 0
        w = model.addVars(T, lb=0, name="w")
        for t in range(T):
            # sum of other abs_u at time t
            sum_other = np.sum([old_abs_u[j, t] for j in range(len(evs)) if j != ev_idx])
            admm_cost += 0.5 / beta * (w[t]*w[t] - mu[t]*mu[t])

            model.addConstr(w[t] >= mu[t] + beta * (abs_u[t] + sum_other - self.config["evcs_power_limit"]), name=f"wPos_{t}")
            model.addConstr(w[t] >= 0, name=f"wNonNeg_{t}")

        model.setObjective(operator_cost + admm_cost, GRB.MINIMIZE)

        # Constraints
        # 1) Initial SoC
        model.addConstr(soc[0] == ev["initial_soc"], "InitialSoC")

        # 2) Exactly one disconnection time, that tracks t_actual
        model.addConstr(sum(b[t] for t in range(T)) == 1, "OneDisconnectionTime")
        model.addConstr(t_actual == sum((t + start_time + 1) * b[t] for t in range(T)), "tActualLink")

        # 3) Desired final SoC constraints
        for t in range(T):
            model.addConstr(soc[t + 1] >= ev["desired_soc"] - (1 - b[t]) * M, f"SoCPos_{t}")
            model.addConstr(soc[t + 1] <= ev["desired_soc"] + (1 - b[t]) * M, f"SoCNeg_{t}")

        # 4) SoC dynamics
        for t in range(T):
            model.addConstr(soc[t + 1] == soc[t] + u[t] * eff, f"SoCDyn_{t}")

        # 5) abs_u, SoC threshold and charging constraints
        for t in range(T):
            model.addConstr(abs_u[t] >= u[t], f"AbsUPos_{t}")
            model.addConstr(abs_u[t] >= -u[t], f"AbsUNeg_{t}")

            # SoC threshold constraints
            model.addConstr(soc[t + 1] >= ev["soc_threshold"] - M * (1 - z[t]), f"SoCThresholdLower_{t}")
            model.addConstr(soc[t + 1] <= ev["soc_threshold"] + M * z[t], f"SoCThresholdUpper_{t}")

            # Discharge limit based on SoC threshold
            model.addConstr(u[t] >= -ev["max_discharge_rate"] * z[t], f"DischargeLimit_{t}")

            # Define delta[t] to indicate if t < t_actual
            model.addConstr(t_actual - (t + start_time) >= 1 - M * (1 - delta[t]), f"Delta_Definition1_{t}")
            model.addConstr(t_actual - (t + start_time) <= M * delta[t], f"Delta_Definition2_{t}")

            # Charging/discharging limits dependent on delta[t]
            model.addConstr(u[t] <= ev["max_charge_rate"] * delta[t], f"MaxChargeRate_{t}")
            model.addConstr(u[t] >= -ev["max_discharge_rate"] * delta[t], f"MinDischargeRate_{t}")

        # 6) Min SoC constraints
        for t in range(T + 1):
            model.addConstr(soc[t] >= ev["min_soc"], f"MinSoC_{t}")

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            # Retrieve solutions
            sol_u = np.array([u[t].X for t in range(T)])
            sol_abs_u = np.array([abs_u[t].X for t in range(T)])
            sol_soc = np.array([soc[t].X for t in range(T+1)])
            sol_t_actual = t_actual
        else:
            logging.warning(f"EV {ev['id']} subproblem not optimal, status {model.status}")
            # fallback to old iteration values
            sol_u = state["u"][ev_idx, :]
            sol_abs_u = state["abs_u"][ev_idx, :]
            sol_soc = state["soc"][ev_idx, :]
            sol_t_actual = state["t_actual"][ev_idx]

        # Store solutions in iteration_state
        state["u"][ev_idx, :] = sol_u
        state["abs_u"][ev_idx, :] = sol_abs_u
        state["soc"][ev_idx, :] = sol_soc
        state["t_actual"][ev_idx] = sol_t_actual

    # -------------------------------------------------------------------------
    # Global Step
    # -------------------------------------------------------------------------

    def _global_step(self, state, evcs_power_limit):
        """
        Updates the dual variable 'mu' based on the sum of abs_u across all EVs.
        """
        sum_abs_u = np.sum(state["abs_u"], axis=0)
        mu = state["mu"]
        beta = self.admm_solver.beta

        # mu(t+1) = max( 0, mu(t) + beta( sum_abs_u - evcs_power_limit ) )
        new_mu = np.maximum(0, mu + beta * (sum_abs_u - evcs_power_limit))
        state["mu"] = new_mu

    # -------------------------------------------------------------------------
    # Post-processing
    # -------------------------------------------------------------------------

    def _postprocess_results(self, evs, T, market_prices, final_state, start_time):
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
            actual_disconnection_time.append(int(final_state["t_actual"][i]))

            # SoC
            for t in range(T+1):
                soc_val = final_state["soc"][i, t]
                soc_over_time[ev["id"]][t] = float(soc_val)

            # compute needed energy
            needed_i = ev["desired_soc"] - ev["initial_soc"]
            if needed_i > 0:
                total_energy_needed += needed_i

        # 2) Per-time-step aggregator costs
        for t in range(T):
            sum_op_cost_t = 0.0
            sum_energy_cost_t = 0.0

            for i, ev in enumerate(evs):
                u_it = final_state["u"][i, t]
                cost_energy = market_prices[t] * u_it
                cost_wear = (ev["battery_wear_cost_coefficient"] *
                             abs(u_it) *
                             ev["energy_efficiency"])

                sum_op_cost_t += (cost_energy + cost_wear)
                sum_energy_cost_t += cost_energy

                # if u_it < 0, we have V2G
                if u_it < 0:
                    total_energy_transferred += -u_it

            operator_cost_vector[t] = sum_op_cost_t
            energy_cost_vector[t] = sum_energy_cost_t

        # 3) V2G fraction
        v2g_fraction = 0.0
        if total_energy_needed > 0:
            v2g_fraction = (total_energy_transferred / total_energy_needed)*100.0

        # 4) Save final dictionary
        self.results = {
            "operator_cost_vector": operator_cost_vector.tolist(),
            "energy_cost_vector": energy_cost_vector.tolist(),
            "sum_operator_cost": float(operator_cost_vector.sum()),
            "sum_energy_costs": float(energy_cost_vector.sum()),
            "soc_over_time": soc_over_time,
            "desired_disconnection_time": desired_disconnection_time,
            "actual_disconnection_time": actual_disconnection_time,
            "v2g_fraction": v2g_fraction,
        }


# -----------------------------------------------------------------------------
# Example usage if you run this file directly
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config_example = {
        "time_range": (0, 4),
        "market_prices": [0.1, 0.12, 0.08, 0.1],
        "evcs_power_limit": 10.0,   # global constraint
        "evs": [
            {
                "id": 0,
                "battery_capacity": 40,
                "initial_soc": 15,
                "desired_soc": 35,
                "max_charge_rate": 5,
                "max_discharge_rate": 4,
                "min_soc": 0,
                "soc_threshold": 12,
                "battery_wear_cost_coefficient": 0.13,
                "disconnection_time_preference_coefficient": 2,
                "energy_efficiency": 0.87,
                "disconnection_time": 3,
            },
            {
                "id": 1,
                "battery_capacity": 35,
                "initial_soc": 5,
                "desired_soc": 25,
                "max_charge_rate": 4,
                "max_discharge_rate": 4,
                "min_soc": 0,
                "soc_threshold": 10,
                "battery_wear_cost_coefficient": 0.1,
                "disconnection_time_preference_coefficient": 4,
                "energy_efficiency": 0.9,
                "disconnection_time": 4,
            },
        ],
    }

    experiment = CoupledCoordinatedSchedulingExperiment(config_example)
    results = experiment.run()

    print("\n===== FINAL RESULTS =====")
    for k, v in results.items():
        print(f"{k}: {v}")
