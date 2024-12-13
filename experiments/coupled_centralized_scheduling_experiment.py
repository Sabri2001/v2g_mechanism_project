from experiments.base_experiment import BaseExperiment
from gurobipy import Model, GRB, Env


class CoupledCentralizedSchedulingExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)
        # ADMM parameters
        self.rho = config.get("rho", 1.0)  # Penalty parameter for ADMM
        self.max_iterations = config.get("max_admm_iterations", 20)
        self.tolerance = config.get("admm_tolerance", 1e-2)

    def solve_ev_subproblem(self, ev, market_prices, T, start_time, end_time, dual_vars):
        """
        Solve each EV subproblem under Exchange ADMM.
        'dual_vars' is a list of length T with the dual penalty for sum(|u[t]|).
        The local EV objective includes the original cost + dual_var[t] * |u[t]|.
        """
        model = Model(f"EV_Scheduling_{ev['id']}")
        model.setParam("LogToConsole", 0)  # Silence the console output

        energy_efficiency = ev["energy_efficiency"]

        # Variables
        u = model.addVars(T, lb=-ev["max_discharge_rate"], ub=ev["max_charge_rate"], name="u")
        soc = model.addVars(T + 1, lb=0, ub=ev["battery_capacity"], name="soc")
        b = model.addVars(T, vtype=GRB.BINARY, name="b")  # Binary for disconnection time
        t_actual = model.addVar(vtype=GRB.INTEGER, lb=start_time + 1, ub=end_time, name="t_actual")
        abs_u = model.addVars(T, lb=0, name="abs_u")
        z = model.addVars(T, vtype=GRB.BINARY, name="z")
        delta = model.addVars(T, vtype=GRB.BINARY, name="delta")

        # Big-M constant
        M = ev["battery_capacity"] + ev["max_charge_rate"] * T

        # Coefficients
        beta = ev["battery_wear_cost_coefficient"]
        alpha = ev["disconnection_time_preference_coefficient"]

        # Objective components
        operator_objective = 0
        energy_cost = 0
        dual_penalty = 0

        for t in range(T):
            # Market energy cost
            energy_cost += market_prices[t] * u[t]
            # Wear cost (using the absolute value of usable energy)
            operator_objective += beta * abs_u[t] * energy_efficiency
            # Dual penalty from Exchange ADMM for violating the global constraint sum(|u[t]|)
            # This is: dual_vars[t] * |u[t]| (with sign depending on how we do the ADMM update)
            dual_penalty += dual_vars[t] * abs_u[t]

        # Quadratic disconnection deviation cost
        operator_objective += 0.5 * alpha * (ev["disconnect_time"] - t_actual) ** 2
        # Combine everything
        total_objective = operator_objective + energy_cost + dual_penalty
        model.setObjective(total_objective, GRB.MINIMIZE)

        # Constraints
        # Initial SoC
        model.addConstr(soc[0] == ev["initial_soc"], "InitialSoC")
        # Only one disconnection time
        model.addConstr(sum(b[t] for t in range(T)) == 1, "OneDisconnectionTime")
        # Link t_actual with b[t]
        model.addConstr(t_actual == sum((t + start_time + 1) * b[t] for t in range(T)), "tActualLink")

        # SoC constraints
        for t in range(T):
            # Final SoC constraint (reach desired_soc at the chosen disconnection time)
            model.addConstr(soc[t + 1] >= ev["desired_soc"] - (1 - b[t]) * M, f"FinalSoCPos_{t}")
            model.addConstr(soc[t + 1] <= ev["desired_soc"] + (1 - b[t]) * M, f"FinalSoCNeg_{t}")

            # SoC dynamics
            model.addConstr(soc[t + 1] == soc[t] + energy_efficiency * u[t], f"SoCDynamics_{t}")

            # Absolute value tracking
            model.addConstr(abs_u[t] >= u[t], f"AbsU_Pos_{t}")
            model.addConstr(abs_u[t] >= -u[t], f"AbsU_Neg_{t}")

            # SoC threshold constraints
            model.addConstr(soc[t + 1] >= ev["soc_threshold"] - M * (1 - z[t]), f"SoCThresholdLower_{t}")
            model.addConstr(soc[t + 1] <= ev["soc_threshold"] + M * z[t], f"SoCThresholdUpper_{t}")

            # Discharge limit based on threshold
            model.addConstr(u[t] >= -ev["max_discharge_rate"] * z[t], f"DischargeLimit_{t}")

            # Delta[t] to indicate t < t_actual
            model.addConstr(t_actual - (t + start_time) >= 1 - M * (1 - delta[t]), f"Delta_Def1_{t}")
            model.addConstr(t_actual - (t + start_time) <= M * delta[t], f"Delta_Def2_{t}")

            # Charging/discharging limit if EV is not yet disconnected
            model.addConstr(u[t] <= ev["max_charge_rate"] * delta[t], f"MaxChargeRate_{t}")
            model.addConstr(u[t] >= -ev["max_discharge_rate"] * delta[t], f"MinDischargeRate_{t}")

        # Minimum SoC constraints
        for t in range(T + 1):
            model.addConstr(soc[t] >= ev["min_soc"], f"MinSoC_{t}")

        model.optimize()

        model.optimize()
        if model.status == GRB.OPTIMAL or model.Status == GRB.SUBOPTIMAL:
            print("Model subproblem solved. Status:", model.status)
            schedule = {
            "u": [u[t].X for t in range(T)],
            "soc": [soc[t].X for t in range(T + 1)],
            "t_actual": t_actual.X,
            "energy_cost": sum(market_prices[t] * u[t].X for t in range(T)),
            "dual_penalty": sum(dual_vars[t] * abs_u[t].X for t in range(T)),
            "total_cost": model.ObjVal
            }
            return schedule
        else:
            print("Model subproblem is infeasible or no optimal solution found. Status:", model.status)

    def run(self):
        """
        Exchange ADMM approach:
          - We have a global constraint sum_i |u_{i,t}| <= evcs_power_limit for each t.
          - Introduce a dual variable lambda[t] for each time slot t that penalizes violation.
          - Each ADMM iteration:
              1) Solve local EV subproblems using the current dual variables.
              2) Aggregate solutions, compute sum(|u_{i,t}|).
              3) Update the dual variables (lambda[t]) to push toward feasibility.
              4) Repeat until convergence or max_iterations.
        """
        start_time, end_time = self.config["time_range"]
        T = end_time - start_time
        market_prices = self.config["market_prices"]
        evs = self.config["evs"]
        evcs_power_limit = self.config["evcs_power_limit"]  # The new global capacity constraint

        # Dual variables for each time slot: lambda[t], initialized to 0
        dual_vars = [0.0] * T

        # Bookkeeping for ADMM iteration
        schedules = [None] * len(evs)

        for iteration in range(self.max_iterations):
            # Step 1: Solve subproblems for each EV
            for i, ev in enumerate(evs):
                schedules[i] = self.solve_ev_subproblem(ev, market_prices, T, start_time, end_time, dual_vars)

            # Step 2: Aggregate to check feasibility w.r.t. global constraint
            # sum_abs_u[t] = sum of absolute charging/discharging of all EVs at time t
            sum_abs_u = [0.0] * T
            for t in range(T):
                for schedule in schedules:
                    sum_abs_u[t] += abs(schedule["u"][t])

            # Step 3: Update dual variables (lambda[t]) for the constraint sum_abs_u[t] <= evcs_power_limit
            # A typical ADMM update: lambda[t] <- lambda[t] + rho * (sum_abs_u[t] - evcs_power_limit)
            # If sum_abs_u[t] > evcs_power_limit, penalty increases
            # We’ll do a “projected gradient” style update (clamping if needed).
            print(f"Dual variables before iteration {iteration+1}: {dual_vars}")
            violation_norm = 0.0
            for t in range(T):
                violation = max(0, sum_abs_u[t] - evcs_power_limit)
                dual_vars[t] += self.rho * violation  # Projection onto lambda[t] >= 0
                violation_norm += violation

            # Step 4: Check convergence
            avg_violation = violation_norm / T
            if avg_violation < self.tolerance:
                print(f"ADMM converged at iteration {iteration+1}, violation = {avg_violation}")
                break
            else:
                print(f"Iteration {iteration+1}, violation = {avg_violation} \n")

        # After ADMM finishes, gather final results
        operator_objective_vector = [0] * T
        energy_cost_vector = [0] * T
        soc_over_time = {}
        desired_disconnect_time = []
        actual_disconnect_time = []

        sum_operator_obj = 0.0
        sum_energy_costs = 0.0

        for i, ev in enumerate(evs):
            schedule = schedules[i]
            soc_over_time[ev["id"]] = schedule["soc"]
            desired_disconnect_time.append(ev["disconnect_time"])
            actual_disconnect_time.append(schedule["t_actual"])

            # Recompute operator cost from the final local schedule
            local_obj = 0.0
            for t in range(T):
                # Energy cost
                e_cost = schedule["u"][t] * market_prices[t]
                # Battery wear
                usable_energy = schedule["u"][t] * ev["energy_efficiency"]
                wear_cost = ev["battery_wear_cost_coefficient"] * abs(usable_energy)
                local_obj += (e_cost + wear_cost)

                operator_objective_vector[t] += (e_cost + wear_cost)
                energy_cost_vector[t] += e_cost

            # Add disconnection penalty (final state)
            alpha = ev["disconnection_time_preference_coefficient"]
            local_obj += 0.5 * alpha * (ev["disconnect_time"] - schedule["t_actual"]) ** 2

            sum_operator_obj += local_obj
            # The local schedule already reported "energy_cost" and "total_cost" from Gurobi, but
            # we'll sum up from the direct calculation for consistency
            sum_energy_costs += schedule["energy_cost"]

        self.results = {
            "operator_objective_vector": operator_objective_vector,
            "energy_cost_vector": energy_cost_vector,
            "sum_operator_objective": sum_operator_obj,
            "sum_energy_costs": sum_energy_costs,
            "soc_over_time": soc_over_time,
            "desired_disconnect_time": desired_disconnect_time,
            "actual_disconnect_time": actual_disconnect_time
        }
        return self.results
