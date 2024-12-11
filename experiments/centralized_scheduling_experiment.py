from experiments.base_experiment import BaseExperiment
from gurobipy import Model, GRB


class CentralizedSchedulingExperiment(BaseExperiment):
    def solve_ev_schedule(self, ev, market_prices, T, start_time, end_time):
        # Gurobi model for a single EV
        model = Model(f"EV_Scheduling_{ev['id']}")

        # Extract energy efficiency
        energy_efficiency = ev["energy_efficiency"]

        # Variables
        u = model.addVars(T, lb=-ev["max_discharge_rate"], ub=ev["max_charge_rate"], name="u")
        soc = model.addVars(T + 1, lb=0, ub=ev["battery_capacity"], name="soc")
        b = model.addVars(T, vtype=GRB.BINARY, name="b")  # Binary variables for disconnection time
        t_actual = model.addVar(vtype=GRB.INTEGER, lb=start_time + 1, ub=end_time, name="t_actual")
        abs_u = model.addVars(T, lb=0, name="abs_u")  # Auxiliary variables for |u|
        z = model.addVars(T, vtype=GRB.BINARY, name="z")  # Binary variables for SoC threshold
        delta = model.addVars(T, vtype=GRB.BINARY, name="delta")  # Binary variables indicating t < t_actual

        # Big-M constant
        M = ev["battery_capacity"] + ev["max_charge_rate"] * T  # Should be large enough

        # Coefficients
        beta = ev["battery_wear_cost_coefficient"]
        alpha = ev["disconnection_time_preference_coefficient"]

        # Objective: Minimize operator objective (energy cost + wear cost + disconnection cost)
        operator_objective = 0
        energy_cost = 0

        for t in range(T):
            # Energy cost
            energy_cost += market_prices[t] * u[t]
            # Operator objective includes wear cost, now scaled by energy_efficiency
            operator_objective += beta * abs_u[t] * energy_efficiency

        # Add disconnection time cost to operator objective
        operator_objective += 0.5 * alpha * (ev["disconnect_time"] - t_actual) ** 2

        # Full operator objective includes energy cost
        operator_objective += energy_cost

        model.setObjective(operator_objective, GRB.MINIMIZE)

        # Constraints
        # Initial SoC constraint
        model.addConstr(soc[0] == ev["initial_soc"], "InitialSoC")

        # Link t_actual with binary variables
        model.addConstr(t_actual == sum((t + start_time + 1) * b[t] for t in range(T)), "tActualLink")

        # Ensure only one disconnection time is selected
        model.addConstr(sum(b[t] for t in range(T)) == 1, "OneDisconnectionTime")

        # Final SoC constraint based on binary variables
        for t in range(T):
            model.addConstr(soc[t + 1] >= ev["desired_soc"] - (1 - b[t]) * M, f"FinalSoCPos_{t}")
            model.addConstr(soc[t + 1] <= ev["desired_soc"] + (1 - b[t]) * M, f"FinalSoCNeg_{t}")

        # SoC dynamics with energy_efficiency
        for t in range(T):
            model.addConstr(soc[t + 1] == soc[t] + u[t] * energy_efficiency, f"SoCDynamics_{t}")

        # Charging/discharging limits and ensure u[t] == 0 when t >= t_actual
        for t in range(T):
            # Ensure abs_u tracks absolute value of u
            model.addConstr(abs_u[t] >= u[t], f"AbsU_Pos_{t}")
            model.addConstr(abs_u[t] >= -u[t], f"AbsU_Neg_{t}")

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

        # Minimum SoC constraints
        for t in range(T + 1):
            model.addConstr(soc[t] >= ev["min_soc"], f"MinSoC_{t}")

        # Solve model
        model.optimize()

        # Extract results
        schedule = {
            "u": [u[t].X for t in range(T)],
            "soc": [soc[t].X for t in range(T + 1)],
            "t_actual": t_actual.X,
            "energy_cost": sum(market_prices[t] * u[t].X for t in range(T)),
            "total_cost": model.ObjVal,
        }
        return schedule

    def run(self):
        start_time, end_time = self.config["time_range"]
        T = end_time - start_time  # Total number of time slots
        market_prices = self.config["market_prices"]  # Should have length T
        evs = self.config["evs"]

        all_results = []
        operator_objective_vector = [0] * T
        energy_cost_vector = [0] * T
        soc_over_time = {ev["id"]: [0] * (T + 1) for ev in evs}  # Track SoC for each EV over time
        desired_disconnect_time = []
        actual_disconnect_time = []

        for ev in evs:
            ev_schedule = self.solve_ev_schedule(ev, market_prices, T, start_time, end_time)
            all_results.append(ev_schedule)

            desired_disconnect_time.append(ev["disconnect_time"])
            actual_disconnect_time.append(ev_schedule["t_actual"])

            # Combine costs for each time step
            for t in range(T):
                # Energy cost remains based on delivered energy
                energy_cost = ev_schedule["u"][t] * market_prices[t]
                # Wear cost now based on usable energy
                usable_energy = ev_schedule["u"][t] * ev["energy_efficiency"]
                wear_cost = ev["battery_wear_cost_coefficient"] * abs(usable_energy)

                operator_objective_vector[t] += energy_cost + wear_cost
                energy_cost_vector[t] += energy_cost

            # Track SoC for this EV
            for t in range(T + 1):
                soc_over_time[ev["id"]][t] = ev_schedule["soc"][t]

        # Summarize results
        self.results = {
            "operator_objective_vector": operator_objective_vector,  # Includes energy cost + wear cost
            "energy_cost_vector": energy_cost_vector,  # Energy cost only
            "sum_operator_objective": sum(operator_objective_vector),  # Total operator objective
            "sum_energy_costs": sum(energy_cost_vector),  # Total energy costs
            "soc_over_time": soc_over_time,  # SoC over time for each EV
            "desired_disconnect_time": desired_disconnect_time,
            "actual_disconnect_time": actual_disconnect_time,
        }
        return self.results
