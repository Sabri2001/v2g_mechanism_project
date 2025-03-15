import logging
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import copy
import time

from experiments.base_experiment import BaseExperiment


class CentralizedSchedulingExperiment(BaseExperiment):
    """
    Solves the EV scheduling problem with a full centralized Gurobi formulation.
    All EVs are optimized simultaneously subject to their individual constraints
    and the global EVCS power limit.
    """
    def __init__(self, config):
        super().__init__(config)
        self.warmstart_solutions = None

    def apply_warmstart(self, warmstart_dict):
        self.warmstart_solutions = warmstart_dict
    
    def run(self):
        # 1) Extract common data from config
        start_time, end_time = self.config["time_range"]
        T = self.config["T"]
        dt = self.config["dt"]
        granularity = self.config["granularity"]
        start_step = start_time * granularity
        end_step = end_time * granularity

        evs = self.config["evs"]
        market_prices = self.config["market_prices"]
        evcs_power_limit = self.config["evcs_power_limit"]
        
        # Create the full model and suppress solver output.
        model = gp.Model("CentralizedScheduling")
        model.setParam("OutputFlag", 0)
        
        # 2) Create decision variables for each EV
        # We will store each EV's variables in a dictionary keyed by its id.
        EV_vars = {}  # keys: EV id ; values: dict of variables for that EV
        for ev in evs:
            ev_id = ev["id"]
            max_charge = ev["max_charge_rate"]
            max_discharge = ev["max_discharge_rate"]
            battery_cap = ev["battery_capacity"]
            # Big-M for this EV – used in disconnection and SoC threshold constraints
            M = battery_cap + max_charge * T
            
            # Create variables
            u = model.addVars(T, lb=-max_discharge, ub=max_charge, name=f"u_{ev_id}")
            abs_u = model.addVars(T, lb=0, name=f"abs_u_{ev_id}")
            b = model.addVars(T, vtype=GRB.BINARY, name=f"b_{ev_id}")
            delta = model.addVars(T, vtype=GRB.BINARY, name=f"delta_{ev_id}")
            soc = model.addVars(T + 1, lb=0, ub=battery_cap, name=f"soc_{ev_id}")
            t_actual = model.addVar(vtype=GRB.INTEGER, lb=start_step + 1, ub=end_step, name=f"t_actual_{ev_id}")
            
            EV_vars[ev_id] = {
                "u": u,
                "abs_u": abs_u,
                "b": b,
                "delta": delta,
                "soc": soc,
                "t_actual": t_actual,
                "M": M
            }
        if self.warmstart_solutions is not None:
            for ev in evs:
                ev_id = ev["id"]
                if ev_id in self.warmstart_solutions:
                    for t in range(T):
                        # Suppose your var for EV ev_id is: EV_vars[ev_id]["u"][t]
                        # Then set:
                        EV_vars[ev_id]["u"][t].start = self.warmstart_solutions[ev_id][t]

            # Then also set model.params.AdvBasis = 1 if Gurobi version requires it:
            model.update()
        
        # 3) Add global constraints: For each time period t, the sum over EVs of abs_u must not exceed the EVCS power limit.
        global_constraints = {}
        global_abs = {}
        for t in range(T):
            global_abs[t] = model.addVar(lb=0, name=f"global_abs_{t}")
            expr = gp.quicksum(EV_vars[ev["id"]]["u"][t] for ev in evs)
            model.addConstr(global_abs[t] >= expr, name=f"global_abs_pos_{t}")
            model.addConstr(global_abs[t] >= -expr, name=f"global_abs_neg_{t}")
            global_constraints[t] = model.addConstr(global_abs[t] <= evcs_power_limit, name=f"global_constraint_{t}")

        # 4) Add individual EV constraints
        for ev in evs:
            ev_id = ev["id"]
            ev_vars = EV_vars[ev_id]
            M = ev_vars["M"]
            max_charge = ev["max_charge_rate"]
            max_discharge = ev["max_discharge_rate"]
            battery_cap = ev["battery_capacity"]
            initial_soc = ev["initial_soc"]
            min_soc = ev["min_soc"]
            eff = ev["energy_efficiency"]
            T_ev = T    # For clarity, T_ev is the number of u (time steps)
            
            # Initial SoC constraint
            model.addConstr(ev_vars["soc"][0] == initial_soc, name=f"InitialSoC_{ev_id}")
            
            # Exactly one disconnection time: sum_t b[t] == 1
            model.addConstr(gp.quicksum(ev_vars["b"][t] for t in range(T_ev)) == 1,
                            name=f"OneDisconnection_{ev_id}")
            
            # Link t_actual with disconnection indicator
            model.addConstr(
                ev_vars["t_actual"] == gp.quicksum((start_step + t + 1) * ev_vars["b"][t] for t in range(T_ev)),
                name=f"t_actual_link_{ev_id}"
            )
            
            # SoC dynamics
            for t in range(T_ev):
                model.addConstr(
                    ev_vars["soc"][t+1] == ev_vars["soc"][t] + ev_vars["u"][t] *dt * eff,
                    name=f"SoC_dynamics_{ev_id}_{t}"
                )
            
            # Absolute value constraints for u
            for t in range(T_ev):
                model.addConstr(ev_vars["abs_u"][t] >= ev_vars["u"][t],
                                name=f"Abs_u_pos_{ev_id}_{t}")
                model.addConstr(ev_vars["abs_u"][t] >= -ev_vars["u"][t],
                                name=f"Abs_u_neg_{ev_id}_{t}")
            
            # Delta variables: define delta[t] as indicator for t < t_actual
            for t in range(T_ev):
                model.addConstr(
                    ev_vars["t_actual"] - (t + start_step) >= 1 - M * (1 - ev_vars["delta"][t]),
                    name=f"Delta_def1_{ev_id}_{t}"
                )
                model.addConstr(
                    ev_vars["t_actual"] - (t + start_step) <= M * ev_vars["delta"][t],
                    name=f"Delta_def2_{ev_id}_{t}"
                )
            
            # Charging/discharging limits depending on delta:
            for t in range(T_ev):
                model.addConstr(
                    ev_vars["u"][t] <= max_charge * ev_vars["delta"][t],
                    name=f"Charge_limit_{ev_id}_{t}"
                )
                model.addConstr(
                    ev_vars["u"][t] >= -max_discharge * ev_vars["delta"][t],
                    name=f"Discharge_limit_delta_{ev_id}_{t}"
                )
            
            # Minimum SoC constraints for all time steps.
            for t in range(T_ev + 1):
                model.addConstr(
                    ev_vars["soc"][t] >= min_soc,
                    name=f"MinSoC_{ev_id}_{t}"
                )
        
        # 5) Define the objective function.
        # For each EV, we sum over time:
        #   energy cost: market_prices[t]*u[t] 
        #   battery wear: battery_wear_cost_coefficient * abs_u[t] * energy_efficiency
        # Plus the quadratic penalty on disconnection time and soc deviation.
        total_cost = 0
        for ev in evs:
            ev_id = ev["id"]
            ev_vars = EV_vars[ev_id]
            battery_wear = ev["battery_wear_cost_coefficient"]
            alpha = ev["disconnection_time_flexibility"]
            beta = ev["soc_flexibility"]
            desired_disc_time = ev["disconnection_time"]
            desired_soc = ev["desired_soc"]
            eff = ev["energy_efficiency"]
            cost_ev = 0
            for t in range(T):
                cost_ev += market_prices[t//granularity] * ev_vars["u"][t] * dt \
                           + battery_wear * ev_vars["abs_u"][t] * dt * eff
            # Quadratic penalty on disconnection time deviation
            desired_step = desired_disc_time * granularity
            cost_ev += 0.5 * alpha * ((desired_step - ev_vars["t_actual"]) * (desired_step - ev_vars["t_actual"]))/granularity / granularity
            # Quadratic penalty on soc deviation
            delta_soc = model.addVar(lb=0, name=f"delta_soc_{ev_id}")
            model.addConstr(delta_soc >= desired_soc - ev_vars["soc"][T], name=f"delta_soc_constraint_{ev_id}")
            cost_ev += 0.5 * beta * delta_soc * delta_soc
            total_cost += cost_ev
        
        model.setObjective(total_cost, GRB.MINIMIZE)
        
        # 6) Solve the centralized model.
        start_t = time.time()
        model.optimize()
        solve_time = time.time() - start_t
        if model.status != GRB.OPTIMAL:
            logging.warning("Centralized model did not solve to optimality.")
        else:
            logging.info(f"Centralized scheduling took {solve_time:.2f} seconds.")
        
        # 7) Extract and aggregate results.
        # Prepare time‐indexed cost vectors and SoC evolution per EV.
        soc_over_time = {ev["id"]: [0.0] * (T + 1) for ev in evs}
        energy_cost_vector = [0.0] * T
        operator_cost_vector = [0.0] * T
        desired_disconnection_time = []
        actual_disconnection_time = []
        total_energy_needed = 0.0
        total_energy_transferred = 0.0
        
        for ev in evs:
            ev_id = ev["id"]
            ev_vars = EV_vars[ev_id]
            t_actual_val = int(round(ev_vars["t_actual"].X))
            actual_disconnection_time.append(t_actual_val/granularity)
            desired_disconnection_time.append(ev["disconnection_time"])
            
            for t in range(T + 1):
                soc_over_time[ev_id][t] = ev_vars["soc"][t].X
            
            needed = ev["desired_soc"] - ev["initial_soc"]
            if needed > 0:
                total_energy_needed += needed
            
            for t in range(T):
                u_val = ev_vars["u"][t].X
                cost_energy = market_prices[t//granularity] * u_val
                cost_wear = ev["battery_wear_cost_coefficient"] * abs(u_val) * ev["energy_efficiency"]
                operator_cost_vector[t] += cost_energy + cost_wear
                energy_cost_vector[t] += cost_energy
                if u_val < 0:
                    total_energy_transferred += -u_val
        
        v2g_fraction = (total_energy_transferred / total_energy_needed) * 100 if total_energy_needed > 0 else 0.0
        
        # Build the main results dictionary.
        self.results = {
            "operator_cost_over_time": operator_cost_vector,
            "energy_cost_over_time": energy_cost_vector,
            "sum_operator_costs": sum(operator_cost_vector),
            "sum_energy_costs": sum(energy_cost_vector),
            "soc_over_time": soc_over_time,
            "desired_disconnection_time": desired_disconnection_time,
            "actual_disconnection_time": actual_disconnection_time,
            "v2g_fraction": v2g_fraction,
            "solve_time": solve_time,
        }
        
        # 8) Additional metrics if "walras_tax" is enabled.
        if self.config.get("walras_tax", False):
            # Attempt to extract dual values from the global constraints.
            duals = []
            for t in range(T):
                try:
                    dual_val = global_constraints[t].Pi
                except AttributeError:
                    dual_val = 0.0
                duals.append(dual_val)
            
            energy_cost_dict = {}
            adaptability_cost_dict = {}
            congestion_cost_dict = {}
            for ev in evs:
                ev_id = ev["id"]
                # Energy cost for EV: sum_t market_prices[t]*u[t]
                energy_cost_ev = sum(market_prices[t//granularity] * EV_vars[ev_id]["u"][t].X for t in range(T))
                energy_cost_dict[ev_id] = energy_cost_ev
                # Adaptability cost: quadratic penalty on disconnect time deviation
                adaptability_cost_ev = 0.5 * ev["disconnection_time_flexibility"] * \
                    ((ev["disconnection_time"] - EV_vars[ev_id]["t_actual"].X) ** 2)
                adaptability_cost_dict[ev_id] = adaptability_cost_ev
                # Congestion cost: sum_t (dual[t] * u[t])
                congestion_cost_ev = sum(du * EV_vars[ev_id]["u"][t].X for t, du in enumerate(duals))
                congestion_cost_dict[ev_id] = congestion_cost_ev
            
            self.results["energy_cost"] = energy_cost_dict
            self.results["adaptability_cost"] = adaptability_cost_dict
            self.results["congestion_cost"] = congestion_cost_dict
        
        # 9) VCG tax computation if "vcg" flag is enabled.
        if self.config.get("vcg", False):
            # First, compute the individual cost incurred by each EV in the full run.
            individual_cost = {}
            for ev in evs:
                ev_id = ev["id"]
                cost_ev = 0.0
                for t in range(T):
                    u_val = EV_vars[ev_id]["u"][t].X
                    cost_ev += market_prices[t//granularity] * u_val + \
                               ev["battery_wear_cost_coefficient"] * abs(u_val) * ev["energy_efficiency"]
                cost_ev += 0.5 * ev["disconnection_time_flexibility"] * \
                           ((ev["disconnection_time"] - EV_vars[ev_id]["t_actual"].X) ** 2)
                individual_cost[ev_id] = cost_ev
            self.results["individual_cost"] = individual_cost
            
            vcg_tax_dict = {}
            for ev in evs:
                ev_id = ev["id"]
                # Sum of the costs of all other EVs in the full run.
                original_others_cost = sum(individual_cost[other_ev["id"]]
                                           for other_ev in evs if other_ev["id"] != ev_id)
                # Build a copy of the config without EV ev.
                config_without_ev = copy.deepcopy(self.config)
                config_without_ev["evs"] = [other_ev for other_ev in evs if other_ev["id"] != ev_id]
                
                # Re-run the centralized experiment for the remaining EVs.
                from experiments.centralized_scheduling_experiment import CentralizedSchedulingExperiment
                experiment_without_ev = CentralizedSchedulingExperiment(config_without_ev)
                results_without_ev = experiment_without_ev.run()
                individual_cost_without_ev = results_without_ev.get("individual_cost", {})
                new_others_cost = sum(individual_cost_without_ev.values())
                
                # The VCG tax for ev is the increase in others' cost due to its presence.
                vcg_tax_dict[ev_id] = original_others_cost - new_others_cost
            
            self.results["vcg_tax"] = vcg_tax_dict
        
        return self.results
