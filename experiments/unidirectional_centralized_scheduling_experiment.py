import logging
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import copy
import time

from experiments.base_experiment import BaseExperiment


class UnidirectionalCentralizedSchedulingExperiment(BaseExperiment):
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
        nb_time_steps = self.config["nb_time_steps"]
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
            M = battery_cap + max_charge * nb_time_steps

            # Create variables
            u = model.addVars(nb_time_steps, lb=0, ub=max_charge, name=f"u_{ev_id}")
            abs_u = model.addVars(nb_time_steps, lb=0, name=f"abs_u_{ev_id}")
            b = model.addVars(nb_time_steps, vtype=GRB.BINARY, name=f"b_{ev_id}")
            delta = model.addVars(nb_time_steps, vtype=GRB.BINARY, name=f"delta_{ev_id}")
            soc = model.addVars(nb_time_steps + 1, lb=0, ub=battery_cap, name=f"soc_{ev_id}")
            actual_disconnection_step = model.addVar(vtype=GRB.INTEGER, lb=start_step + 1, ub=end_step, name=f"t_actual_{ev_id}")
            
            EV_vars[ev_id] = {
                "u": u,
                "abs_u": abs_u,
                "b": b,
                "delta": delta,
                "soc": soc,
                "actual_disconnection_step": actual_disconnection_step,
                "M": M
            }
        if self.warmstart_solutions is not None:
            for ev in evs:
                ev_id = ev["id"]
                if ev_id in self.warmstart_solutions:
                    for step in range(nb_time_steps):
                        # Suppose your var for EV ev_id is: EV_vars[ev_id]["u"][step]
                        # Then set:
                        EV_vars[ev_id]["u"][step].start = self.warmstart_solutions[ev_id][step]

            # Then also set model.params.AdvBasis = 1 if Gurobi version requires it:
            model.update()
        
        # 3) Add global constraints: For each time period t, the abs of the sum over EVs of u must not exceed the EVCS power limit.
        global_constraints = {}
        global_abs = {}
        for step in range(nb_time_steps):
            global_abs[step] = model.addVar(lb=0, name=f"global_abs_{step}")
            expr = gp.quicksum(EV_vars[ev["id"]]["u"][step] for ev in evs)
            model.addConstr(global_abs[step] >= expr, name=f"global_abs_pos_{step}")
            model.addConstr(global_abs[step] >= -expr, name=f"global_abs_neg_{step}")
            global_constraints[step] = model.addConstr(global_abs[step] <= evcs_power_limit, name=f"global_constraint_{step}")

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
            
            # Initial SoC constraint
            model.addConstr(ev_vars["soc"][0] == initial_soc, name=f"InitialSoC_{ev_id}")
            
            # Exactly one disconnection step: sum_step b[step] == 1
            model.addConstr(gp.quicksum(ev_vars["b"][step] for step in range(nb_time_steps)) == 1,
                            name=f"OneDisconnection_{ev_id}")

            # Link actual_disconnection_step with disconnection indicator
            model.addConstr(
                ev_vars["actual_disconnection_step"] == gp.quicksum((start_step + step + 1) * ev_vars["b"][step] for step in range(nb_time_steps)),
                name=f"actual_disconnection_step_link_{ev_id}"
            )
            
            # SoC dynamics
            for step in range(nb_time_steps):
                model.addConstr(
                    ev_vars["soc"][step+1] == ev_vars["soc"][step] + ev_vars["u"][step] * dt * eff,
                    name=f"SoC_dynamics_{ev_id}_{step}"
                )
            
            # Absolute value constraints for u
            for step in range(nb_time_steps):
                model.addConstr(ev_vars["abs_u"][step] >= ev_vars["u"][step],
                                name=f"Abs_u_pos_{ev_id}_{step}")
                model.addConstr(ev_vars["abs_u"][step] >= -ev_vars["u"][step],
                                name=f"Abs_u_neg_{ev_id}_{step}")

            # Delta variables: define delta[step] as indicator for step < actual_disconnection_step
            for step in range(nb_time_steps):
                model.addConstr(
                    ev_vars["actual_disconnection_step"] - (step + start_step) >= 1 - M * (1 - ev_vars["delta"][step]),
                    name=f"Delta_def1_{ev_id}_{step}"
                )
                model.addConstr(
                    ev_vars["actual_disconnection_step"] - (step + start_step) <= M * ev_vars["delta"][step],
                    name=f"Delta_def2_{ev_id}_{step}"
                )
            
            # Charging/discharging limits depending on delta:
            for step in range(nb_time_steps):
                model.addConstr(
                    ev_vars["u"][step] <= max_charge * ev_vars["delta"][step],
                    name=f"Charge_limit_{ev_id}_{step}"
                )
                model.addConstr(
                    ev_vars["u"][step] >= -max_discharge * ev_vars["delta"][step],
                    name=f"Discharge_limit_delta_{ev_id}_{step}"
                )
            
            # Minimum SoC constraints for all time steps.
            for step in range(nb_time_steps + 1):
                model.addConstr(
                    ev_vars["soc"][step] >= min_soc,
                    name=f"MinSoC_{ev_id}_{step}"
                )
        
        # 5) Define the objective function.
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
            for step in range(nb_time_steps):
                cost_ev += market_prices[step//granularity] * ev_vars["u"][step] * dt \
                           + battery_wear * ev_vars["abs_u"][step] * dt * eff
            # Quadratic penalty on disconnection time deviation
            desired_step = desired_disc_time * granularity
            cost_ev += 0.5 * alpha * ((desired_step - ev_vars["actual_disconnection_step"]) * (desired_step - ev_vars["actual_disconnection_step"]))/granularity / granularity
            # Quadratic penalty on soc deviation
            delta_soc = model.addVar(lb=0, name=f"delta_soc_{ev_id}")
            model.addConstr(delta_soc >= desired_soc - ev_vars["soc"][nb_time_steps], name=f"delta_soc_constraint_{ev_id}")
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
        soc_over_time = {ev["id"]: [0.0] * (nb_time_steps + 1) for ev in evs}
        energy_cost_vector = [0.0] * nb_time_steps
        wear_cost_vector = [0.0] * nb_time_steps
        desired_disconnection_time = {}
        actual_disconnection_time = {}
        soc_cost_per_ev = {}
        delay_cost_per_ev = {}

        for ev in evs:
            ev_id = ev["id"]
            ev_vars = EV_vars[ev_id]

            actual_disconnection_step_val = int(round(ev_vars["actual_disconnection_step"].X))
            actual_disconnection_time[ev_id] = actual_disconnection_step_val / granularity  # hours
            desired_disconnection_time[ev_id] = ev["disconnection_time"]

            # Operator cost per EV
            soc_cost_per_ev[ev_id] = 0.5 * ev["soc_flexibility"] * (ev["desired_soc"] - ev_vars["soc"][nb_time_steps].X) ** 2
            delay_cost_per_ev[ev_id] = 0.5 * ev["disconnection_time_flexibility"] * \
                                    (ev["disconnection_time"] - actual_disconnection_time[ev_id]) ** 2

            # Store SoC evolution
            for step in range(nb_time_steps + 1):
                soc_over_time[ev_id][step] = ev_vars["soc"][step].X

            # Energy and wear costs
            for step in range(nb_time_steps):
                u_val = ev_vars["u"][step].X
                cost_energy = market_prices[step // granularity] * u_val * dt
                cost_wear = ev["battery_wear_cost_coefficient"] * abs(u_val) * ev["energy_efficiency"] * dt
                wear_cost_vector[step] += cost_wear
                energy_cost_vector[step] += cost_energy

        total_cost = sum(energy_cost_vector) + sum(wear_cost_vector) + \
                    sum(soc_cost_per_ev.values()) + sum(delay_cost_per_ev.values())
        total_energy_cost = sum(energy_cost_vector)

        
        # Build the main results dictionary.
        self.results = {
            "wear_cost_over_time": wear_cost_vector,
            "energy_cost_over_time": energy_cost_vector,
            "soc_cost_per_ev": soc_cost_per_ev,
            "delay_cost_per_ev": delay_cost_per_ev,
            "total_cost": total_cost,
            "total_energy_cost": total_energy_cost,
            "soc_over_time": soc_over_time,
            "desired_disconnection_time": desired_disconnection_time,
            "actual_disconnection_time": actual_disconnection_time,
            "solve_time": solve_time,
        }

        
        # 8) VCG tax computation if "vcg" flag is enabled.
        if self.config.get("vcg", False):
            # First, compute the individual cost incurred by each EV in the full run.
            individual_cost = {}
            for ev in evs:
                ev_id = ev["id"]
                cost_ev = 0.0
                for step in range(nb_time_steps):
                    u_val = EV_vars[ev_id]["u"][step].X
                    cost_ev += market_prices[step // granularity] * u_val * dt + \
                               ev["battery_wear_cost_coefficient"] * abs(u_val) * ev["energy_efficiency"] * dt
                cost_ev += 0.5 * ev["disconnection_time_flexibility"] * \
                           ((ev["disconnection_time"] - EV_vars[ev_id]["actual_disconnection_step"].X / granularity) ** 2)
                cost_ev += 0.5 * ev["soc_flexibility"] * \
                           (ev["desired_soc"] - EV_vars[ev_id]["soc"][nb_time_steps].X) ** 2
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
                remaining_evs = [other_ev for other_ev in evs if other_ev["id"] != ev_id]
                config_without_ev["evs"] = remaining_evs
                
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
