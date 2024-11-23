from experiments.base_experiment import BaseExperiment
import numpy as np


class UncoordinatedChargingExperiment(BaseExperiment):
    def run(self):
        time_range = self.config["time_range"]  # [10, 22]
        start_time, end_time = time_range
        T = end_time - start_time  # Total number of time slots
        market_prices = self.config["market_prices"]  # Should have length T
        evs = self.config["evs"]  # Access the list of EVs

        results = {
            "operator_objective_vector": np.zeros(T).tolist(),
            "energy_cost_vector": np.zeros(T).tolist(),  # Separate energy cost
            "sum_operator_objective": 0,
            "sum_energy_costs": 0,  # Total energy costs
            "soc_over_time": {ev["id"]: np.zeros(T).tolist() for ev in evs}  # Track SoC for each EV over time
        }

        for ev in evs:
            # Extract EV-specific attributes to local variables
            soc = ev["initial_soc"]
            desired_soc = ev["desired_soc"]
            max_rate = ev["max_charge_rate"]
            disconnect_time = ev["disconnect_time"]
            beta = ev["battery_wear_cost_coefficient"]
            ev_id = ev["id"]  # Assume each EV has a unique ID

            for t in range(T):
                actual_time = t + start_time  # Adjust time to match the actual time in the time range

                # Check if the EV has reached its desired SOC or is disconnected
                if soc >= desired_soc or actual_time >= disconnect_time:
                    # Fill remaining time steps with the final SOC
                    for t_remaining in range(t, T):
                        results["soc_over_time"][ev_id][t_remaining] = soc
                    break

                # Determine power to charge
                power = min(max_rate, desired_soc - soc)
                soc += power

                # Record the SOC for this EV at time t
                results["soc_over_time"][ev_id][t] = soc

                # Calculate costs
                energy_cost = market_prices[t] * power
                wear_cost = beta * abs(power)  # Linear wear cost based on charging/discharging power

                # Update results
                results["operator_objective_vector"][t] += energy_cost + wear_cost
                results["energy_cost_vector"][t] += energy_cost

            else:
                # If the loop wasn't broken, fill remaining time steps with the final SOC
                for t_remaining in range(t, T):
                    results["soc_over_time"][ev_id][t_remaining] = soc

        # Sum over time for operator objective and energy costs
        results["sum_operator_objective"] = sum(results["operator_objective_vector"])
        results["sum_energy_costs"] = sum(results["energy_cost_vector"])

        self.results = results
        return self.results
