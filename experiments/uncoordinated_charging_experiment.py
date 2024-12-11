from experiments.base_experiment import BaseExperiment
import numpy as np


class UncoordinatedChargingExperiment(BaseExperiment):
    def run(self):
        # Extract configuration settings
        time_range = self.config["time_range"]  # [10, 22]
        start_time, end_time = time_range
        T = end_time - start_time  # Total number of time slots
        market_prices = self.config["market_prices"]  # Should have length T
        evs = self.config["evs"]  # Access the list of EVs

        # Validate input lengths
        if len(market_prices) != T:
            raise ValueError("Length of market_prices must match the time range (T).")

        # Initialize results dictionary
        results = {
            "operator_objective_vector": np.zeros(T).tolist(),
            "energy_cost_vector": np.zeros(T).tolist(),  # Separate energy cost
            "sum_operator_objective": 0,
            "sum_energy_costs": 0,  # Total energy costs
            "soc_over_time": {ev["id"]: [ev["initial_soc"]] + [0] * T for ev in evs},  # Include initial SoC
        }

        # Simulate charging for each EV
        for ev in evs:
            # Extract EV-specific attributes to local variables
            soc = ev["initial_soc"]
            desired_soc = ev["desired_soc"]
            max_rate = ev["max_charge_rate"]
            disconnect_time = ev["disconnect_time"]
            beta = ev["battery_wear_cost_coefficient"]
            ev_id = ev["id"]
            energy_efficiency = ev["energy_efficiency"]

            for t in range(T):
                actual_time = t + start_time  # Adjust time to match the actual time in the time range

                # Check if the EV has reached its desired SOC or is disconnected
                if soc >= desired_soc or actual_time >= disconnect_time:
                    # Fill remaining time steps with the final SOC
                    for t_remaining in range(t + 1, T + 1):
                        results["soc_over_time"][ev_id][t_remaining] = soc
                    break

                # Determine energy to charge (delivered energy)
                energy = min(max_rate, desired_soc - soc)
                
                # Calculate usable energy (energy that contributes to SOC)
                usable_energy = energy * energy_efficiency
                soc += usable_energy  # Update SOC with usable energy

                # Record the SOC for this EV at time t
                results["soc_over_time"][ev_id][t + 1] = soc

                # Calculate costs
                energy_cost = market_prices[t] * energy  # Based on delivered energy
                wear_cost = beta * abs(usable_energy)  # Based on usable energy

                # Update results
                results["operator_objective_vector"][t] += energy_cost + wear_cost
                results["energy_cost_vector"][t] += energy_cost

            else:
                # If the loop wasn't broken, fill remaining time steps with the final SOC
                for t_remaining in range(t + 1, T + 1):
                    results["soc_over_time"][ev_id][t_remaining] = soc

        # Sum over time for operator objective and energy costs
        results["sum_operator_objective"] = sum(results["operator_objective_vector"])
        results["sum_energy_costs"] = sum(results["energy_cost_vector"])
        results["desired_disconnect_time"] = [ev["disconnect_time"] for ev in evs]
        results["actual_disconnect_time"] = [ev["disconnect_time"] for ev in evs]

        # Save and return results
        self.results = results
        return self.results
