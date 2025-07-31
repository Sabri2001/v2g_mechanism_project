from experiments.base_experiment import BaseExperiment
import numpy as np


class UncoordinatedChargingExperiment(BaseExperiment):
    def run(self):
        """
        An uncoordinated charging strategy that iterates over each time step and each EV
        to determine the charge delivered. The EVCS power limit is respected by clamping
        each EV's possible charge rate to evcs_power_limit / number_of_EVs.
        """
        # Extract configuration settings
        time_range = self.config["time_range"]  # e.g. [10, 22]
        start_time, end_time = time_range
        T = end_time - start_time  # Total number of time slots
        market_prices = self.config["market_prices"]  # Should have length T
        evs = self.config["evs"]  # List of EV dictionaries
        evcs_power_limit = self.config["evcs_power_limit"]  # Global EVCS power limit

        # Validate input lengths
        if len(market_prices) != T:
            raise ValueError("Length of market_prices must match the time range (T).")

        # Prepare results structure
        results = {
            "operator_cost_over_time": np.zeros(T).tolist(),
            "energy_cost_over_time": np.zeros(T).tolist(),  # Track only the energy component
            "sum_operator_costs": 0,
            "sum_energy_costs": 0,
            "soc_over_time": {
                ev["id"]: [ev["initial_soc"]] + [0] * T for ev in evs
            },
        }

        # Precompute the maximum allowable charge rate per EV from the EVCS
        # We assume all EVs *could* charge simultaneously, so we divide evenly.
        num_evs = len(evs)
        evcs_per_ev_rate = evcs_power_limit / num_evs  # kW or kWh per hour time step

        results["individual_cost"] = {}
        results["individual_payment"] = {}

        for ev in evs:
            ev_id = ev["id"]
            soc = ev["initial_soc"]
            desired_soc = ev["desired_soc"]
            max_rate = ev["max_charge_rate"]
            disconnection_time = ev["disconnection_time"]
            battery_wear = ev["battery_wear_cost_coefficient"]
            energy_efficiency = ev["energy_efficiency"]
            beta = ev["soc_flexibility"]

            # Initialize per-EV totals
            individual_cost = 0.0
            individual_payment = 0.0

            for t in range(T):
                actual_time = start_time + t

                if soc >= desired_soc or actual_time >= disconnection_time:
                    for t_remaining in range(t + 1, T + 1):
                        results["soc_over_time"][ev_id][t_remaining] = soc
                    break

                possible_charge = min(
                    desired_soc - soc,
                    max_rate,
                    evcs_per_ev_rate
                )

                usable_energy = possible_charge * energy_efficiency
                soc += usable_energy
                results["soc_over_time"][ev_id][t + 1] = soc

                energy_cost = market_prices[t] * possible_charge
                wear_cost = battery_wear * abs(usable_energy)

                individual_cost += wear_cost
                individual_payment += market_prices[t] * possible_charge

                results["operator_cost_over_time"][t] += energy_cost + wear_cost
                results["energy_cost_over_time"][t] += energy_cost

            else:
                for t_remaining in range(t + 1, T + 1):
                    results["soc_over_time"][ev_id][t_remaining] = soc

            # Final SoC after all time steps
            final_soc = results["soc_over_time"][ev_id][T]
            soc_deviation_cost = 0.5 * beta * (desired_soc - final_soc) ** 2
            individual_cost += soc_deviation_cost

            # Store individual results
            results["individual_cost"][ev_id] = individual_cost
            results["individual_payment"][ev_id] = individual_payment

        # Aggregate final costs
        results["sum_operator_costs"] = sum(results["operator_cost_over_time"])
        results["sum_energy_costs"] = sum(results["energy_cost_over_time"])

        # For consistency with other experiments, set these fields even if uncoordinated
        results["desired_disconnection_time"] = [ev["disconnection_time"] for ev in evs]
        results["actual_disconnection_time"] = [ev["disconnection_time"] for ev in evs]

        # No V2G in uncoordinated scenario, so set fraction to 0
        results["v2g_fraction"] = 0

        # Save final results
        self.results = results
        return self.results
