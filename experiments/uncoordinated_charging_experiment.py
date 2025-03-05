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

        # Iterate over each EV
        for ev in evs:
            ev_id = ev["id"]
            soc = ev["initial_soc"]
            desired_soc = ev["desired_soc"]
            max_rate = ev["max_charge_rate"]
            disconnection_time = ev["disconnection_time"]
            beta = ev["battery_wear_cost_coefficient"]
            energy_efficiency = ev["energy_efficiency"]

            # For each time slot
            for t in range(T):
                actual_time = start_time + t

                # If EV has reached desired SoC or is past its disconnect time,
                # fill the remaining slots with the final SoC and break.
                if soc >= desired_soc or actual_time >= disconnection_time:
                    for t_remaining in range(t + 1, T + 1):
                        results["soc_over_time"][ev_id][t_remaining] = soc
                    break

                # Calculate how much we *could* deliver based on:
                #   1) The gap to desired_soc,
                #   2) The EV's max_charge_rate,
                #   3) The EVCS share per vehicle.
                possible_charge = min(
                    desired_soc - soc,        # Gap to desired SoC
                    max_rate,                 # EV's max charge rate
                    evcs_per_ev_rate          # EVCS-limited rate for this EV
                )

                # Calculate usable energy (accounting for efficiency)
                usable_energy = possible_charge * energy_efficiency
                soc += usable_energy

                # Record SoC in results
                results["soc_over_time"][ev_id][t + 1] = soc

                # Calculate costs
                # We pay market price on the delivered energy (not the "usable" portion).
                # Because "delivered energy" is possible_charge * 1 hour (kWh).
                # Wear cost depends on the actual energy that goes into the battery (usable_energy).
                energy_cost = market_prices[t] * possible_charge
                wear_cost = beta * abs(usable_energy)  # Could be more sophisticated

                # Update result vectors
                results["operator_cost_over_time"][t] += (energy_cost + wear_cost)
                results["energy_cost_over_time"][t] += energy_cost

            else:
                # If the for-loop completes without a break, we fill the remainder with current SoC
                for t_remaining in range(t + 1, T + 1):
                    results["soc_over_time"][ev_id][t_remaining] = soc

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
