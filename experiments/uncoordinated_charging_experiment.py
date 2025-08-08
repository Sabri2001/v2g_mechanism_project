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
        time_range = self.config["time_range"]
        T = self.config["T"]
        dt = self.config["dt"]
        granularity = self.config["granularity"]
        start_time, _ = time_range
        start_step = start_time * granularity
        market_prices = self.config["market_prices"]
        evs = self.config["evs"]
        evcs_power_limit = self.config["evcs_power_limit"]

        # Prepare results structure
        results = {
            "operator_cost_over_time": np.zeros(T).tolist(),
            "energy_cost_over_time": np.zeros(T).tolist(),
            "total_cost": 0,
            "total_energy_cost": 0,
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

        active_evs = {ev["id"]: ev for ev in evs}
        remaining_soc = {ev["id"]: ev["initial_soc"] for ev in evs}

        for step_idx in range(T):
            actual_step = start_step + step_idx

            # Identify EVs that are still connected and not fully charged
            charging_evs = [
                ev for ev_id, ev in active_evs.items()
                if remaining_soc[ev_id] < ev["desired_soc"]
                and actual_step < ev["disconnection_time"] * granularity
            ]

            num_active = len(charging_evs)
            if num_active == 0:
                continue  # No one to charge this step

            evcs_per_ev_rate = evcs_power_limit / num_active

            for ev in charging_evs:
                ev_id = ev["id"]
                soc = remaining_soc[ev_id]
                desired_soc = ev["desired_soc"]
                max_rate = ev["max_charge_rate"]
                battery_wear = ev["battery_wear_cost_coefficient"]
                energy_efficiency = ev["energy_efficiency"]

                possible_charge = min(
                    (desired_soc - soc) / energy_efficiency,
                    max_rate * dt,
                    evcs_per_ev_rate * dt
                )

                usable_energy = possible_charge * energy_efficiency
                soc += usable_energy
                remaining_soc[ev_id] = soc
                results["soc_over_time"][ev_id][step_idx + 1] = soc

                energy_cost = market_prices[step_idx // granularity] * possible_charge
                wear_cost = battery_wear * abs(usable_energy)

                results["operator_cost_over_time"][step_idx] += energy_cost + wear_cost
                results["energy_cost_over_time"][step_idx] += energy_cost

                results["individual_cost"].setdefault(ev_id, 0.0)
                results["individual_payment"].setdefault(ev_id, 0.0)
                results["individual_cost"][ev_id] += wear_cost + energy_cost
                results["individual_payment"][ev_id] += energy_cost

        # Final loop to handle remaining SoC and deviation cost
        for ev in evs:
            ev_id = ev["id"]
            soc = remaining_soc[ev_id]
            desired_soc = ev["desired_soc"]
            beta = ev["soc_flexibility"]

            for t_remaining in range(T + 1):
                if results["soc_over_time"][ev_id][t_remaining] == 0:
                    results["soc_over_time"][ev_id][t_remaining] = soc

            soc_deviation_cost = 0.5 * beta * (desired_soc - soc) ** 2
            results["individual_cost"][ev_id] += soc_deviation_cost

        # Compute total cost
        total_cost = sum(results["individual_cost"].values())

        # Aggregate final costs
        results["total_cost"] = total_cost
        results["total_energy_cost"] = sum(results["energy_cost_over_time"])

        # For consistency with other experiments, set these fields even if uncoordinated
        results["desired_disconnection_time"] = [ev["disconnection_time"] for ev in evs]
        results["actual_disconnection_time"] = [ev["disconnection_time"] for ev in evs]

        # No V2G in uncoordinated scenario, so set fraction to 0
        # results["v2g_fraction"] = 0

        # Save final results
        self.results = results
        return self.results
