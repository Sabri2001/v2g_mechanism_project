import yaml
import os
import pandas as pd
import random
import pickle
import numpy as np


class ConfigHandler:
    def __init__(self, config_path):
        self.config_path = config_path
        self.sampled_data_per_run = []  # List to store sampled data per run_id
        self.gmm_alpha_model = None
        self.gmm_disconnection_model = None
        self.random_state = None
        self.load_gmm_models()

    def load_gmm_models(self):
        """
        Load the GMM models for stochastic sampling of EV parameters.
        """
        # Load GMM model for disconnection_time_preference_coefficient
        alpha_model_path = "models/gmm_alpha.pkl"
        if not os.path.exists(alpha_model_path):
            raise FileNotFoundError(f"GMM model not found at {alpha_model_path}")
        with open(alpha_model_path, "rb") as file:
            self.gmm_alpha_model = pickle.load(file)

        # Load GMM model for disconnect_time
        disconnect_model_path = "models/gmm_disconnection_model.pkl"
        if not os.path.exists(disconnect_model_path):
            raise FileNotFoundError(f"GMM model not found at {disconnect_model_path}")
        with open(disconnect_model_path, "rb") as file:
            self.gmm_disconnection_model = pickle.load(file)

    def load_config(self):
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.validate_config()

        # Initialize random state using the seed
        seed = self.config.get("seed", None)
        self.random_state = np.random.RandomState(seed)

        if self.config["stochastic"]:
            self.num_runs = self.config["num_runs"]
            self.sample_data_per_run()
        else:
            self.num_runs = 1
            self.set_static_data()
        return self.config

    def validate_config(self):
        required_fields = ["name_xp", "experiment_types", "time_range", "evs", "stochastic"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

        # Validate stochastic
        if not isinstance(self.config["stochastic"], bool):
            raise ValueError("stochastic must be a boolean value.")

        if self.config["stochastic"]:
            if "num_runs" not in self.config or not isinstance(self.config["num_runs"], int):
                raise ValueError("num_runs must be provided as an integer when stochastic is true.")
            if self.config["num_runs"] <= 0:
                raise ValueError("num_runs must be a positive integer when stochastic is true.")
        else:
            if "market_prices" not in self.config or not isinstance(self.config["market_prices"], list):
                raise ValueError("When stochastic is false, market_prices must be provided as a list.")

        # Validate EVs
        if not isinstance(self.config["evs"], list) or not self.config["evs"]:
            raise ValueError("evs must be a non-empty list of EV configurations.")

        for ev in self.config["evs"]:
            self.validate_ev(ev, self.config["time_range"], self.config["stochastic"])

    def validate_ev(self, ev, time_range, stochastic):
        # Required EV attributes
        required_ev_fields = [
            "id", "battery_capacity", "initial_soc", "desired_soc",
            "max_charge_rate", "max_discharge_rate", "min_soc",
            "battery_wear_cost_coefficient"
        ]
        # 'disconnect_time' and 'disconnection_time_preference_coefficient' are optional if stochastic is True
        if not stochastic:
            required_ev_fields.extend(["disconnect_time", "disconnection_time_preference_coefficient"])

        for field in required_ev_fields:
            if field not in ev:
                raise ValueError(f"EV with id {ev.get('id', 'unknown')} is missing required field: {field}")

        # Type and range validation for EV fields
        if not isinstance(ev["id"], int):
            raise ValueError("EV id must be an integer.")

        if not (0 <= ev["initial_soc"] <= ev["battery_capacity"]):
            raise ValueError(f"EV with id {ev['id']} has an invalid initial state of charge.")
        if not (0 <= ev["desired_soc"] <= ev["battery_capacity"]):
            raise ValueError(f"EV with id {ev['id']} has an invalid desired state of charge.")
        if ev["min_soc"] < 0 or ev["min_soc"] > ev["battery_capacity"]:
            raise ValueError(f"EV with id {ev['id']} has an invalid min_soc value.")
        if ev["max_charge_rate"] <= 0:
            raise ValueError(f"EV with id {ev['id']} has an invalid max_charge_rate.")
        if ev["max_discharge_rate"] <= 0:
            raise ValueError(f"EV with id {ev['id']} has an invalid max_discharge_rate.")

        if not stochastic:
            # Validate disconnect_time
            if not (time_range[0] + 1 <= ev["disconnect_time"] <= time_range[1]):
                raise ValueError(f"EV with id {ev['id']} has a disconnect_time out of the time_range.")

    def sample_data_per_run(self):
        """
        Sample market prices and EV parameters for each run and store them.
        """
        for run_id in range(1, self.num_runs + 1):
            sampled_day_data = self.sample_market_prices()
            evs = self.sample_ev_parameters()
            self.sampled_data_per_run.append({
                "run_id": run_id,
                "sampled_day_data": sampled_day_data,
                "evs": evs
            })

    def sample_market_prices(self):
        """
        Samples market prices for a single run and converts prices from $/MWh to $/kWh.
        """
        market_prices_csv = self.config["market_prices_csv"]
        time_range = self.config["time_range"]

        df = pd.read_csv(market_prices_csv)
        start_time, end_time = time_range
        time_columns = [str(hour) for hour in range(start_time, end_time)]

        required_columns = ["Date"] + time_columns
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Market prices CSV does not contain all required columns.")

        filtered_df = df[required_columns]

        # Sample a row using the shared random state
        sampled_row = filtered_df.sample(n=1, random_state=self.random_state)
        day_date = sampled_row.iloc[0]["Date"]

        # Convert prices from $/MWh to $/kWh
        prices = [price / 1000 for price in sampled_row.iloc[0][time_columns].tolist()]
        
        return {"date": day_date, "prices": prices}

    def sample_ev_parameters(self):
        """
        Sample 'disconnection_time_preference_coefficient' and 'disconnect_time' for each EV using the GMM models.
        Returns a list of EV configurations with sampled parameters.
        """
        alpha_weights = self.gmm_alpha_model.weights_
        alpha_means = self.gmm_alpha_model.means_
        alpha_covariances = self.gmm_alpha_model.covariances_

        disconnect_weights = self.gmm_disconnection_model.weights_
        disconnect_means = self.gmm_disconnection_model.means_
        disconnect_covariances = self.gmm_disconnection_model.covariances_

        start_time, end_time = self.config["time_range"]

        sampled_evs = []
        for ev in self.config["evs"]:
            # Copy the EV config to avoid modifying the original
            ev_copy = ev.copy()

            # Sample 'disconnection_time_preference_coefficient'
            component_alpha = self.random_state.choice(len(alpha_weights), p=alpha_weights)
            mean_alpha = alpha_means[component_alpha][0]
            covariance_alpha = alpha_covariances[component_alpha][0][0]  # Assuming 1D Gaussians
            sampled_alpha = self.random_state.normal(loc=mean_alpha, scale=np.sqrt(covariance_alpha))
            ev_copy["disconnection_time_preference_coefficient"] = round(sampled_alpha)

            # Sample 'disconnect_time'
            component_disconnect = self.random_state.choice(len(disconnect_weights), p=disconnect_weights)
            mean_disconnect = disconnect_means[component_disconnect][0]
            covariance_disconnect = disconnect_covariances[component_disconnect][0][0]  # Assuming 1D Gaussians
            sampled_disconnect_time = self.random_state.normal(loc=mean_disconnect, scale=np.sqrt(covariance_disconnect))
            sampled_disconnect_time = round(sampled_disconnect_time)

            # Round to the nearest integer hour and ensure it's within the time range
            sampled_disconnect_time = int(round(sampled_disconnect_time))
            # Adjust to ensure it's within the valid time range
            if sampled_disconnect_time < start_time + 1:
                sampled_disconnect_time = start_time + 1
            elif sampled_disconnect_time > end_time:
                sampled_disconnect_time = end_time

            ev_copy["disconnect_time"] = sampled_disconnect_time

            # Append the sampled EV configuration
            sampled_evs.append(ev_copy)

        return sampled_evs

    def set_static_data(self):
        """
        Set static market prices and EV parameters for non-stochastic simulations.
        """
        self.sampled_data_per_run.append({
            "run_id": 1,
            "sampled_day_data": {"date": None, "prices": self.config["market_prices"]},
            "evs": self.config["evs"]
        })

    def get_sampled_data_per_run(self):
        """
        Returns the sampled data per run for use in experiments.
        """
        return self.sampled_data_per_run
