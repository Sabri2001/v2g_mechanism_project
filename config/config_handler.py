import os
import yaml
import pickle
import random
import pandas as pd
import numpy as np


class ConfigHandler:
    """
    Handles configuration loading, validation, and sampling of data for experiments.
    """

    def __init__(self, config_path: str):
        """
        Initializes the ConfigHandler with only the path to the YAML config file.

        Args:
            config_path (str): Path to the YAML config file.
        """
        self.config_path = config_path

        # In the new approach, we do NOT accept alpha_factor, price_factor, or name_xp here;
        # they will be read from the config once the YAML is loaded.
        self.alpha_factor = 1
        self.price_factor = 1

        self.config = {}
        self.sampled_data_per_run = []
        self.num_runs = 1
        self.random_state = None

        # GMM and KDE models
        self.gmm_alpha_model = None
        self.gmm_disconnection_model = None
        self.initial_final_soc_kde_model = None

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def load_config(self):
        """
        Loads and validates the YAML configuration, initializes random seed,
        and prepares either stochastic or static data.
        """
        self._read_yaml()
        self._validate_config_fields()
        self._init_random_state()

        # Read alpha_factor, price_factor, and possibly override name_xp from the config
        self.alpha_factor = self.config.get("alpha_factor", 1)
        self.price_factor = self.config.get("price_factor", 1)

        if "granularity" not in self.config:
            self.config["granularity"] = 1

        # Load models
        self._load_alpha_model(self.config["alpha_model_path"])
        self._load_disconnection_model(self.config["disconnection_model_path"])
        self._load_soc_model(self.config["soc_model_path"])

        # Handle stochastic vs. non-stochastic
        if self.config["stochastic"]:
            self.num_runs = self.config["num_runs"]
            self._sample_data_per_run()
        else:
            self.num_runs = 1
            self._set_static_data()

        return self.config

    def get_sampled_data_per_run(self):
        """
        Returns:
            list: A list of dictionaries, each containing 'run_id', 'sampled_day_data', and 'evs'.
        """
        return self.sampled_data_per_run

    # -------------------------------------------------------------------------
    # Private Methods: Loading Models
    # -------------------------------------------------------------------------

    def _load_alpha_model(self, alpha_model_path):
        """Loads the GMM model for disconnection_time_flexibility."""
        if not os.path.exists(alpha_model_path):
            raise FileNotFoundError(f"GMM model not found at {alpha_model_path}")

        with open(alpha_model_path, "rb") as file:
            self.gmm_alpha_model = pickle.load(file)

    def _load_disconnection_model(self, disconnection_model_path):
        """Loads the GMM model for EV disconnect_time."""
        if not os.path.exists(disconnection_model_path):
            raise FileNotFoundError(f"GMM model not found at {disconnection_model_path}")

        with open(disconnection_model_path, "rb") as file:
            self.gmm_disconnection_model = pickle.load(file)

    def _load_soc_model(self, soc_model_path):
        """Loads the KDE model for (initial_soc, desired_soc) sampling."""
        if not os.path.exists(soc_model_path):
            raise FileNotFoundError(f"KDE model not found at {soc_model_path}")

        with open(soc_model_path, "rb") as file:
            self.initial_final_soc_kde_model = pickle.load(file)

    # -------------------------------------------------------------------------
    # Private Methods: Reading & Validating Config
    # -------------------------------------------------------------------------

    def _read_yaml(self):
        """Reads the config from the YAML file into self.config."""
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def _validate_config_fields(self):
        """Validates top-level fields and delegates further checks."""
        required_fields = ["name_xp", "experiment_types", "time_range", "evs", "stochastic"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

        # Check EVCS power limit
        self._check_evcs_power_limit()

        # Check the 'stochastic' boolean
        if not isinstance(self.config["stochastic"], bool):
            raise ValueError("stochastic must be a boolean value.")

        # Delegate validation based on stochastic or non-stochastic
        if self.config["stochastic"]:
            self._validate_stochastic_config()
        else:
            self._validate_non_stochastic_config()

        # Check EV list
        if not isinstance(self.config["evs"], list) or not self.config["evs"]:
            raise ValueError("evs must be a non-empty list of EV configurations.")

        # Validate each EV
        for ev in self.config["evs"]:
            self._validate_ev_fields(ev)

    def _check_evcs_power_limit(self):
        """Validates the EVCS power limit field."""
        evcs_limit = self.config.get("evcs_power_limit")
        if not isinstance(evcs_limit, (int, float)):
            raise ValueError("evcs_power_limit must be a number (int or float).")
        if evcs_limit <= 0:
            raise ValueError("evcs_power_limit must be a positive value.")

    def _validate_stochastic_config(self):
        """Validates fields required only when `stochastic=True`."""
        num_runs = self.config.get("num_runs")
        if not isinstance(num_runs, int) or num_runs <= 0:
            raise ValueError("When stochastic is true, num_runs must be a positive integer.")

        # alpha_factor & price_factor must be numeric
        if "alpha_factor" not in self.config or not isinstance(self.config["alpha_factor"], (int, float)):
            raise ValueError("alpha_factor must be provided as a numeric value when stochastic is true.")
        if "price_factor" not in self.config or not isinstance(self.config["price_factor"], (int, float)):
            raise ValueError("price_factor must be provided as a numeric value when stochastic is true.")

    def _validate_non_stochastic_config(self):
        """Validates fields required only when `stochastic=False`."""
        # market_prices must be provided
        if "market_prices" not in self.config or not isinstance(self.config["market_prices"], list):
            raise ValueError("When stochastic is false, market_prices must be provided as a list.")

    def _validate_ev_fields(self, ev):
        """Validates required EV fields based on whether config is stochastic or not."""
        # Basic required fields for all EVs
        required_ev_fields = [
            "id", "battery_capacity",
            "max_charge_rate", "max_discharge_rate", "min_soc",
            "battery_wear_cost_coefficient"
        ]
        # If not stochastic, these fields are also required
        if not self.config["stochastic"]:
            required_ev_fields += [
                "disconnection_time", "disconnection_time_flexibility",
                "initial_soc", "desired_soc"
            ]

        # Check presence
        for field in required_ev_fields:
            if field not in ev:
                ev_id = ev.get('id', 'unknown')
                raise ValueError(f"EV with id {ev_id} is missing required field: {field}")

        # Check types
        if not isinstance(ev["id"], int):
            raise ValueError("EV id must be an integer.")

        # Validate SoC if non-stochastic
        if not self.config["stochastic"]:
            if not (0 <= ev["initial_soc"] <= ev["battery_capacity"]):
                raise ValueError(f"EV with id {ev['id']} has an invalid initial SOC.")
            if not (0 <= ev["desired_soc"] <= ev["battery_capacity"]):
                raise ValueError(f"EV with id {ev['id']} has an invalid desired SOC.")

    def _init_random_state(self):
        """
        Initializes the numpy RandomState using config['seed'] if provided.
        """
        seed = self.config.get("seed", None)
        self.random_state = np.random.RandomState(seed)

    # -------------------------------------------------------------------------
    # Private Methods: Data Sampling
    # -------------------------------------------------------------------------

    def _sample_data_per_run(self):
        """Samples market prices and EV parameters for each run in stochastic mode."""
        for run_id in range(1, self.num_runs + 1):
            sampled_day_data = self._sample_market_prices()
            evs = self._sample_ev_parameters()
            self.sampled_data_per_run.append({
                "run_id": run_id,
                "sampled_day_data": sampled_day_data,
                "evs": evs
            })

    def _sample_market_prices(self):
        """
        Samples a row from the specified CSV, returning date + prices in $/kWh.
        """
        market_prices_csv = self.config["market_prices_csv"]
        time_range = self.config["time_range"]
        df = pd.read_csv(market_prices_csv)

        start_time, end_time = time_range
        # Format hours as two-digit strings (e.g., "01", "02", ...)
        time_columns = [f"{hour:02d}" for hour in range(start_time, end_time)]
        required_columns = ["Date"] + time_columns

        # Ensure CSV has the columns
        if not all(col in df.columns for col in required_columns):
            raise ValueError("Market prices CSV does not contain all required columns.")

        # Sample exactly one row
        filtered_df = df[required_columns]
        sampled_row = filtered_df.sample(n=1, random_state=self.random_state)
        day_date = sampled_row.iloc[0]["Date"]

        # Convert from $/MWh to $/kWh and apply price_factor
        prices = [
            (price / 1000) * self.price_factor
            for price in sampled_row.iloc[0][time_columns].tolist()
        ]

        return {"date": day_date, "prices": prices}

    def _sample_ev_parameters(self):
        """
        Samples EV parameters (disconnection_time_flexibility,
        disconnect_time, initial_soc, desired_soc) using the GMM/KDE models.
        """
        alpha_weights = self.gmm_alpha_model.weights_
        alpha_means = self.gmm_alpha_model.means_
        alpha_covs = self.gmm_alpha_model.covariances_

        disc_weights = self.gmm_disconnection_model.weights_
        disc_means = self.gmm_disconnection_model.means_
        disc_covs = self.gmm_disconnection_model.covariances_

        start_time, end_time = self.config["time_range"]
        sampled_evs = []

        for ev in self.config["evs"]:
            ev_copy = ev.copy()

            # 1) Sample disconnection_time_flexibility
            alpha_idx = self.random_state.choice(len(alpha_weights), p=alpha_weights)
            alpha_mean = alpha_means[alpha_idx][0]
            alpha_std = np.sqrt(alpha_covs[alpha_idx][0][0])  # 1D
            sampled_alpha = self.random_state.normal(loc=alpha_mean, scale=alpha_std)
            ev_copy["disconnection_time_flexibility"] = round(sampled_alpha * self.alpha_factor)

            # 2) Sample disconnect_time
            disc_idx = self.random_state.choice(len(disc_weights), p=disc_weights)
            disc_mean = disc_means[disc_idx][0]
            disc_std = np.sqrt(disc_covs[disc_idx][0][0])
            sampled_disconnect = int(round(self.random_state.normal(loc=disc_mean, scale=disc_std)))
            # Clamp to valid time range
            ev_copy["disconnect_time"] = max(start_time + 1, min(end_time, sampled_disconnect))

            # 3) Sample (initial_soc, desired_soc) and min_soc
            ev_copy = self._sample_soc(ev_copy)

            sampled_evs.append(ev_copy)

        return sampled_evs

    def _sample_soc(self, ev_copy):
        """
        Samples (initial_soc, desired_soc) via self.initial_final_soc_kde_model,
        ensuring desired_soc > initial_soc.
        """
        while True:
            # The KDE can return negative or >100 if tail is wide, so clamp results.
            pair = self.initial_final_soc_kde_model.resample(
                size=1,
                seed=self.random_state.randint(0, 2**31)
            )
            init_pct, desired_pct = pair[0][0], pair[1][0]

            battery_capacity = ev_copy["battery_capacity"]
            init_soc = round(np.clip((init_pct / 100) * battery_capacity, 0, battery_capacity))
            desired_soc = round(np.clip((desired_pct / 100) * battery_capacity, 0, battery_capacity))

            if desired_soc > init_soc:
                ev_copy["initial_soc"] = init_soc
                ev_copy["min_soc"] = init_soc
                ev_copy["desired_soc"] = desired_soc
                return ev_copy

    def _set_static_data(self):
        """
        If stochastic is False, we only have one run and reuse the config-specified data directly.
        """
        self.sampled_data_per_run.append({
            "run_id": 1,
            "sampled_day_data": {
                "date": None,
                "prices": [price * 0.001 for price in self.config["market_prices"]] # Convert from $/MWh to $/kWh
            },
            "evs": self.config["evs"]
        })
