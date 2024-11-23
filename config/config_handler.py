import yaml


class ConfigHandler:
    def __init__(self, config_path):
        self.config_path = config_path

    def load_config(self):
        with open(self.config_path, "r") as file:
            self.config = yaml.safe_load(file)
        self.validate_config()
        return self.config

    def validate_config(self):
        # Required fields in the root of the config
        required_fields = ["experiment_type", "time_range", "market_prices", "evs"]
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Missing required config field: {field}")

        # Validate time range
        if not (isinstance(self.config["time_range"], list) and len(self.config["time_range"]) == 2):
            raise ValueError("time_range must be a list with two elements (start and end time).")

        if not (0 <= self.config["time_range"][0] < self.config["time_range"][1] <= 24):
            raise ValueError("time_range must contain valid hours between 0 and 24.")

        # Validate market prices
        time_range_length = self.config["time_range"][1] - self.config["time_range"][0]
        if not (isinstance(self.config["market_prices"], list) and len(self.config["market_prices"]) == time_range_length):
            raise ValueError("market_prices must be a list with a length matching the time range.")

        # Validate EVs
        if not isinstance(self.config["evs"], list):
            raise ValueError("evs must be a list of EV configurations.")

        for ev in self.config["evs"]:
            self.validate_ev(ev, self.config["time_range"])

        # Optional plots
        if "plots" in self.config:
            valid_plots = {"objective_vs_cost", "soc_evolution"}
            if not all(plot in valid_plots for plot in self.config["plots"]):
                raise ValueError(f"Invalid plot type. Supported plots are: {valid_plots}")

    def validate_ev(self, ev, time_range):
        # Required EV attributes
        required_ev_fields = [
            "id", "disconnect_time", "battery_capacity", "initial_soc", "desired_soc",
            "max_charge_rate", "max_discharge_rate", "min_soc",
            "battery_wear_cost_coefficient", "disconnection_time_preference_coefficient"
        ]
        for field in required_ev_fields:
            if field not in ev:
                raise ValueError(f"EV with id {ev.get('id', 'unknown')} is missing required field: {field}")

        # Type and range validation for EV fields
        if not isinstance(ev["id"], int):
            raise ValueError("EV id must be an integer.")
        if not (time_range[0] <= ev["disconnect_time"] <= time_range[1]):
            raise ValueError(f"EV with id {ev['id']} has a disconnect_time out of the time_range.")
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
