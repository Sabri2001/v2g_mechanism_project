import yaml
import itertools
import logging
import os
import tempfile

from main import main as run_main


# Define the path to your base configuration file
BASE_CONFIG_PATH = 'config/config.yaml'

# Define the lists for alpha_factor and price_factor
alpha_factors = [1, 0.7]
price_factors = [1, 2]


def main_grid_search():
    # Set up basic logging for this script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load the base config once
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # Create a grid of all combinations
    combinations = list(itertools.product(alpha_factors, price_factors))

    for idx, (alpha, price) in enumerate(combinations):
        name_xp = f"alpha_{alpha}_price_{price}"

        logging.info(f"\n=== Running Experiment: {name_xp} ===")
        logging.info(f"alpha_factor: {alpha}, price_factor: {price}")

        # Make a copy of the base config and update relevant fields
        updated_config = dict(base_config)  # shallow copy
        updated_config["alpha_factor"] = alpha
        updated_config["price_factor"] = price
        updated_config["name_xp"] = name_xp

        # Write to a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmpfile:
            yaml.safe_dump(updated_config, tmpfile)
            tmp_config_path = tmpfile.name

        try:
            # Run the experiment using the main function with the updated config
            run_main(tmp_config_path)
        finally:
            # Clean up temp config file
            if os.path.exists(tmp_config_path):
                os.remove(tmp_config_path)

        logging.info(f"=== Completed Experiment: {name_xp} ===\n")


if __name__ == "__main__":
    main_grid_search()
