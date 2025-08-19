#!/bin/bash

# === USER PARAMETERS ===
ALPHA_FACTORS=(0.1 0.25 1.0)
EVCS_POWER_LIMITS=(10 15 30)
BASE_CONFIG="../config/tsg/xp_4/cost_savings_congestion_line.yaml"
CONFIG_OUTPUT_DIR="../config/tsg/xp_4/generated"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PLOT_SCRIPT="../plot_experiments/plot_xp_4.py"

mkdir -p "$CONFIG_OUTPUT_DIR"

# Loop over all combinations
for alpha in "${ALPHA_FACTORS[@]}"; do
  for limit in "${EVCS_POWER_LIMITS[@]}"; do

    # Create unique folder and config name
    SUBFOLDER="cost_savings_limit_${limit}_af_${alpha}_${TIMESTAMP}"
    MODIFIED_CONFIG="${CONFIG_OUTPUT_DIR}/xp_alpha_${alpha}_limit_${limit}.yaml"

    echo "Preparing config for alpha=${alpha}, evcs_power_limit=${limit}"

    # Copy and modify config
    sed \
      -e "s|^folder:.*|folder: \"tsg/xp_4/${SUBFOLDER}\"|" \
      -e "s|^alpha_factor:.*|alpha_factor: ${alpha}|" \
      -e "s|^evcs_power_limit:.*|evcs_power_limit: ${limit}|" \
      "$BASE_CONFIG" > "$MODIFIED_CONFIG"

    # Run the experiment
    echo "Running experiment with config: $MODIFIED_CONFIG"
    python ../main.py "$MODIFIED_CONFIG"

  done
done

# Plotting results
python "$PLOT_SCRIPT"
