#!/bin/bash

# === USER PARAMETERS ===
ALPHA_FACTORS=(0.1 1.0)
BATTERY_FACTORS=(0.1 0.25 0.5 1.0)
BASE_CONFIG="config/tsg/xp_2/cost_savings_line.yaml"
CONFIG_OUTPUT_DIR="config/tsg/xp_2/generated"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$CONFIG_OUTPUT_DIR"

# Loop over all combinations
for alpha in "${ALPHA_FACTORS[@]}"; do
  for battery in "${BATTERY_FACTORS[@]}"; do

    # Create unique folder and config name
    SUBFOLDER="cost_savings_soc_bw_${battery}_af_${alpha}_${TIMESTAMP}"
    MODIFIED_CONFIG="${CONFIG_OUTPUT_DIR}/xp_alpha_${alpha}_battery_${battery}.yaml"

    echo "Preparing config for alpha=${alpha}, battery=${battery}"

    # Copy and modify config
    sed \
      -e "s|^folder:.*|folder: \"tsg/xp_2/${SUBFOLDER}\"|" \
      -e "s|^alpha_factor:.*|alpha_factor: ${alpha}|" \
      -e "s|^battery_factor:.*|battery_factor: ${battery}|" \
      "$BASE_CONFIG" > "$MODIFIED_CONFIG"

    # Run the experiment
    echo "Running experiment with config: $MODIFIED_CONFIG"
    python main.py "$MODIFIED_CONFIG"

  done
done
