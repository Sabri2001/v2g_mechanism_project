#!/bin/bash

# === USER PARAMETERS ===
ALPHA_FACTORS=(0.1 0.25 1.0)
DISCONNECTION_TIMES=(16 17 18 19 20)
BASE_CONFIG="../config/tsg/xp_5/vcg.yaml"
CONFIG_OUTPUT_DIR="../config/tsg/xp_5/generated"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PLOT_SCRIPT="../plot_experiments/plot_xp_5.py"

mkdir -p "$CONFIG_OUTPUT_DIR"

# Loop over all combinations
for alpha in "${ALPHA_FACTORS[@]}"; do
  for time in "${DISCONNECTION_TIMES[@]}"; do

    # Create unique folder and config name
    SUBFOLDER="vcg_time_${time}_af_${alpha}_${TIMESTAMP}"
    MODIFIED_CONFIG="${CONFIG_OUTPUT_DIR}/xp_alpha_${alpha}_time_${time}.yaml"

    echo "Preparing config for alpha=${alpha}, disconnection_time=${time}"

    # Copy and modify config
    sed \
      -e "s|^folder:.*|folder: \"tsg/xp_5/${SUBFOLDER}\"|" \
      -e "s|^alpha_factor:.*|alpha_factor: ${alpha}|" \
      -e "s|^override_disconnection_time:.*|override_disconnection_time: ${time}|" \
      "$BASE_CONFIG" > "$MODIFIED_CONFIG"

    # Run the experiment
    echo "Running experiment with config: $MODIFIED_CONFIG"
    python ../main.py "$MODIFIED_CONFIG"

  done
done

# Plotting results
python "$PLOT_SCRIPT"
