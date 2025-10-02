#!/bin/bash

# === USER PARAMETERS ===
BASE_CONFIG="../config/tsg/xp_7/value_elicitation.yaml"
CONFIG_OUTPUT_DIR="../config/tsg/xp_7/generated"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$CONFIG_OUTPUT_DIR"

# List of alpha model paths
ALPHA_MODELS=(
  "../models/gmm_alpha.pkl"
  "../models/gmm_alpha_bimodal.pkl"
)

# List of corresponding alpha modes
ALPHA_MODE=(
  unimodal
  bimodal
)

# Iterate over indices
for i in "${!ALPHA_MODELS[@]}"; do
  MODEL_PATH="${ALPHA_MODELS[$i]}"
  ALPHA_MODE="${ALPHA_MODE[$i]}"

  MODEL_NAME=$(basename "$MODEL_PATH" .pkl)
  MODIFIED_CONFIG="${CONFIG_OUTPUT_DIR}/xp_${MODEL_NAME}_${TIMESTAMP}.yaml"

  echo "Preparing config with alpha_model_path=${MODEL_PATH}, alpha_mode=${ALPHA_MODE}"

  # Copy and modify config
  sed \
    -e "s|^alpha_model_path:.*|alpha_model_path: ${MODEL_PATH}|" \
    -e "s|^alpha_mode:.*|alpha_mode: ${ALPHA_MODE}|" \
    "$BASE_CONFIG" > "$MODIFIED_CONFIG"

  # Add value_elicitation flag
  echo "value_elicitation: true" >> "$MODIFIED_CONFIG"

  # Modify name_xp to include mode alpha
  sed -i "" "s|^name_xp:.*|name_xp: value_elicitation_alpha_mode_${ALPHA_MODE}|" "$MODIFIED_CONFIG"

  # Run the experiment
  echo "Running experiment with config: $MODIFIED_CONFIG"
  python ../main.py "$MODIFIED_CONFIG"
done

# Log summary across all experiments
python ../log_experiments/log_xp_7.py ../outputs/tsg/xp_7/
