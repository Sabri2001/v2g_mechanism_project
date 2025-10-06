#!/bin/bash
# === USER PARAMETERS ===
TAU_VALUES=(17 18 19 20 21)
ALPHA_VALUES=(40 50 60 70 80)
TARGET_EV_ID=0
TRUE_TAU=19
TRUE_ALPHA=60
BASE_CONFIG="../config/tsg/xp_9/incentive_lie.yaml"
CONFIG_OUTPUT_DIR="../config/tsg/xp_9/generated"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PLOT_SCRIPT="../plot_experiments/plot_xp_9.py"

# Create directories
mkdir -p "$CONFIG_OUTPUT_DIR"

echo "Starting Incentive to lie experiment grid..."
echo "Target EV: ${TARGET_EV_ID}"
echo "TRUE values: tau=${TRUE_TAU}, alpha=${TRUE_ALPHA}"
echo "Testing bid values:"
echo " Tau: ${TAU_VALUES[@]}"
echo " Alpha: ${ALPHA_VALUES[@]}"
echo "Timestamp: ${TIMESTAMP}"

RESULTS_BASE_PATTERN="../outputs/tsg/xp_9"

# Loop over all bid combinations
for tau in "${TAU_VALUES[@]}"; do
  for alpha in "${ALPHA_VALUES[@]}"; do
    echo ""
    echo "================================================"
    echo "Testing BID: tau=${tau}, alpha=${alpha}"
    echo "================================================"
    
    MODIFIED_CONFIG="${CONFIG_OUTPUT_DIR}/incentive_to_lie_tau_${tau}_alpha_${alpha}.yaml"
    
    # Copy base config and modify parameters
    cp "$BASE_CONFIG" "$MODIFIED_CONFIG"
    
    # Update folder, name_xp, disconnection_time_bid, and disconnection_time_flexibility_bid (macOS compatible)
    sed -i '' \
      -e "s|^folder:.*|folder: \"tsg/xp_9\"|" \
      -e "s|^name_xp:.*|name_xp: \"tau_${tau}_alpha_${alpha}\"|" \
      -e "s|^disconnection_time_bid:.*|disconnection_time_bid: ${tau}|" \
      -e "s|^disconnection_time_flexibility_bid:.*|disconnection_time_flexibility_bid: ${alpha}|" \
      "$MODIFIED_CONFIG"
    
    echo "Running experiment with config: $MODIFIED_CONFIG"
    python ../main.py "$MODIFIED_CONFIG"
    
    if [ $? -ne 0 ]; then
      echo "ERROR: Experiment failed for tau=${tau}, alpha=${alpha}"
    else
      echo "SUCCESS: Completed tau=${tau}, alpha=${alpha}"
    fi
  done
done

echo ""
echo "================================================"
echo "All experiments completed!"
echo "================================================"
echo "Generating Incentive to lie grid plot..."

RESULTS_DIR=$(ls -td ${RESULTS_BASE_PATTERN}/tau_*_alpha_*_* 2>/dev/null | head -1)
if [ -z "$RESULTS_DIR" ]; then
  echo "ERROR: Could not find results directory"
  exit 1
fi

echo "Using results from: ${RESULTS_DIR}"
python "$PLOT_SCRIPT" "$RESULTS_DIR" "$TARGET_EV_ID" "$TRUE_TAU" "$TRUE_ALPHA" "${TAU_VALUES[@]}" "---" "${ALPHA_VALUES[@]}"

echo ""
echo "Done! Check results in: ${RESULTS_DIR}"
