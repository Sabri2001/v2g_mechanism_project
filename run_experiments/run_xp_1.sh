#!/bin/bash

# EV index to plot
EV_INDEX=0

# Config and output directories
CONFIG_DIR="../config/tsg/xp_1"
OUTPUT_DIR="../outputs/tsg/xp_1"
PLOT_SCRIPT="../plot_experiments/plot_xp_1.py"
LOG_LIST_FILE="../outputs/tsg/xp_1/log_files.txt"

# Clear previous log list
# > "$LOG_LIST_FILE"

# Run each config file
for config_path in "$CONFIG_DIR"/*.yaml; do
    config_name=$(basename "$config_path" .yaml)
    echo "Running experiment: $config_name"

    # Run the experiment
    python ../main.py "$config_path"

    # Find the most recent matching output directory for this config
    latest_log=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "${config_name}_*" | sort | tail -n 1)

    # Construct full path to the expected log.json
    log_path="${latest_log}/centralized/run_1/log.json"

    # If it exists, add it to the list
    if [ -f "$log_path" ]; then
        echo "$log_path" >> "$LOG_LIST_FILE"
    else
        echo "Warning: Log file not found for $config_name"
    fi
done

# Plot using the collected log files
python "$PLOT_SCRIPT" --log_list "$LOG_LIST_FILE" --ev_index "$EV_INDEX"
