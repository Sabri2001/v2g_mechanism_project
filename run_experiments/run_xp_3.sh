#!/bin/bash

# Config and output directories
CONFIG_FILE="../config/tsg/xp_3/congestion_soc.yaml"

echo "Running experiment 3 with configuration: $CONFIG_FILE"
python ../main.py "$CONFIG_FILE"
