#!/bin/bash

# Config and output directories
CONFIG_FILE="../config/tsg/xp_10/negative_soc.yaml"

echo "Running experiment 10 with configuration: $CONFIG_FILE"
python ../main.py "$CONFIG_FILE"
