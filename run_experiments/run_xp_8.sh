#!/bin/bash

# === USER PARAMETERS ===
BASE_CONFIG="../config/tsg/xp_8/admm.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run experiments
python ../main.py "$BASE_CONFIG"
    
# Log summary across all experiments
python ../log_experiments/log_xp_8.py
