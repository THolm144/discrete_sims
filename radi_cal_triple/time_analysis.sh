#!/bin/bash

# Define the base directory containing the simulation sweeps
RUNS_DIR="/home/uakgun/env/THOMAS/discrete_sims/radi_cal_triple/runs/radi_cal_triple"

# Ensure the base directory exists
if [ ! -d "$RUNS_DIR" ]; then
    echo "Error: Directory $RUNS_DIR does not exist."
    exit 1
fi

# Find the most recently modified sweep_* sub-directory inside RUNS_DIR
LATEST_BATCH=$(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -name "sweep_*" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)

# Check if a directory was actually found
if [ -z "$LATEST_BATCH" ]; then
    echo "Error: No sweep directories found inside $RUNS_DIR."
    exit 1
fi

# Strip the trailing slash if present for a cleaner path string
LATEST_BATCH=${LATEST_BATCH%/}

echo "============================================================"
echo "  Targeting latest sweep batch: $LATEST_BATCH"
echo "  Executing batch analysis pipeline..."
echo "============================================================"

# 1. Execute the time profile plotting script on the found directory
echo "-> Running plot_hits_vs_time.py..."
python3 plot_hits_vs_time.py --batch-dir "$LATEST_BATCH"

echo "------------------------------------------------------------"

# 2. Execute the timing resolution analysis script on the same directory
echo "-> Running timing_res.py..."
python3 timing_res.py --batch-dir "$LATEST_BATCH"

echo "============================================================"
echo "  Pipeline execution finished successfully."
echo "============================================================"