#!/bin/bash

# Define the base directory containing the simulation runs
RUNS_DIR="runs/rc_hex_triple"

# Ensure the base directory exists
if [ ! -d "$RUNS_DIR" ]; then
    echo "Error: Directory $RUNS_DIR does not exist. Are you in the right folder?"
    exit 1
fi

# Find the most recently modified subdirectory
# `ls -td` sorts by time (newest first), `head -n 1` grabs the top one
LATEST_BATCH=$(ls -td "$RUNS_DIR"/*/ 2>/dev/null | head -n 1)

# Check if a directory was actually found
if [ -z "$LATEST_BATCH" ]; then
    echo "Error: No batch directories found inside $RUNS_DIR."
    exit 1
fi

# Strip the trailing slash for a cleaner path string
LATEST_BATCH=${LATEST_BATCH%/}

echo "============================================================"
echo "  Targeting latest batch: $LATEST_BATCH"
echo "============================================================"

# Execute the analysis script on the found directory
python3 energy_calc.py --batch-dir "$LATEST_BATCH"
