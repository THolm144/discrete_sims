#!/bin/bash
RUNS_DIR="runs/dsb1_rc_hex"

# Ensure the base directory exists
if [ ! -d "$RUNS_DIR" ]; then
    echo "Error: Directory $RUNS_DIR does not exist. Are you in the right folder?"
    exit 1
fi

# Find the most recently modified sub-directory inside RUNS_DIR, regardless of name
# This targets the absolute newest folder at the top level of RUNS_DIR based entirely on timestamp
LATEST_BATCH=$(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)

# Check if a directory was actually found
if [ -z "$LATEST_BATCH" ]; then
    echo "Error: No directories found inside $RUNS_DIR."
    exit 1
fi

# Strip the trailing slash if present for a cleaner path string
LATEST_BATCH=${LATEST_BATCH%/}

echo "============================================================"
echo "  Targeting latest batch: $LATEST_BATCH"
echo "  Reconstructing longitudinal profile via ToF kinematics..."
echo "============================================================"

# Execute the Time-of-Flight kinematic reconstruction script on the found directory
python3 tof_reconstruction.py --batch-dir "$LATEST_BATCH"