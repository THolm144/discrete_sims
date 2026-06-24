#!/bin/bash
# Define the base directory containing the simulation runs
RUNS_DIR="runs/radi_cal_energy"

# Ensure the base directory exists
if [ ! -d "$RUNS_DIR" ]; then
echo "Error: Directory $RUNS_DIR does not exist. Are you in the right folder?"
exit 1
fi

# Find the most recently modified subdirectory
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
echo "  Reconstructing longitudinal profile via ToF kinematics..."
echo "============================================================"

# Execute the Time-of-Flight kinematic reconstruction script on the found directory
python3 tof_reconstruction.py --batch-dir "$LATEST_BATCH"