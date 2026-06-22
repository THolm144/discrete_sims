#!/bin/bash

# Define the relative path to the runs directory
TARGET_BASE="runs/radi_cal"

# 1. Verify the base runs directory exists
if [ ! -d "$TARGET_BASE" ]; then
    echo "Error: Base directory '$TARGET_BASE' does not exist."
    echo "Make sure you are running this script from: ~/THOMAS/discrete_sims/radi_cal"
    exit 1
fi

# 2. Find the most recent sweep directory
# Uses ls -td to sort by modification time, or fallback to chronological folder names
LATEST_SWEEP=$(ls -td "${TARGET_BASE}"/sweep_* 2>/dev/null | head -n 1)

if [ -z "$LATEST_SWEEP" ]; then
    echo "Error: No 'sweep_*' directories found inside $TARGET_BASE"
    exit 1
fi

echo "========================================================================"
echo " Target Sweep Folder: $LATEST_SWEEP"
echo "========================================================================"

# 3. Iterate through each energy directory inside the sweep
# The trailing slash ensures we only match directories, not summary text files
for energy_dir in "${LATEST_SWEEP}"/*keV/; do
    
    # Check if the glob found anything valid
    if [ ! -d "$energy_dir" ]; then
        echo "No valid energy subdirectories (*keV) found in $LATEST_SWEEP."
        exit 1
    fi

    # Clean up trailing slash for logging clarity
    energy_dir="${energy_dir%/}"
    energy_name=$(basename "$energy_dir")

    echo ""
    echo "------------------------------------------------------------------------"
    echo " Processing Energy: $energy_name"
    echo " Directory Path   : $energy_dir"
    echo "------------------------------------------------------------------------"

    # 4. Run the python timing script
    python3 timing_res.py --batch-dir "$energy_dir"

done

echo ""
echo "========================================================================"
echo " Processing complete for all energy configurations in the current sweep!"
echo "========================================================================"