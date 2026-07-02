#!/bin/bash

# Define the base directory containing the simulation sweeps
RUNS_DIR="/home/uakgun/env/THOMAS/discrete_sims/radi_cal_energy/runs/radi_cal_energy"

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

LATEST_BATCH=${LATEST_BATCH%/}

echo "============================================================"
echo "  Targeting latest sweep batch: $LATEST_BATCH"
echo "  Processing each energy folder individually..."
echo "============================================================"

# Loop over each individual energy subdirectory (e.g., 25GeV, 50GeV)
# Sorting ensuring they process in numerical order
for ENERGY_DIR in $(ls -d "$LATEST_BATCH"/*GeV/ 2>/dev/null | sort -V); do
    
    # Verify it's a valid directory
    [ -d "$ENERGY_DIR" ] || continue
    ENERGY_DIR=${ENERGY_DIR%/}
    ENERGY_NAME=$(basename "$ENERGY_DIR")

    echo ""
    echo " ────────────────────────────────────────────────────────────"
    echo "  Processing Energy Target: $ENERGY_NAME"
    echo "  Directory: $ENERGY_DIR"
    echo " ────────────────────────────────────────────────────────────"

    # 1. Run time profile plotting script
    if [ -f "plot_hits_vs_time.py" ]; then
        echo "  -> Running plot_hits_vs_time.py..."
        python3 plot_hits_vs_time.py --batch-dir "$ENERGY_DIR"
    else
        echo "  [Warning] plot_hits_vs_time.py not found in current working directory."
    fi

    echo "  ------------------------------------------------------------"

    # 2. Run timing resolution analysis script 
    if [ -f "timing_res.py" ]; then
        echo "  -> Running timing_res.py..."
        python3 timing_res.py --batch-dir "$ENERGY_DIR"
    else
        echo "  [Warning] timing_res.py not found in current working directory."
    fi
done

echo ""
echo "============================================================"
echo "  All individual energy steps analyzed successfully."
echo "============================================================"