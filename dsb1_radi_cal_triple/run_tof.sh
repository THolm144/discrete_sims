#!/bin/bash
RUNS_DIR="runs/dsb1_radi_cal_triple"

# Ensure the base directory exists
if [ ! -d "$RUNS_DIR" ]; then
    echo "Error: Directory $RUNS_DIR does not exist. Are you in the right folder?"
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
echo "  Reconstructing longitudinal profile via ToF kinematics..."
echo "  Looping through energy subdirectories..."
echo "============================================================"

# Loop over each individual energy subdirectory (e.g., 25GeV, 50GeV)
# 'sort -V' ensures natural numerical order so 25 processes before 50, 100, 200
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

    # Execute the Time-of-Flight kinematic reconstruction script
    if [ -f "tof_reconstruction.py" ]; then
        echo "  -> Running tof_reconstruction.py..."
        python3 tof_reconstruction.py --batch-dir "$ENERGY_DIR"
    else
        echo "  [Error] tof_reconstruction.py not found in current working directory."
    fi

done

echo ""
echo "============================================================"
echo "  ToF Reconstruction pipeline execution finished."
echo "============================================================"