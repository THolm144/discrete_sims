#!/bin/bash

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
PYTHON_SCRIPT="thickness_analysis.py"  # Path to your python script
RUNS_DIR="runs/quartz_cal"  # Change to "/runs" if it's at the system root

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
LATEST_BATCH=${LATEST_BATCH%/}                # Change to "/runs" if it's at the system root

# ──────────────────────────────────────────────────────────────────────
# SANITY CHECKS
# ──────────────────────────────────────────────────────────────────────
# 1. Ensure the base runs directory exists
if [ ! -d "$RUNS_DIR" ]; then
    echo "[Error] Base directory '$RUNS_DIR' does not exist."
    exit 1
fi

# 2. Ensure the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "[Error] Python script '$PYTHON_SCRIPT' not found."
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────
# FIND LATEST NON-EMPTY FOLDER
# ──────────────────────────────────────────────────────────────────────
LATEST_DIR=""

echo "Scanning '$RUNS_DIR' for the latest active batch..."

# Loop through directories sorted by modification time (newest first)
# 'ls -td' handles sorting natively across Linux and macOS
for dir in $(ls -td "$RUNS_DIR"/*/ 2>/dev/null); do
    # Check if the directory actually contains anything (ignores completely empty folders)
    if [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        LATEST_DIR="$dir"
        break # Stop at the first (newest) non-empty directory found
    fi
done

# If no directory was found
if [ -z "$LATEST_DIR" ]; then
    echo "[Error] No valid, non-empty run directories found in '$RUNS_DIR'."
    exit 1
fi

# ──────────────────────────────────────────────────────────────────────
# EXECUTION
# ──────────────────────────────────────────────────────────────────────
echo "[+] Found latest data target: $LATEST_DIR"
echo "[+] Triggering Python simulation analysis..."
echo "------------------------------------------------"

python3 "$PYTHON_SCRIPT" --master-dir "$LATEST_DIR"