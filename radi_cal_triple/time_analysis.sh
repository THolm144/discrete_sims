#!/bin/bash

# Define the base directory containing the simulation runs
RUNS_DIR="runs/radi_cal_triple"

# Ensure the base directory exists
if [ ! -d "$RUNS_DIR" ]; then
    echo "Error: Directory $RUNS_DIR does not exist. Are you in the right folder?"
    exit 1
fi

# Find the most recently modified sub-directory inside RUNS_DIR, regardless of name
# This looks only at top-level directories under RUNS_DIR (-maxdepth 1)
LATEST_BATCH=$(find "$RUNS_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)

# Check if a directory was actually found
if [ -z "$LATEST_BATCH" ]; then
    echo "Error: No directories found inside $RUNS_DIR."
    exit 1
fi

# Strip the trailing slash if present for a cleaner path string
LATEST_BATCH=${LATEST_BATCH%/}

# ─────────────────────────────────────────────────────────────────────────────
# Runs the analysis pipeline (time profile + timing resolution) on a single
# directory. The directory can contain run_N dirs directly (a flat batch) or
# be an energy sub-folder of a sweep containing several run_N dirs -- both
# scripts already recurse (rglob) for the ROOT files they need, so pointing
# --batch-dir at either kind of directory aggregates everything beneath it.
# ─────────────────────────────────────────────────────────────────────────────
run_pipeline() {
    local target_dir="$1"

    echo "-> Running plot_hits_vs_time.py..."
    python3 plot_hits_vs_time.py --batch-dir "$target_dir"

    echo "------------------------------------------------------------"

    echo "-> Running timing_res.py..."
    python3 timing_res.py --batch-dir "$target_dir"
}

echo "============================================================"
echo "  Targeting latest batch: $LATEST_BATCH"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Detect whether LATEST_BATCH is a flat batch (run_N dirs directly inside it)
# or a sweep (energy sub-folders like 25GeV/, 50GeV/, each containing run_N
# dirs two levels down from LATEST_BATCH).
# ─────────────────────────────────────────────────────────────────────────────
HAS_FLAT_RUNS=$(find "$LATEST_BATCH" -mindepth 1 -maxdepth 1 -type d -name "run_*" | head -n 1)
HAS_SWEEP_RUNS=$(find "$LATEST_BATCH" -mindepth 2 -maxdepth 2 -type d -name "run_*" | head -n 1)

if [ -n "$HAS_FLAT_RUNS" ]; then
    # ── Flat batch: run_N dirs live directly under LATEST_BATCH ──
    echo "  Detected flat batch (run_* directories directly inside)."
    echo "  Executing batch analysis pipeline..."
    echo "============================================================"
    run_pipeline "$LATEST_BATCH"

elif [ -n "$HAS_SWEEP_RUNS" ]; then
    # ── Sweep: energy sub-folders, each containing run_N dirs ──
    echo "  Detected sweep (energy sub-folders with run_* dirs inside)."
    echo "============================================================"

    # Natural/version sort so 25GeV, 50GeV, 100GeV order sensibly
   mapfile -t ENERGY_DIRS < <(find "$LATEST_BATCH" -mindepth 2 -maxdepth 2 -type d -name "run_*" -printf '%h\n' | sort -u -V)

    if [ ${#ENERGY_DIRS[@]} -eq 0 ]; then
        echo "Error: Sweep structure detected but no energy sub-folders found."
        exit 1
    fi

    for energy_dir in "${ENERGY_DIRS[@]}"; do
        energy_label=$(basename "$energy_dir")
        echo ""
        echo "------------------------------------------------------------"
        echo "  Energy point: $energy_label"
        echo "  Directory   : $energy_dir"
        echo "------------------------------------------------------------"
        run_pipeline "$energy_dir"
    done

else
    echo "Error: Could not find run_* directories directly inside $LATEST_BATCH"
    echo "       or nested one level down (energy sub-folders). Unrecognized"
    echo "       directory structure -- nothing to analyze."
    exit 1
fi

echo ""
echo "============================================================"
echo "  Pipeline execution finished successfully."
echo "============================================================"