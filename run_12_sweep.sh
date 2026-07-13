#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_12_sweep.sh — Multi-Model Localized OpenGATE Sweeper
# ─────────────────────────────────────────────────────────────────────────────

# Keep a record of the absolute root path to jump back to
ROOT_DIR=$(pwd)

# All 12 subdirectories matching your models
WORLDS=(
    "radi_cal"               "radi_cal_triple"             "rc_hex"             "rc_hex_triple"
    "dsb1_radi_cal_energy"   "dsb1_radi_cal_triple"        "dsb1_rc_hex"        "dsb1_rc_hex_triple"
    "luagce_radi_cal_energy" "luagce_radi_cal_triple"      "luagce_rc_hex"      "luagce_rc_hex_triple"
)

# --- Global Parameter Controls ---
PARTICLE="e-"
BEAM_RADIUS=0.01
OPTICAL="on"
CHERENKOV="off"
PHYSICS_LIST="QGSP_BERT_EMV"

# --- Core-Pool Optimization & Math ---
N_PARTICLES_PER_RUN=14
N_RUNS_PER_ENERGY=43
THREADS_PER_RUN=1       # 1 thread eliminates multi-threading lock contention
MAX_CONCURRENT_SIMS=43  # Utilize all 200 physical cores safely

# Define target sweep energies in keV (25 GeV, 50 GeV, 100 GeV, 200 GeV)
ENERGIES_KEV=(25000000 50000000 100000000 200000000)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================================================"
echo " Starting OpenGATE Energy Sweep Pipeline across 12 Localized Models"
echo "========================================================================"
echo " Particle Type     : ${PARTICLE}"
echo " Concurrent Workers: ${MAX_CONCURRENT_SIMS} Single-Threaded Cores"
echo " Sweep Timestamp   : ${TIMESTAMP}"
echo "========================================================================"

for WORLD in "${WORLDS[@]}"; do
    
    # 1. Verify directory exists, then move into it
    if [ ! -d "$WORLD" ]; then
        echo " [!] Warning: Directory ${WORLD} not found. Skipping."
        continue
    fi

    cd "$WORLD" || continue
    
    # Define local path relative to the module folder (just like your original script)
    MASTER_BATCH_DIR="runs/sweep_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    echo ""
    echo " ─────────────────────────────────────────────────────────────────────"
    echo " >> WORKING IN DIRECTORY: ${WORLD}"
    echo " >> Local Output Target : ${WORLD}/${MASTER_BATCH_DIR}"
    echo " ─────────────────────────────────────────────────────────────────────"

    # Loop through each energy step
    for ENERGY in "${ENERGIES_KEV[@]}"; do
        ENERGY_GBS=$(( ENERGY / 1000000 ))
        ENERGY_DIR="${MASTER_BATCH_DIR}/${ENERGY_GBS}GeV"
        mkdir -p "$ENERGY_DIR"

        echo "  [+] Queueing primaries for [${ENERGY_GBS}GeV]..."

        for RUN_ID in $(seq 0 $((N_RUNS_PER_ENERGY - 1))); do
            # Throttle parallel tasks using local jobs tracking
            while [ $(jobs -rp | wc -l) -ge $MAX_CONCURRENT_SIMS ]; do
                sleep 0.1
            done

            LOG_FILE="${MASTER_BATCH_DIR}/logs/${ENERGY_GBS}GeV_run_${RUN_ID}.log"
            RUN_OUT_DIR="${ENERGY_DIR}"

            # Dispatch simulation worker using the local module's simulator.py
            # Pass the base world name (e.g., "rc_hex") if required, or keep $WORLD depending on how config maps it
            python3 simulator.py \
                --world        "$WORLD" \
                --particle     "$PARTICLE" \
                --energy-kev   "$ENERGY" \
                --n            "$N_PARTICLES_PER_RUN" \
                --threads      "$THREADS_PER_RUN" \
                --beam-radius  "$BEAM_RADIUS" \
                --optical      "$OPTICAL" \
                --cherenkov    "$CHERENKOV" \
                --hits-optical-only on \
                --physics-list "$PHYSICS_LIST" \
                --run-id       "$RUN_ID" \
                --output-dir   "$RUN_OUT_DIR" > "$LOG_FILE" 2>&1 &
        done
    done

    echo "  [+] All simulation tasks dispatched for ${WORLD}. Waiting for core pool..."
    wait
    echo "  [✓] Simulation phase complete for ${WORLD}. Running local analysis pipeline..."

    # ─────────────────────────────────────────────────────────────────────────
    # POST-PROCESSING PIPELINE (Executed locally per module)
    # ─────────────────────────────────────────────────────────────────────────
    for ENERGY in "${ENERGIES_KEV[@]}"; do
        ENERGY_GBS=$(( ENERGY / 1000000 ))
        ENERGY_DIR="${MASTER_BATCH_DIR}/${ENERGY_GBS}GeV"
        ANALYSIS_LOG="${ENERGY_DIR}/analysis_pipeline_log.txt"
        touch "$ANALYSIS_LOG"

        echo "      -> [${ENERGY_GBS}GeV] Processing local data logs..."

        if [ -f "analyze.py" ]; then
            python3 analyze.py --batch-dir "$ENERGY_DIR" --workers 64 >> "$ANALYSIS_LOG" 2>&1
        fi

        if [ -f "timing_res.py" ]; then
            python3 timing_res.py --batch-dir "$ENERGY_DIR" >> "$ANALYSIS_LOG" 2>&1
        fi

        if [ -f "tof_reconstruction.py" ]; then
            python3 tof_reconstruction.py --batch-dir "$ENERGY_DIR" >> "$ANALYSIS_LOG" 2>&1
        fi
    done
    echo "  [✓] Post-processing step finished for ${WORLD}."

    # 2. Return to root before processing next world item
    cd "$ROOT_DIR" || exit
done

echo ""
echo "========================================================================"
echo " ENTIRE 12-MODEL PIPELINE COMPLETE."
echo " Outputs natively populated within each folder's /runs/ directory."
echo "========================================================================"