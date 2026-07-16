#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_energy_sweep.sh — Hyper-Optimized 4-Energy Sweeper (Single Module)
# ─────────────────────────────────────────────────────────────────────────────

WORLD="dsb1_rc_hex_triple" # Update this for each module you test
PARTICLE="e-"
BEAM_RADIUS=0.01
OPTICAL="on"
CHERENKOV="off"
PHYSICS_LIST="QGSP_BERT_EMV"

# --- MAX SPEED MATH TWEAK ---
# 10 single-threaded runs dispatched simultaneously per energy.
# 4 energies * 10 runs = 40 background processes total.
# 10 runs * 1500 particles = 15,000 total particles per energy.
N_PARTICLES_PER_RUN=1500
N_RUNS_PER_ENERGY=10
THREADS_PER_RUN=1

# Define target sweep energies in keV (25 GeV, 50 GeV, 70 GeV, 90 GeV)
ENERGIES_KEV=(25000000 50000000 70000000 90000000)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_BATCH_DIR="runs/${WORLD}/sweep_${TIMESTAMP}"
mkdir -p "$MASTER_BATCH_DIR/logs"

echo "========================================================================"
echo " Launching Max-Speed 4-Energy Sweep for Module: ${WORLD}"
echo "========================================================================"
echo " Target Geometry   : ${WORLD}"
echo " Configuration     : ${N_RUNS_PER_ENERGY} runs of ${N_PARTICLES_PER_RUN} particles per Energy"
echo " Concurrent Sims   : 40 single-threaded jobs running simultaneously"
echo " Total CPU Load    : 40 cores per module script"
echo " Output Master Dir : ${MASTER_BATCH_DIR}"
echo "========================================================================"

# Dispatch ALL jobs immediately to the background (10 runs x 4 energies = 40 jobs)
for ENERGY in "${ENERGIES_KEV[@]}"; do
    ENERGY_GBS=$(( ENERGY / 1000000 ))
    ENERGY_DIR="${MASTER_BATCH_DIR}/${ENERGY_GBS}GeV"
    mkdir -p "$ENERGY_DIR"

    echo " [+] Blasting 10 parallel jobs for [${ENERGY_GBS}GeV]..."

    for RUN_ID in $(seq 0 $((N_RUNS_PER_ENERGY - 1))); do
        LOG_FILE="${MASTER_BATCH_DIR}/logs/${ENERGY_GBS}GeV_run_${RUN_ID}.log"
        
        # Notice threads is now 1, and we push directly to background (&)
 python3 simulator.py --beam-x 0.00000 --beam-y 0.35000 \
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
            --output-dir   "$ENERGY_DIR" > "$LOG_FILE" 2>&1 &
    done
done

echo " [+] All 40 jobs dispatched to CPU pool. Waiting for execution to finish..."
wait
echo " [✓] Simulation sweep complete. Initiating analysis pipeline..."
echo "------------------------------------------------------------------------"

# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
for ENERGY in "${ENERGIES_KEV[@]}"; do
    ENERGY_GBS=$(( ENERGY / 1000000 ))
    ENERGY_DIR="${MASTER_BATCH_DIR}/${ENERGY_GBS}GeV"
    ANALYSIS_LOG="${ENERGY_DIR}/analysis_pipeline_log.txt"
    touch "$ANALYSIS_LOG"

    echo " [+] [${ENERGY_GBS}GeV] Post-Processing (Running concurrently)..."

    # Maintain 10 workers for analysis to keep total CPU load consistent
    (
        if [ -f "analyze.py" ]; then
            python3 analyze.py --batch-dir "$ENERGY_DIR" --workers 10 >> "$ANALYSIS_LOG" 2>&1
        fi

        if [ -f "timing_res.py" ]; then
            python3 timing_res.py --batch-dir "$ENERGY_DIR" >> "$ANALYSIS_LOG" 2>&1
        fi

        if [ -f "tof_reconstruction.py" ]; then
            python3 tof_reconstruction.py --batch-dir "$ENERGY_DIR" >> "$ANALYSIS_LOG" 2>&1
        fi
    ) &
done

wait
echo "========================================================================"
echo " Sweep and analysis complete. Dataset stored in:"
echo " $MASTER_BATCH_DIR"
echo "========================================================================"
