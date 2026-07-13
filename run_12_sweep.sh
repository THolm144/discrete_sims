#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_12_sweep_hyper.sh — Maximum CPU Saturation Parallel OpenGATE Sweeper
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR=$(pwd)

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

# --- Extreme Cluster Optimization ---
N_PARTICLES_PER_RUN=24
N_RUNS_PER_ENERGY=43
THREADS_PER_RUN=1

# GLOBAL CONCURRENCY LIMIT: Set this to match your total physical core count
# Since each simulation uses 1 thread, this keeps up to 480 cores saturated at all times.
MAX_GLOBAL_CONCURRENT_SIMS=480

# Define target sweep energies in keV
ENERGIES_KEV=(25000000 50000000 100000000 200000000)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================================================"
echo " Launching Hyper-Parallel OpenGATE Sweeper [All Models Simultaneous]"
echo "========================================================================"
echo " Particle Type         : ${PARTICLE}"
echo " Global Core Pool Limit: ${MAX_GLOBAL_CONCURRENT_SIMS} Single-Threaded Cores"
echo " Sweep Timestamp       : ${TIMESTAMP}"
echo "========================================================================"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: DISPATCH SIMULATIONS (GLOBAL POOL TRAPPING)
# ─────────────────────────────────────────────────────────────────────────────
echo " [+] Dispatching all model sweeps into a unified background pool..."

for WORLD in "${WORLDS[@]}"; do
    if [ ! -d "$WORLD" ]; then
        echo " [!] Warning: Directory ${WORLD} not found. Skipping."
        continue
    fi

    # Define the output directory path from the root perspective
    MASTER_BATCH_DIR="${ROOT_DIR}/${WORLD}/runs/sweep_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    for ENERGY in "${ENERGIES_KEV[@]}"; do
        ENERGY_GBS=$(( ENERGY / 1000000 ))
        ENERGY_DIR="${MASTER_BATCH_DIR}/${ENERGY_GBS}GeV"
        mkdir -p "$ENERGY_DIR"

        for RUN_ID in $(seq 0 $((N_RUNS_PER_ENERGY - 1))); do
            
            # Global throttle: checks ALL background tasks spawned by this script
            while [ $(jobs -rp | wc -l) -ge $MAX_GLOBAL_CONCURRENT_SIMS ]; do
                sleep 0.05
            done

            LOG_FILE="${MASTER_BATCH_DIR}/logs/${ENERGY_GBS}GeV_run_${RUN_ID}.log"
            RUN_OUT_DIR="${ENERGY_DIR}"

            # Execute python inside a subshell context so it retains local file paths natively
            (
                cd "$WORLD" || exit
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
                    --output-dir   "$RUN_OUT_DIR" > "$LOG_FILE" 2>&1
            ) & # Spawns into background globally
        done
    done
done

echo " [+] All simulation threads across all 12 models successfully queued."
echo " [+] Keeping cluster saturated. Waiting for final simulation tasks to complete..."
wait
echo " [✓] Simulation phase complete for all models. Initiating analysis phase..."
echo "------------------------------------------------------------------------"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: POST-PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
# Now that simulations are done, we can safely run analysis routines.
# We do this sequentially per model to ensure python file handling/memory leaks 
# don't thrash the filesystem or crash the kernel.
for WORLD in "${WORLDS[@]}"; do
    if [ ! -d "$WORLD" ]; then continue; fi
    
    cd "${ROOT_DIR}/${WORLD}" || continue
    MASTER_BATCH_DIR="runs/sweep_${TIMESTAMP}"

    echo " [+] Processing analytical pipeline for: ${WORLD}"

    for ENERGY in "${ENERGIES_KEV[@]}"; do
        ENERGY_GBS=$(( ENERGY / 1000000 ))
        ENERGY_DIR="${MASTER_BATCH_DIR}/${ENERGY_GBS}GeV"
        ANALYSIS_LOG="${ENERGY_DIR}/analysis_pipeline_log.txt"
        touch "$ANALYSIS_LOG"

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
done

cd "$ROOT_DIR" || exit
echo "========================================================================"
echo " GLOBAL PIPELINE COMPLETE."
echo " All outputs have populated their native /runs/ subdirectories."
echo "========================================================================"