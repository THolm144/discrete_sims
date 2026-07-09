#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_energy_sweep.sh — Hyper-Speed Single-Threaded Parallel OpenGATE Sweeper
# ─────────────────────────────────────────────────────────────────────────────

WORLD="luagce_radi_cal_triple"
PARTICLE="e-"
BEAM_RADIUS=0.01
OPTICAL="on"
CHERENKOV="off"
PHYSICS_LIST="QGSP_BERT_EMV"

# --- Optimized Core & Scale Math ---
# 4 Energies total. We have 200 cores. 
# We can allocate exactly 50 single-threaded runs per energy step simultaneously.
# 50 runs * 20 particles per run = 1,000 total particles per energy target.
N_PARTICLES_PER_RUN=60
N_RUNS_PER_ENERGY=30
THREADS_PER_RUN=1      # 1 thread eliminates multi-threading lock contention
MAX_CONCURRENT_SIMS=120 # Utilize all 200 physical cores

# Define target sweep energies in keV (25 GeV, 50 GeV, 100 GeV, 200 GeV)
ENERGIES_KEV=(25000000 50000000 100000000 200000000)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_BATCH_DIR="runs/${WORLD}/sweep_${TIMESTAMP}"
mkdir -p "$MASTER_BATCH_DIR"

echo "========================================================================"
echo " Launching Hyper-Speed Single-Threaded OpenGATE Energy Sweep"
echo "========================================================================"
echo " Target Geometry   : ${WORLD}"
echo " Particle Type     : ${PARTICLE}"
echo " Total Primaries   : $(( N_PARTICLES_PER_RUN * N_RUNS_PER_ENERGY )) per Energy"
echo " Configuration     : ${N_RUNS_PER_ENERGY} runs of ${N_PARTICLES_PER_RUN} particles"
echo " Concurrent Workers: ${MAX_CONCURRENT_SIMS} Single-Threaded Cores"
echo " Output Master Dir : ${MASTER_BATCH_DIR}"
echo "========================================================================"

# Create log directories
mkdir -p "${MASTER_BATCH_DIR}/logs"

# Loop through each energy and queue up runs inside a controlled pool
for ENERGY in "${ENERGIES_KEV[@]}"; do
    ENERGY_GBS=$(( ENERGY / 1000000 ))
    ENERGY_DIR="${MASTER_BATCH_DIR}/${ENERGY_GBS}GeV"
    mkdir -p "$ENERGY_DIR"

    echo " [+] Queueing 1,000 primaries for [${ENERGY_GBS}GeV]..."

    for RUN_ID in $(seq 0 $((N_RUNS_PER_ENERGY - 1))); do
        # Maintain our max pool capacity of 200 active background processes
        while [ $(jobs -rp | wc -l) -ge $MAX_CONCURRENT_SIMS ]; do
            sleep 0.1
        done

        LOG_FILE="${MASTER_BATCH_DIR}/logs/${ENERGY_GBS}GeV_run_${RUN_ID}.log"
        RUN_OUT_DIR="${ENERGY_DIR}"
        mkdir -p "$RUN_OUT_DIR"

        # Dispatch single particle bundle into isolated directory
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

echo " [+] All simulation processes dispatched to core pool. Waiting for execution to finish..."
wait
echo " [✓] Simulation sweep complete. Initiating sequential analysis pipeline..."
echo "------------------------------------------------------------------------"

# ─────────────────────────────────────────────────────────────────────────────
# POST-PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
for ENERGY in "${ENERGIES_KEV[@]}"; do
    ENERGY_GBS=$(( ENERGY / 1000000 ))
    ENERGY_DIR="${MASTER_BATCH_DIR}/${ENERGY_GBS}GeV"
    ANALYSIS_LOG="${ENERGY_DIR}/analysis_pipeline_log.txt"
    touch "$ANALYSIS_LOG"

    echo " [+] [${ENERGY_GBS}GeV] Post-Processing..."

    if [ -f "analyze.py" ]; then
        echo "     -> Running analyze.py..."
        python3 analyze.py --batch-dir "$ENERGY_DIR" --workers 64 >> "$ANALYSIS_LOG" 2>&1
    fi

    if [ -f "timing_res.py" ]; then
        echo "     -> Running timing_res.py..."
        python3 timing_res.py --batch-dir "$ENERGY_DIR" >> "$ANALYSIS_LOG" 2>&1
    fi

    if [ -f "tof_reconstruction.py" ]; then
        echo "     -> Running tof_reconstruction.py..."
        python3 tof_reconstruction.py --batch-dir "$ENERGY_DIR" >> "$ANALYSIS_LOG" 2>&1
    fi
done

echo "========================================================================"
echo " Sweep complete. Dataset verified and stored in:"
echo " $MASTER_BATCH_DIR"
echo "========================================================================"