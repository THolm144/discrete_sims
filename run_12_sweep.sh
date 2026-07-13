#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_12_sweep_xargs.sh — High-Velocity 512-Core OpenGATE Sweeper (With Counter)
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR=$(pwd)
JOB_FILE="${ROOT_DIR}/job_list.txt"
> "$JOB_FILE" # Clear any previous job file

WORLDS=(
    "radi_cal"               "radi_cal_triple"             "rc_hex"             "rc_hex_triple"
    "dsb1_radi_cal_energy"   "dsb1_radi_cal_triple"        "dsb1_rc_hex"        "dsb1_rc_hex_triple"
    "luagce_radi_cal_energy" "luagce_radi_cal_triple"      "luagce_rc_hex"      "luagce_rc_hex_triple"
)

PARTICLE="e-"
BEAM_RADIUS=0.01
OPTICAL="on"
CHERENKOV="off"
PHYSICS_LIST="QGSP_BERT_EMV"

N_PARTICLES_PER_RUN=4
N_RUNS_PER_ENERGY=43
THREADS_PER_RUN=1
MAX_GLOBAL_CONCURRENT_SIMS=480

ENERGIES_KEV=(25000000 50000000 100000000 200000000)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================================================"
echo " Launching High-Velocity OpenGATE Sweeper [xargs Cluster Engine]"
echo "========================================================================"
echo " Max Parallel Workers : ${MAX_GLOBAL_CONCURRENT_SIMS} Cores"
echo " Total Matrix Size    : 2,064 Simulation Batches"
echo "========================================================================"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: GENERATE JOB LIST
# ─────────────────────────────────────────────────────────────────────────────
echo " [+] Generating unified cluster execution matrix..."

for WORLD in "${WORLDS[@]}"; do
    if [ ! -d "$WORLD" ]; then continue; fi

    MASTER_BATCH_DIR="${ROOT_DIR}/${WORLD}/runs/sweep_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    for ENERGY in "${ENERGIES_KEV[@]}"; do
        ENERGY_GBS=$(( ENERGY / 1000000 ))
        ENERGY_DIR="${MASTER_BATCH_DIR}/${ENERGY_GBS}GeV"
        mkdir -p "$ENERGY_DIR"

        for RUN_ID in $(seq 0 $((N_RUNS_PER_ENERGY - 1))); do
            LOG_FILE="${MASTER_BATCH_DIR}/logs/${ENERGY_GBS}GeV_run_${RUN_ID}.log"
            RUN_OUT_DIR="${ENERGY_DIR}"

            # Append the execution command
            echo "cd ${ROOT_DIR}/${WORLD} && python3 simulator.py --world ${WORLD} --particle ${PARTICLE} --energy-kev ${ENERGY} --n ${N_PARTICLES_PER_RUN} --threads ${THREADS_PER_RUN} --beam-radius ${BEAM_RADIUS} --optical ${OPTICAL} --cherenkov ${CHERENKOV} --hits-optical-only on --physics-list ${PHYSICS_LIST} --run-id ${RUN_ID} --output-dir ${RUN_OUT_DIR} > ${LOG_FILE} 2>&1" >> "$JOB_FILE"
        done
    done
done

TOTAL_JOBS=$(wc -l < "$JOB_FILE")
echo " [✓] Generated ${TOTAL_JOBS} jobs in job_list.txt."
echo "--------------------------------────────────────────────────────--------"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: MASS PARALLEL EXECUTION WITH LIVE COUNTER
# ─────────────────────────────────────────────────────────────────────────────
echo " [+] Handing execution matrix to xargs engine..."
echo " [+] Filling 480 processing cores..."

# Start a background loop to track log files and update the screen
(
    # Give xargs a moment to spin up and create folders/files
    sleep 2 
    while [ -f "$JOB_FILE" ] || [ $(pgrep -f "simulator.py" | wc -l) -gt 0 ]; do
        # Count all .log files inside the newly created sweep log directories
        COMPLETED_JOBS=$(find ${ROOT_DIR}/*/runs/sweep_${TIMESTAMP}/logs -name "*.log" 2>/dev/null | wc -l)
        
        # Pull how many Python simulations are currently processing
        ACTIVE_CORES=$(pgrep -f "simulator.py" | wc -l)

        printf "\r     -> Cluster Status: %4d / %d Done | [%3d Cores Occupied]" "$COMPLETED_JOBS" "$TOTAL_JOBS" "$ACTIVE_CORES"
        sleep 1
    done
) &
TRACKER_PID=$!

# Run the xargs parallel processing engine smoothly
xargs -P "$MAX_GLOBAL_CONCURRENT_SIMS" -I {} sh -c "{}" < "$JOB_FILE"

# Clean up files and close background tracker
rm -f "$JOB_FILE"
wait $TRACKER_PID 2>/dev/null

echo ""
echo " [✓] All background core pools finished. Proceeding to analytics pipeline..."
echo "------------------------------------------------------------------------"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: SEQUENTIAL POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
for WORLD in "${WORLDS[@]}"; do
    if [ ! -d "$WORLD" ]; then continue; fi
    
    cd "${ROOT_DIR}/${WORLD}" || continue
    MASTER_BATCH_DIR="runs/sweep_${TIMESTAMP}"

    echo " [+] Processing summaries for: ${WORLD}"

    for ENERGY in "${ENERGIES_KEV[@]}"; do
        ENERGY_GBS=$(( ENERGY / 1000000 ))
        ENERGY_DIR="${MASTER_BATCH_DIR}/${ENERGY_GBS}GeV"
        ANALYSIS_LOG="${ENERGY_DIR}/analysis_pipeline_log.txt"
        touch "$ANALYSIS_LOG"

        if [ -f "analyze.py" ]; then
            python3 analyze.py --batch-dir "$ENERGY_DIR" --workers 128 >> "$ANALYSIS_LOG" 2>&1
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
echo " PIPELINE RUN COMPLETE."
echo "========================================================================"