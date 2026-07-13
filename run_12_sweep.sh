#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_12_sweep_512.sh — High-Velocity 512-Core OpenGATE Sweeper
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR=$(pwd)

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

N_PARTICLES_PER_RUN=24
N_RUNS_PER_ENERGY=43
THREADS_PER_RUN=1
MAX_GLOBAL_CONCURRENT_SIMS=480

ENERGIES_KEV=(25000000 50000000 100000000 200000000)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================================================"
echo " Launching High-Velocity OpenGATE Sweeper [Active Telemetry]"
echo "========================================================================"
echo " Target Pool Capacity : ${MAX_GLOBAL_CONCURRENT_SIMS} Single-Threaded Cores"
echo " Total Job Footprint  : 2,064 Simulation Batches"
echo "========================================================================"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: HIGH-VELOCITY DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
echo " [+] Rapidly populating core pool..."

TOTAL_LAUNCHED=0

for WORLD in "${WORLDS[@]}"; do
    if [ ! -d "$WORLD" ]; then continue; fi

    MASTER_BATCH_DIR="${ROOT_DIR}/${WORLD}/runs/sweep_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    for ENERGY in "${ENERGIES_KEV[@]}"; do
        ENERGY_GBS=$(( ENERGY / 1000000 ))
        ENERGY_DIR="${MASTER_BATCH_DIR}/${ENERGY_GBS}GeV"
        mkdir -p "$ENERGY_DIR"

        for RUN_ID in $(seq 0 $((N_RUNS_PER_ENERGY - 1))); do
            
            # Pool Throttling: Only stalls if the cluster is genuinely saturated
            while [ $(jobs -rp | wc -l) -ge $MAX_GLOBAL_CONCURRENT_SIMS ]; do
                sleep 0.05
            done

            LOG_FILE="${MASTER_BATCH_DIR}/logs/${ENERGY_GBS}GeV_run_${RUN_ID}.log"
            RUN_OUT_DIR="${ENERGY_DIR}"

            # Fire instantly with zero mandatory tailing delays
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
            ) & 

            ((TOTAL_LAUNCHED++))

            # Active Telemetry: Updates line in place so you can watch it climb
            printf "\r     -> Pipeline Load Status: %4d / 2064 Tasks Allocated" "$TOTAL_LAUNCHED"
        done
    done
done

echo ""
echo " [✓] Global pool fully saturated. Handing execution over to core group."
echo " [+] Waiting for background workers to drop to 0..."
wait
echo " [✓] Simulation phase complete. Running data analysis blocks..."
echo "------------------------------------------------------------------------"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: SEQUENTIAL POST-PROCESSING
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