#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_energy_sweep.sh — Parallel OpenGATE Sweeper & Automated Post-Processing
# ─────────────────────────────────────────────────────────────────────────────

WORLD="radi_cal_energy"
PARTICLE="proton"
N_PARTICLES=1000
THREADS_PER_RUN=50
BEAM_RADIUS=0.01
OPTICAL="on"
CHERENKOV="off"
PHYSICS_LIST="QGSP_BERT_EMV"

# Define target sweep energies in keV (25 GeV, 50 GeV, 100 GeV, 200 GeV)
ENERGIES_KEV=(25000000 50000000 100000000 200000000)

# Create a shared batch timestamp directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_BATCH_DIR="runs/${WORLD}/sweep_${TIMESTAMP}"

mkdir -p "$MASTER_BATCH_DIR"

echo "========================================================================"
echo " Launching Simultaneous OpenGATE Energy Sweep"
echo "========================================================================"
echo " Target Geometry   : ${WORLD}"
echo " Particle Type     : ${PARTICLE}"
echo " Primaries / Run   : ${N_PARTICLES}"
echo " Cores Per Energy  : ${THREADS_PER_RUN} (Total: $(( ${#ENERGIES_KEV[@]} * THREADS_PER_RUN )) threads)"
echo " Output Master Dir : ${MASTER_BATCH_DIR}"
echo "========================================================================"

# Track child PIDs for global synchronization at the script end
SPAWNED_PIDS=()

# Loop through each energy and fork them simultaneously into the background
for ENERGY in "${ENERGIES_KEV[@]}"; do
    # Convert keV to human-readable text for folder separation
    ENERGY_GBS=$(( ENERGY / 1000000 ))
    ENERGY_DIR="${MASTER_BATCH_DIR}/${ENERGY_GBS}GeV"
    
    mkdir -p "$ENERGY_DIR"

    # Enclose the simulation + analysis sequence for this specific energy inside a background subshell
    (
        echo " [+] [${ENERGY_GBS}GeV] Starting OpenGATE Simulation..."
        
        # 1. Run the simulation (Outputs inside run_0 subfolder matching simulator.py structure)
        python3 simulator.py \
            --world        "$WORLD" \
            --particle     "$PARTICLE" \
            --energy-kev   "$ENERGY" \
            --n            "$N_PARTICLES" \
            --threads      "$THREADS_PER_RUN" \
            --beam-radius  "$BEAM_RADIUS" \
            --optical      "$OPTICAL" \
            --cherenkov    "$CHERENKOV" \
            --physics-list "$PHYSICS_LIST" \
            --run-id       0 \
            --output-dir   "$ENERGY_DIR" > "${ENERGY_DIR}/sim_execution_log.txt" 2>&1

        SIM_STATUS=$?
        if [ $SIM_STATUS -ne 0 ]; then
            echo " [!] [${ENERGY_GBS}GeV] Simulation failed! Check logs: ${ENERGY_DIR}/sim_execution_log.txt"
            exit 1
        fi

        echo " [+] [${ENERGY_GBS}GeV] Simulation complete. Running data analysis pipeline..."
        ANALYSIS_LOG="${ENERGY_DIR}/analysis_pipeline_log.txt"
        touch "$ANALYSIS_LOG"

        # 2. Run standard dose/hit tallying
        if [ -f "analyze.py" ]; then
            echo "     -> [${ENERGY_GBS}GeV] Running analyze.py..."
            python3 analyze.py --batch-dir "$ENERGY_DIR" >> "$ANALYSIS_LOG" 2>&1
        fi

        # 3. Run timing resolution evaluation
        if [ -f "timing_res.py" ]; then
            echo "     -> [${ENERGY_GBS}GeV] Running timing_res.py..."
            python3 timing_res.py --batch-dir "$ENERGY_DIR" >> "$ANALYSIS_LOG" 2>&1
        fi

        # 4. Run longitudinal Time-of-Flight kinematic reconstruction
        if [ -f "tof_reconstruction.py" ]; then
            echo "     -> [${ENERGY_GBS}GeV] Running tof_reconstruction.py..."
            python3 tof_reconstruction.py --batch-dir "$ENERGY_DIR" >> "$ANALYSIS_LOG" 2>&1
        fi

        echo " [✓] [${ENERGY_GBS}GeV] Full pipeline execution finished successfully."
    ) &
    
    # Store the background job PID
    SPAWNED_PIDS+=($!)
done

echo " [+] All energy processes spawned. Monitoring concurrent workloads..."
echo "------------------------------------------------------------------------"

# Block the main terminal execution until all 4 parallel paths complete
for PID in "${SPAWNED_PIDS[@]}"; do
    wait "$PID"
done

echo "========================================================================"
echo " Sweep complete. Comprehensive dataset stored in:"
echo " $MASTER_BATCH_DIR"
echo "========================================================================"