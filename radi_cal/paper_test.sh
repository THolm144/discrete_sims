#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim.sh — OpenGATE simulation launcher (Organized Multi-Energy)
# Runs N_RUNS per energy in parallel, each with THREADS Geant4 threads.
# All runs at a given energy complete before analysis begins.
# ─────────────────────────────────────────────────────────────────────────────

WORLD="radi_cal"
PARTICLE="e-"
ENERGIES_KEV=(25000000 50000000 100000000 150000000)
N_PARTICLES=1000
THREADS=16       # Geant4 MT threads per run (1000 / 16 ≈ 62 particles/thread)
N_RUNS=5         # Runs per energy, launched in parallel
BEAM_RADIUS=0.01
OPTICAL="on"

# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="runs/${WORLD}/sweep_${TIMESTAMP}"
SUMMARY_FILE="${BASE_DIR}/sweep_summary.txt"
PLOTS_DIR="${BASE_DIR}/compiled_plots"
LOG_DIR="${BASE_DIR}/logs"

mkdir -p "$BASE_DIR"
mkdir -p "$PLOTS_DIR"
mkdir -p "$LOG_DIR"

echo "======================================================================" | tee "$SUMMARY_FILE"
echo " RADiCAL Multi-Energy Sweep Summary"                                   | tee -a "$SUMMARY_FILE"
echo " Date    : $(date)"                                                     | tee -a "$SUMMARY_FILE"
echo " Threads : ${THREADS} per run  |  Runs : ${N_RUNS}  |  Particles : ${N_PARTICLES}" | tee -a "$SUMMARY_FILE"
echo " Total cores in use per energy : $((THREADS * N_RUNS))"                | tee -a "$SUMMARY_FILE"
echo "======================================================================" | tee -a "$SUMMARY_FILE"
echo ""
echo "Sweep directory : ${BASE_DIR}"
echo "Optical physics : ${OPTICAL}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — wait on a list of PIDs, exit non-zero if any child failed
# ─────────────────────────────────────────────────────────────────────────────
wait_all() {
    local pids=("$@")
    local status=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "  [ERROR] Child process $pid exited with error." | tee -a "$SUMMARY_FILE"
            status=1
        fi
    done
    return $status
}

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
for ENERGY_KEV in "${ENERGIES_KEV[@]}"; do
    OUT_DIR="${BASE_DIR}/${ENERGY_KEV}keV"
    mkdir -p "$OUT_DIR"

    echo "======================================================================" | tee -a "$SUMMARY_FILE"
    echo "  ENERGY : ${ENERGY_KEV} keV"                                          | tee -a "$SUMMARY_FILE"
    echo "======================================================================" | tee -a "$SUMMARY_FILE"

    # ── Launch all runs in parallel ───────────────────────────────────────────
    pids=()
    for i in $(seq 0 $((N_RUNS - 1))); do
        LOG_FILE="${LOG_DIR}/${ENERGY_KEV}keV_run${i}.log"
        echo "  Launching run ${i} → log: ${LOG_FILE}"

        python3 simulator.py \
            --world       "$WORLD"       \
            --particle    "$PARTICLE"    \
            --energy-kev  "$ENERGY_KEV"  \
            --n           "$N_PARTICLES" \
            --threads     "$THREADS"     \
            --beam-radius "$BEAM_RADIUS" \
            --optical     "$OPTICAL"     \
            --run-id      "$i"           \
            --output-dir  "$OUT_DIR"     \
            > "$LOG_FILE" 2>&1 &

        pids+=($!)
    done

    # ── Wait for all runs at this energy ─────────────────────────────────────
    echo "  Waiting for ${N_RUNS} runs to complete..."
    if ! wait_all "${pids[@]}"; then
        echo "  [WARNING] One or more runs failed for ${ENERGY_KEV} keV — check logs in ${LOG_DIR}."
        echo "  Skipping analysis for this energy."
        echo ""
        continue
    fi
    echo "  All runs complete for ${ENERGY_KEV} keV."
    echo ""

    # ── Analysis ──────────────────────────────────────────────────────────────
    echo "  Running standard analysis..."
    python3 analyze.py --batch-dir "$OUT_DIR" | tee -a "$SUMMARY_FILE"

    echo "  Reconstructing timing resolution..."
    python3 timing_res.py --batch-dir "$OUT_DIR" | tee -a "$SUMMARY_FILE"

    echo "  Rendering 3D visualisation..."
    python3 plot_3d.py --batch-dir "$OUT_DIR"

    # ── Collect plots ─────────────────────────────────────────────────────────
    if ls "$OUT_DIR"/*.png 1>/dev/null 2>&1; then
        for plot in "$OUT_DIR"/*.png; do
            filename=$(basename "$plot")
            cp "$plot" "${PLOTS_DIR}/${ENERGY_KEV}keV_${filename}"
        done
        echo "  Plots copied to ${PLOTS_DIR}/"
    fi

    echo ""
done

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
echo "======================================================================" | tee -a "$SUMMARY_FILE"
echo " Sweep complete : $(date)"                                               | tee -a "$SUMMARY_FILE"
echo " Summary : $SUMMARY_FILE"                                                | tee -a "$SUMMARY_FILE"
echo " Plots   : $PLOTS_DIR"                                                   | tee -a "$SUMMARY_FILE"
echo " Logs    : $LOG_DIR"                                                     | tee -a "$SUMMARY_FILE"
echo "======================================================================" | tee -a "$SUMMARY_FILE"