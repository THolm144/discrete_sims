#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim_fast_sweep.sh — Hyper-Speed OpenGATE Sweep (Max 200 Threads)
# Includes complete Direct Timing and ToF Reconstruction pipelines.
# ─────────────────────────────────────────────────────────────────────────────

MAX_THREADS=200       
THREADS=16            # High Geant4 MT threads = low memory overhead

WORLD="radi_cal_energy"
PARTICLE="e-"
ENERGY_KEV=20000000   # Single energy target from your script
N_PARTICLES=1000      
N_RUNS=60             
BEAM_RADIUS=0.01
OPTICAL="on"
CHERENKOV="off"
PHYSICS_LIST="QGSP_BERT_EMV"

# Calculate optimal parallel jobs (200 / 16 = 12 parallel runs max)
PARALLEL_RUNS=$(( MAX_THREADS / THREADS ))
if [ $PARALLEL_RUNS -lt 1 ]; then PARALLEL_RUNS=1; fi

# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="runs/${WORLD}/sweep_${TIMESTAMP}"
SUMMARY_FILE="${BASE_DIR}/sweep_summary.txt"
PLOTS_DIR="${BASE_DIR}/compiled_plots"
LOG_DIR="${BASE_DIR}/logs"
OUT_DIR="${BASE_DIR}/${ENERGY_KEV}keV"

mkdir -p "$OUT_DIR"
mkdir -p "$PLOTS_DIR"
mkdir -p "$LOG_DIR"

echo "======================================================================" | tee "$SUMMARY_FILE"
echo " OpenGATE Hyper-Speed Sweep Summary (Thread Pool)"                     | tee -a "$SUMMARY_FILE"
echo " Date    : $(date)"                                                     | tee -a "$SUMMARY_FILE"
echo " Threads : ${THREADS} per run  |  Max Parallel Jobs : ${PARALLEL_RUNS}" | tee -a "$SUMMARY_FILE"
echo " Allocated Cores : $((THREADS * PARALLEL_RUNS)) / Max Target: ${MAX_THREADS}" | tee -a "$SUMMARY_FILE"
echo "======================================================================" | tee -a "$SUMMARY_FILE"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
echo "--> Starting simulation pool for ${ENERGY_KEV} keV..."

for i in $(seq 0 $((N_RUNS - 1))); do
    # Keep the window full at maximum allowed parallel processes
    while [ $(jobs -rp | wc -l) -ge $PARALLEL_RUNS ]; do
        sleep 0.2
    done

    LOG_FILE="${LOG_DIR}/${ENERGY_KEV}keV_run${i}.log"
    echo "  Launching run ${i}/${N_RUNS} → log: ${LOG_FILE}"

    setsid python3 simulator.py \
        --world        "$WORLD" \
        --particle     "$PARTICLE" \
        --energy-kev   "$ENERGY_KEV" \
        --n            "$N_PARTICLES" \
        --threads      "$THREADS" \
        --beam-radius  "$BEAM_RADIUS" \
        --optical      "$OPTICAL" \
        --cherenkov    "$CHERENKOV" \
        --physics-list "$PHYSICS_LIST" \
        --run-id       "$i" \
        --output-dir   "$OUT_DIR" > "$LOG_FILE" 2>&1 &
done

echo "  All jobs dispatched. Waiting for final active batch to clear..."
wait
echo "  All simulation runs complete for ${ENERGY_KEV} keV."
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
echo "  Analysing standard dose and hits..."
python3 analyze.py --batch-dir "$OUT_DIR" | tee -a "$SUMMARY_FILE"

echo "  Calculating Theoretical Energy & Plotting LYSO Histogram..."
python3 energy_calc.py --batch-dir "$OUT_DIR" | tee -a "$SUMMARY_FILE"

echo "  Reconstructing Pure Direct LocalTime Timing Resolution..."
python3 timing_res.py --batch-dir "$OUT_DIR" | tee -a "$SUMMARY_FILE"

echo "  Reconstructing ToF Longitudinal Kinematic Profile..."
python3 tof_reconstruction.py --batch-dir "$OUT_DIR" | tee -a "$SUMMARY_FILE"

echo "  Rendering 3D visualisation..."
python3 plot_3d.py --batch-dir "$OUT_DIR"

# ── Collect and Organize Output Plots ─────────────────────────────────────────
if ls "$OUT_DIR"/*.png 1>/dev/null 2>&1; then
    for plot in "$OUT_DIR"/*.png; do
        filename=$(basename "$plot")
        cp "$plot" "${PLOTS_DIR}/${ENERGY_KEV}keV_${filename}"
    done
    echo "  Plots successfully copied to central repository: ${PLOTS_DIR}/"
fi

echo "======================================================================" | tee -a "$SUMMARY_FILE"
echo " Sweep complete : $(date)"                                               | tee -a "$SUMMARY_FILE"
echo " Summary        : $SUMMARY_FILE"                                         | tee -a "$SUMMARY_FILE"
echo " Compiled Plots : $PLOTS_DIR"                                            | tee -a "$SUMMARY_FILE"
echo "======================================================================" | tee -a "$SUMMARY_FILE"