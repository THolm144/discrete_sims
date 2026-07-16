#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim_sweep.sh — OpenGATE simulation launcher (Organized Multi-Energy)
# ─────────────────────────────────────────────────────────────────────────────
#TEST TEXT
WORLD="luagce_radi_cal_energy"
PARTICLE="e-"
# Adjusted energies to match some of the paper's test beam data points
ENERGIES_KEV=(25000000 50000000 100000000 150000000) 
N_PARTICLES=50 
THREADS=8
N_RUNS=1
BEAM_RADIUS=0.01
OPTICAL="on"

# ─────────────────────────────────────────────────────────────────────────────
# DIRECTORY SETUP
# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="runs/${WORLD}/sweep_${TIMESTAMP}"
SUMMARY_FILE="${BASE_DIR}/sweep_summary.txt"
PLOTS_DIR="${BASE_DIR}/compiled_plots"

# Create root directories
mkdir -p "$BASE_DIR"
mkdir -p "$PLOTS_DIR"

# Initialize master summary file
echo "======================================================================" | tee -a "$SUMMARY_FILE"
echo " RADiCAL Multi-Energy Sweep Summary" | tee -a "$SUMMARY_FILE"
echo " Date: $(date)" | tee -a "$SUMMARY_FILE"
echo "======================================================================" | tee -a "$SUMMARY_FILE"

echo "Starting multi-energy sweep → ${BASE_DIR}"
echo "Optical Physics: ${OPTICAL}"

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────────────────────────────────────
for ENERGY_KEV in "${ENERGIES_KEV[@]}"; do
    OUT_DIR="${BASE_DIR}/${ENERGY_KEV}keV"
    
    echo "──────────────────────────────────────────────────────────────────────"
    echo "  RUNNING ENERGY: ${ENERGY_KEV} keV"
    echo "──────────────────────────────────────────────────────────────────────"

    # Run Simulation
    for i in $(seq 0 $((N_RUNS - 1))); do
        echo "  Run ${i}..."
 python3 simulator.py --beam-x -0.37032 --beam-y 0.37032   \
            --world       $WORLD \
            --particle    $PARTICLE \
            --energy-kev  $ENERGY_KEV \
            --n           $N_PARTICLES \
            --threads     $THREADS \
            --beam-radius $BEAM_RADIUS \
            --optical     $OPTICAL \
            --run-id      $i \
            --output-dir  $OUT_DIR
    done

    # Run Standard Analysis
    echo "Analysing ${ENERGY_KEV} keV standard run data..."
    python3 analyze.py --batch-dir "$OUT_DIR" | tee -a "$SUMMARY_FILE"

    # Run Timing Resolution Analysis
    echo "Reconstructing Timing Resolution for ${ENERGY_KEV} keV..."
    python3 timing_res.py --batch-dir "$OUT_DIR" | tee -a "$SUMMARY_FILE"

    # NEW: Run Energy Calculation & LYSO Histogram
    echo "Calculating Theoretical Energy & LYSO Histogram for ${ENERGY_KEV} keV..."
    python3 energy_calc.py --batch-dir "$OUT_DIR" | tee -a "$SUMMARY_FILE"

    # Generate 3D Visualization
    echo "Rendering 3D visualisation..."
    python3 plot_3d.py --batch-dir "$OUT_DIR"
    
    # ─────────────────────────────────────────────────────────────────────────
    # FILE CONSOLIDATION
    # ─────────────────────────────────────────────────────────────────────────
    # Copy any generated plots into the master plots folder, prefixing them 
    # with the energy so they don't overwrite each other.
    if ls "$OUT_DIR"/*.png 1> /dev/null 2>&1; then
        for plot in "$OUT_DIR"/*.png; do
            filename=$(basename "$plot")
            cp "$plot" "${PLOTS_DIR}/${ENERGY_KEV}keV_${filename}"
        done
    fi

done

echo "======================================================================" | tee -a "$SUMMARY_FILE"
echo "All energies complete."
echo "Master Summary compiled at: $SUMMARY_FILE"
echo "All plots saved neatly in:  $PLOTS_DIR"
