#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim.sh — OpenGATE Simulation Launcher (Sequential Version)
# ─────────────────────────────────────────────────────────────────────────────

WORLD="quartz_cal"
PARTICLE="proton"
ENERGY_KEV=1000000000
N_PARTICLES=1
THREADS=5                 # CPU threads per run execution
N_RUNS=5
BEAM_RADIUS=0.01
OPTICAL="on"
SIPM_HITS="on"
DOSE = "off"
PHYSICS_LIST="QGSP_BERT_EMV"

IRON_START=1
IRON_END=70
IRON_STEP=1

# ─────────────────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_OUT_DIR="runs/${WORLD}/${ENERGY_KEV}keV_${TIMESTAMP}"

mkdir -p "$MASTER_OUT_DIR"

echo "============================================================"
echo " Launching Sequential OpenGATE Simulation Stack"
echo "============================================================"
echo " Target Geometry   : ${WORLD}"
echo " Iron Sweep        : ${IRON_START}cm to ${IRON_END}cm (Step: ${IRON_STEP}cm)"
echo " Primaries/Run     : ${N_PARTICLES} (across ${THREADS} threads)"
echo " Runs / Thickness  : ${N_RUNS}"
echo " Master Batch Dir  : ${MASTER_OUT_DIR}"
echo " Mode              : Pure Sequential (Single-threaded Execution)"
echo "============================================================"

# Loop through each thickness step sequentially
for IRON_CM in $(seq $IRON_START $IRON_STEP $IRON_END); do
    OUT_DIR="${MASTER_OUT_DIR}/iron_${IRON_CM}cm"
    mkdir -p "$OUT_DIR"
    
    echo " [+] Processing thickness: ${IRON_CM}cm"

    # 1. Execute simulations sequentially for this thickness
    echo "     -> Running simulations..."
    for i in $(seq 0 $((N_RUNS - 1))); do
        python3 simulator.py \
            --world "$WORLD" \
            --particle "$PARTICLE" \
            --energy-kev $ENERGY_KEV \
            --iron-cm $IRON_CM \
            --n $N_PARTICLES \
            --threads $THREADS \
            --beam-radius $BEAM_RADIUS \
            --optical "$OPTICAL" \
            --sipm-hits "$SIPM_HITS" \
            --dose "$DOSE" \
            --run-id $i \
            --physics-list "$PHYSICS_LIST" \
            --output-dir "$OUT_DIR" > "$OUT_DIR/sim_log_${i}.txt" 2>&1
    done

    # 2. Execute analysis scripts sequentially for this thickness
    echo "     -> Running local analysis..."
    
    # Initialize log file for analysis
    ANALYSIS_LOG="$OUT_DIR/analysis_log.txt"
    touch "$ANALYSIS_LOG"

    if [ -f "analyze.py" ]; then
        python3 analyze.py --batch-dir "$OUT_DIR" >> "$ANALYSIS_LOG" 2>&1
        ANALYSIS_STATUS=$?
    else
        ANALYSIS_STATUS=0
    fi

    # Only run plot_3d.py if analyze.py succeeded (mimicking the original && behavior)
    if [ -f "plot_3d.py" ] && [ $ANALYSIS_STATUS -eq 0 ]; then
        python3 plot_3d.py --batch-dir "$OUT_DIR" >> "$ANALYSIS_LOG" 2>&1
    fi
done

# 3. Final Cross-Sweep Analysis (remains sequential at the end)
echo ""
echo "============================================================"
echo " Cross-Sweep Analysis"
echo "============================================================"

if [ -f "thickness_analysis.py" ]; then
    echo " [INFO] Running cross-thickness hit analysis..."
    python3 thickness_analysis.py --master-dir "$MASTER_OUT_DIR"
else
    echo " [INFO] thickness_analysis.py not found. Skipping."
fi

echo "============================================================"
echo " Pipeline Complete. All sweep data stored in:"
echo " $MASTER_OUT_DIR"
echo "============================================================"