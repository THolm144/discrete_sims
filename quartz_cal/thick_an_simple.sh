#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim.sh — OpenGATE Simulation Launcher (Parallelized Version)
# ─────────────────────────────────────────────────────────────────────────────

WORLD="quartz_cal"
PARTICLE="proton"
ENERGY_KEV=1000000000
N_PARTICLES=300
THREADS=1                  # 1 thread per simulation instance (single-threaded)
PARALLEL_RUNS=300          # 300 parallel runs per batch
REPEATS=3                  # Total batches per thickness (1 run + 2 repeats)
BEAM_RADIUS=0.01
OPTICAL="on"
SIPM_HITS="on"
DOSE="off"                 # Fixed syntax error (removed spaces around =)
PHYSICS_LIST="QGSP_BERT_EMV"

IRON_START=1
IRON_END=70
IRON_STEP=1

# ─────────────────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_OUT_DIR="runs/${WORLD}/${ENERGY_KEV}keV_${TIMESTAMP}"

mkdir -p "$MASTER_OUT_DIR"

echo "============================================================"
echo " Launching Parallel OpenGATE Simulation Stack"
echo "============================================================"
echo " Target Geometry   : ${WORLD}"
echo " Iron Sweep        : ${IRON_START}cm to ${IRON_END}cm (Step: ${IRON_STEP}cm)"
echo " Primaries/Run     : ${N_PARTICLES}"
echo " Mode              : Single-Threaded (${THREADS} thread/instance)"
echo " Parallel Batch    : ${PARALLEL_RUNS} concurrent instances"
echo " Repeats / Width   : ${REPEATS}"
echo " Master Batch Dir  : ${MASTER_OUT_DIR}"
echo "============================================================"

# Loop through each thickness (width) sequentially
for IRON_CM in $(seq $IRON_START $IRON_STEP $IRON_END); do
    OUT_DIR="${MASTER_OUT_DIR}/iron_${IRON_CM}cm"
    mkdir -p "$OUT_DIR"
    
    echo ""
    echo " [+] Processing thickness: ${IRON_CM}cm"

    # Repeat loop: 3 iterations per width
    for REPEAT in $(seq 1 $REPEATS); do
        echo "     -> Launching batch ${REPEAT}/${REPEATS} (${PARALLEL_RUNS} parallel single-threaded runs)..."
        
        # Folder per repeat to prevent overwriting outputs
        REPEAT_DIR="${OUT_DIR}/repeat_${REPEAT}"
        mkdir -p "$REPEAT_DIR"

        # Spawn 300 single-threaded processes in the background
        for i in $(seq 0 $((PARALLEL_RUNS - 1))); do
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
                --output-dir "$REPEAT_DIR" > "$REPEAT_DIR/sim_log_${i}.txt" 2>&1 &
        done

        # Wait for all 300 background runs in this batch to finish before continuing
        wait
        echo "        Finished batch ${REPEAT}/${REPEATS} for ${IRON_CM}cm."
    done

    # Execute local analysis for this thickness after all 3 repeats complete
    echo "     -> Running local analysis for ${IRON_CM}cm..."
    
    ANALYSIS_LOG="$OUT_DIR/analysis_log.txt"
    touch "$ANALYSIS_LOG"

    if [ -f "analyze.py" ]; then
        python3 analyze.py --batch-dir "$OUT_DIR" >> "$ANALYSIS_LOG" 2>&1
        ANALYSIS_STATUS=$?
    else
        ANALYSIS_STATUS=0
    fi

    if [ -f "plot_3d.py" ] && [ $ANALYSIS_STATUS -eq 0 ]; then
        python3 plot_3d.py --batch-dir "$OUT_DIR" >> "$ANALYSIS_LOG" 2>&1
    fi
done

# Final Cross-Sweep Analysis
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