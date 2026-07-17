#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim.sh — OpenGATE Simulation Launcher (Optimized for 50-Core Parallelism)
# ─────────────────────────────────────────────────────────────────────────────

WORLD="quartz_cal"
PARTICLE="proton"
ENERGY_KEV=1000000000
N_PARTICLES=20
THREADS=1                 # CPU threads per run execution
N_RUNS=1
BEAM_RADIUS=0.01
OPTICAL="on"
SIPM_HITS="on"
PHYSICS_LIST="QGSP_BERT_EMV"

IRON_START=1
IRON_END=70
IRON_STEP=1

# --- Parallelization Config ---
MAX_CONCURRENT_SIMS=5   
MAX_CONCURRENT_ANALYSIS=5 # Analysis is likely single-threaded, use all 50
# ─────────────────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_OUT_DIR="runs/${WORLD}/${ENERGY_KEV}keV_${TIMESTAMP}"
SIM_JOBS_FILE="${MASTER_OUT_DIR}/jobs_sim.txt"
ANALYSIS_JOBS_FILE="${MASTER_OUT_DIR}/jobs_analysis.txt"

mkdir -p "$MASTER_OUT_DIR"

echo "============================================================"
echo " Launching Parallel OpenGATE Simulation Stack"
echo "============================================================"
echo " Target Geometry   : ${WORLD}"
echo " Iron Sweep        : ${IRON_START}cm to ${IRON_END}cm (Step: ${IRON_STEP}cm)"
echo " Primaries/Run     : ${N_PARTICLES} (across ${THREADS} threads)"
echo " Runs / Thickness  : ${N_RUNS}"
echo " Master Batch Dir  : ${MASTER_OUT_DIR}"
echo " Concurrent Sims   : ${MAX_CONCURRENT_SIMS} (Using ~48 cores)"
echo "============================================================"
echo " Building job queues..."

# 1. Build the Job Lists
for IRON_CM in $(seq $IRON_START $IRON_STEP $IRON_END); do
    OUT_DIR="${MASTER_OUT_DIR}/iron_${IRON_CM}cm"
    mkdir -p "$OUT_DIR"
    
    # Queue up simulations
    for i in $(seq 0 $((N_RUNS - 1))); do
        echo "python3 simulator.py --world '$WORLD' --particle '$PARTICLE' --energy-kev $ENERGY_KEV --iron-cm $IRON_CM --n $N_PARTICLES --threads $THREADS --beam-radius $BEAM_RADIUS --optical '$OPTICAL' --sipm-hits '$SIPM_HITS' --run-id $i --physics-list '$PHYSICS_LIST' --output-dir '$OUT_DIR' > '$OUT_DIR/sim_log_${i}.txt' 2>&1" >> "$SIM_JOBS_FILE"
    done

    # Queue up analysis (these rely on the batch directory, so they run after sims)
    ANALYSIS_CMD=""
    if [ -f "analyze.py" ]; then
        ANALYSIS_CMD="python3 analyze.py --batch-dir '$OUT_DIR'"
    fi
    if [ -f "plot_3d.py" ]; then
        if [ -n "$ANALYSIS_CMD" ]; then ANALYSIS_CMD="$ANALYSIS_CMD && "; fi
        ANALYSIS_CMD="${ANALYSIS_CMD}python3 plot_3d.py --batch-dir '$OUT_DIR'"
    fi
    
    if [ -n "$ANALYSIS_CMD" ]; then
        echo "$ANALYSIS_CMD > '$OUT_DIR/analysis_log.txt' 2>&1" >> "$ANALYSIS_JOBS_FILE"
    fi
done

# 2. Execute Simulations in Parallel
echo " [+] Starting ${MAX_CONCURRENT_SIMS}-way parallel simulation phase..."
# xargs reads the job file and runs MAX_CONCURRENT_SIMS in parallel
xargs -a "$SIM_JOBS_FILE" -P $MAX_CONCURRENT_SIMS -I {} bash -c "{}"

echo " [+] All simulations complete. Starting analysis phase..."

# 3. Execute Per-Thickness Analysis in Parallel
if [ -s "$ANALYSIS_JOBS_FILE" ]; then
    echo " [+] Running per-thickness analysis (${MAX_CONCURRENT_ANALYSIS} concurrent)..."
    xargs -a "$ANALYSIS_JOBS_FILE" -P $MAX_CONCURRENT_ANALYSIS -I {} bash -c "{}"
else
    echo " [INFO] No per-thickness analysis scripts found or queued."
fi

# 4. Final Cross-Sweep Analysis (must be sequential at the end)
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