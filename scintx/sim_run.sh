#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim.sh — Parallelized OpenGATE Task-Array Launcher (Optimized for 400 Cores)
# ─────────────────────────────────────────────────────────────────────────────

WORLD=scintx_sipm_array
PARTICLE=e-
ENERGY_KEV=9000        # 9 MeV FLEX-9 Target Energy
BEAM_RADIUS=1.0

# ── 400 CORES ALLOCATION PROFILE ──
THREADS_PER_RUN=16     # Sweet spot for Geant4 scaling
MAX_PARALLEL_JOBS=25   # 16 threads * 25 jobs = 400 Cores active simultaneously
N_RUNS=25              # Total jobs (25 jobs means 1 fully saturated wave)

# MODIFIED: Scaled down from 500k to 10k to achieve a ~1-3 hour runtime 
# with 400 active cores under heavy optical tracking overhead.
N_PARTICLES_PER_RUN=10000 

OPTICAL="on"
TRACK_OPTICAL="off"
PHYSICS_LIST="QGSP_BERT_EMV"
NO_CERENKOV="off"

# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="runs/${WORLD}/${ENERGY_KEV}keV_${TIMESTAMP}"
CERENKOV_FLAG=$( [ "$NO_CERENKOV" = "on" ] && echo "--no-cerenkov" )

mkdir -p "$OUT_DIR"
echo "=========================================================="
echo " Launching OpenGATE Array on 400 Cores"
echo " Target Directory : $OUT_DIR"
echo " Architecture     : ${MAX_PARALLEL_JOBS} concurrent jobs x ${THREADS_PER_RUN} threads"
echo " Primaries/Job    : ${N_PARTICLES_PER_RUN}"
echo "================================────────────────=========="

for i in $(seq 0 $((N_RUNS - 1))); do
    echo "  -> Spawning Task ${i}..."
    
    python3 simulator.py \
        --world         $WORLD \
        --particle      $PARTICLE \
        --energy-kev    $ENERGY_KEV \
        --n             $N_PARTICLES_PER_RUN \
        --threads       $THREADS_PER_RUN \
        --beam-radius   $BEAM_RADIUS \
        --optical       $OPTICAL \
        --track-optical $TRACK_OPTICAL \
        --physics-list  $PHYSICS_LIST \
        --run-id        $i \
        --output-dir    "$OUT_DIR" \
        $CERENKOV_FLAG & # ← Runs asynchronously in background

    # Core allocation check block: If we hit MAX_PARALLEL_JOBS, pause and wait for the batch to clear
    if [[ $(( (i + 1) % MAX_PARALLEL_JOBS )) -eq 0 ]]; then
        echo " [WAIT] Processing thread block wave. Awaiting completion..."
        wait
    fi
done

# Clear any lingering asymmetric processes
wait
echo "All parallel processes completed successfully. Beginning downstream aggregation..."

# ── POST PROCESSING CALLS ──
if [ -f "analyze.py" ]; then
    python3 analyze.py --batch-dir "$OUT_DIR"
fi

if [ -f "plot_3d.py" ]; then
    python3 plot_3d.py --batch-dir "$OUT_DIR"
fi

echo "Process Complete. Aggregated output in: $OUT_DIR"