#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim.sh — OpenGATE simulation launcher
# ─────────────────────────────────────────────────────────────────────────────
WORLD=luagce_rc_hex
PARTICLE=e-
ENERGY_KEV=150000000
N_PARTICLES=1
THREADS=1
N_RUNS=1
PARALLEL=1
BEAM_RADIUS=0.01
OPTICAL="on"
CHERENKOV="off"
PHYSICS_LIST="QGSP_BERT_EMV"
# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="runs/${WORLD}/${ENERGY_KEV}keV_${TIMESTAMP}"
mkdir -p "$OUT_DIR"

echo "Starting ${N_RUNS} runs (${PARALLEL} parallel) → ${OUT_DIR}"
echo "Total particles: $((N_RUNS * N_PARTICLES))"
echo "Optical Physics: ${OPTICAL}  |  Cherenkov: ${CHERENKOV}"

# Keep track of individual process IDs to monitor them reliably
PIDS=()

for i in $(seq 0 $((N_RUNS - 1))); do
    echo "  Launching run ${i}..."
    python3 simulator.py \
        --world "$WORLD" \
        --particle "$PARTICLE" \
        --energy-kev "$ENERGY_KEV" \
        --n "$N_PARTICLES" \
        --threads "$THREADS" \
        --beam-radius "$BEAM_RADIUS" \
        --optical "$OPTICAL" \
        --cherenkov "$CHERENKOV" \
        --physics-list "$PHYSICS_LIST" \
        --run-id "$i" \
        --output-dir "${OUT_DIR}" &
    
    PIDS+=($!)

    if (( (i + 1) % PARALLEL == 0 )); then
        echo "  [Waiting for batch $((( i + 1) / PARALLEL)) to complete...]"
        wait
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# LIVE ROOT FILE TRACKING MONITOR
# ─────────────────────────────────────────────────────────────────────────────
echo "Simulations are executing. Monitoring ROOT file output..."
TOTAL_EXPECTED=$N_RUNS

while true; do
    # Count completed ROOT files
    CURRENT_COUNT=$(find "$OUT_DIR" -name "detector_hits*.root" 2>/dev/null | wc -l)
    PCT=$(( CURRENT_COUNT * 100 / TOTAL_EXPECTED ))
    
    # Dynamic live line display
    printf "\r[SIMULATION PROGRESS]: %d/%d ROOT files written (%d%%) " "$CURRENT_COUNT" "$TOTAL_EXPECTED" "$PCT"
    
    # Break if everything is written out
    if [ "$CURRENT_COUNT" -ge "$TOTAL_EXPECTED" ]; then
        break
    fi
    
    # Robust check: Count how many of our launched PIDS are still alive in the OS process tree
    STILL_ALIVE=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            ((STILL_ALIVE++))
        fi
    done
    
    # If all processes have died, exit the loop cleanly
    if [ "$STILL_ALIVE" -eq 0 ]; then
        break
    fi

    sleep 2
done
printf "\n"

# Final safety sync
wait
echo "All runs complete and ROOT files flushed to disk."

echo "Analysing standard dose and hits..."
python3 analyze.py --batch-dir $OUT_DIR --workers 64

echo "Calculating Theoretical Energy & Plotting LYSO Histogram..."
python3 energy_calc.py --batch-dir $OUT_DIR

echo "Rendering 3D visualisation..."
python3 plot_3d.py --batch-dir $OUT_DIR

echo "Results in: $OUT_DIR"