#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim.sh — OpenGATE simulation launcher (parallel, 15 cores)
# Usage: bash run_sim.sh
# ─────────────────────────────────────────────────────────────────────────────
WORLD=radi_cal_energy
PARTICLE=e-
ENERGY_KEV=20000000
N_PARTICLES=5
THREADS=1
N_RUNS=60
PARALLEL=15
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

for i in $(seq 0 $((N_RUNS - 1))); do
    echo "  Launching run ${i}..."
    setsid python3 simulator.py \
        --world        $WORLD \
        --particle     $PARTICLE \
        --energy-kev   $ENERGY_KEV \
        --n            $N_PARTICLES \
        --threads      $THREADS \
        --beam-radius  $BEAM_RADIUS \
        --optical      $OPTICAL \
        --cherenkov    $CHERENKOV \
        --physics-list "$PHYSICS_LIST" \
        --run-id       $i \
        --output-dir   $OUT_DIR &

    # Every PARALLEL launches, wait for the batch to finish
    if (( (i + 1) % PARALLEL == 0 )); then
        echo "  [Waiting for batch $((( i + 1) / PARALLEL)) to complete...]"
        wait
    fi
done

# Wait for any remaining runs in the last partial batch
wait
echo "All runs complete."

echo "Analysing standard dose and hits..."
python3 analyze.py --batch-dir $OUT_DIR

echo "Calculating Theoretical Energy & Plotting LYSO Histogram..."
python3 energy_calc.py --batch-dir $OUT_DIR

echo "Rendering 3D visualisation..."
python3 plot_3d.py --batch-dir $OUT_DIR

echo "Results in: $OUT_DIR"