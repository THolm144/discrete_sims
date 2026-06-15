#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim.sh — OpenGATE simulation launcher
# Usage: bash run_sim.sh
# ─────────────────────────────────────────────────────────────────────────────

WORLD=quartz_cal    # world module name
PARTICLE=proton            # primary particle type (e.g., proton, electron, gamma)
ENERGY_KEV=50000        # energy in keV (200000 = 200 MeV)
N_PARTICLES=100        # particles per run
THREADS=8               # CPU threads per run
N_RUNS=5                # number of runs
BEAM_RADIUS=1.0         # beam radius in cm
OPTICAL="on"           # "on" for full optics, "off" for fast dose analysis
PHYSICS_LIST="QGSP_BERT_EMV"  # Physics config
# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="runs/${WORLD}/${ENERGY_KEV}keV_${TIMESTAMP}"
echo "Starting ${N_RUNS} runs → ${OUT_DIR}"
echo "Optical Physics: ${OPTICAL}"

for i in $(seq 0 $((N_RUNS - 1))); do
    echo "  Run ${i}..."
    python3 simulator.py \
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

echo "Done. Analysing..."
python3 analyze.py --batch-dir $OUT_DIR

echo "Rendering 3D visualisation..."
python3 plot_3d.py --batch-dir $OUT_DIR

echo "Results in: $OUT_DIR"