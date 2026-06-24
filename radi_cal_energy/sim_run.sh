#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim.sh — OpenGATE simulation launcher
# Usage: bash run_sim.sh
# ─────────────────────────────────────────────────────────────────────────────
WORLD=radi_cal_energy
PARTICLE=e-
ENERGY_KEV=20000000
N_PARTICLES=5
THREADS=1
N_RUNS=200
BEAM_RADIUS=0.01
OPTICAL="on"
CHERENKOV="off"         # "on" or "off" — only applies when OPTICAL=on
PHYSICS_LIST="QGSP_BERT_EMV"
# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="runs/${WORLD}/${ENERGY_KEV}keV_${TIMESTAMP}"
echo "Starting ${N_RUNS} runs → ${OUT_DIR}"
echo "Optical Physics: ${OPTICAL}  |  Cherenkov: ${CHERENKOV}"
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
        --cherenkov   $CHERENKOV \
        --physics-list "$PHYSICS_LIST" \
        --run-id      $i \
        --output-dir  $OUT_DIR
done
echo "Done. Analysing standard dose and hits..."
python3 analyze.py --batch-dir $OUT_DIR
echo "Calculating Theoretical Energy & Plotting LYSO Histogram..."
python3 energy_calc.py --batch-dir $OUT_DIR
echo "Rendering 3D visualisation..."
python3 plot_3d.py --batch-dir $OUT_DIR
echo "Results in: $OUT_DIR"