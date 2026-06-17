#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim.sh — OpenGATE simulation launcher
# Usage: bash run_sim.sh
# ─────────────────────────────────────────────────────────────────────────────
WORLD=scintx_sipm_array
PARTICLE=e-
ENERGY_KEV=9000
N_PARTICLES=100
THREADS=10
N_RUNS=5
BEAM_RADIUS=1.0
OPTICAL="on"
TRACK_OPTICAL="on"
PHYSICS_LIST="QGSP_BERT_EMV"
NO_CERENKOV="off"    # "on" to disable Cherenkov, "off" for normal
# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="runs/${WORLD}/${ENERGY_KEV}keV_${TIMESTAMP}"
CERENKOV_FLAG=$( [ "$NO_CERENKOV" = "on" ] && echo "--no-cerenkov" )

echo "Starting ${N_RUNS} runs → ${OUT_DIR}"
echo "Optical Physics: ${OPTICAL} (Tracking: ${TRACK_OPTICAL})"

for i in $(seq 0 $((N_RUNS - 1))); do
    echo "  Run ${i}..."
    python3 simulator.py \
        --world         $WORLD \
        --particle      $PARTICLE \
        --energy-kev    $ENERGY_KEV \
        --n             $N_PARTICLES \
        --threads       $THREADS \
        --beam-radius   $BEAM_RADIUS \
        --optical       $OPTICAL \
        --track-optical $TRACK_OPTICAL \
        --physics-list  $PHYSICS_LIST \
        --run-id        $i \
        --output-dir    $OUT_DIR \
        $CERENKOV_FLAG
done

echo "Done. Analysing..."
python3 analyze.py --batch-dir $OUT_DIR
echo "Rendering 3D visualisation..."
python3 plot_3d.py --batch-dir $OUT_DIR
echo "Results in: $OUT_DIR"