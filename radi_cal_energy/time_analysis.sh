#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim.sh — OpenGATE Simulation Launcher for Iron-Quartz Array
# ─────────────────────────────────────────────────────────────────────────────

WORLD="quartz_cal"  # Target the NxN Quartz crystal / SiPM scanner array
PARTICLE="proton"          # Primary beam particle type
ENERGY_KEV=1000000000       # Beam energy in keV (1000000000 keV = 1 TeV)
N_PARTICLES=100            # Total primary particles per run
THREADS=8                  # CPU threads per run execution
N_RUNS=5                   # Number of independent simulation runs
BEAM_RADIUS=0.01            # Beam disc radius in cm
IRON_CM=5.0                # <--- NEW: Thickness of the Iron Shield in cm
OPTICAL="on"               # ENABLES full optical photon transportation tracking
SIPM_HITS="on"             # ENABLES PhaseSpace tracking inside active SiPM volumes
PHYSICS_LIST="QGSP_BERT_EMV" # Hadronic + standard EM physics base
# ─────────────────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="runs/${WORLD}/${ENERGY_KEV}keV_${TIMESTAMP}"

echo "============================================================"
echo " Launching Modular OpenGATE Simulation Stack"
echo "============================================================"
echo " Target Geometry : ${WORLD}"
echo " Iron Shield     : ${IRON_CM} cm"
echo " Primaries/Run   : ${N_PARTICLES} (across ${THREADS} threads)"
echo " Optical Physics : ${OPTICAL} (Cerenkov/Scintillation active)"
echo " SiPM Collection : ${SIPM_HITS}"
echo " Batch Directory : ${OUT_DIR}"
echo "============================================================"

# Loop through and execute sequential runs with distinct random seeds
for i in $(seq 0 $((N_RUNS - 1))); do
    echo " -> Executing Run [${i}/${N_RUNS}]..."
    python3 simulator.py \
        --world       "$WORLD" \
        --particle    "$PARTICLE" \
        --energy-kev  $ENERGY_KEV \
        --iron-cm     $IRON_CM \
        --n           $N_PARTICLES \
        --threads     $THREADS \
        --beam-radius $BEAM_RADIUS \
        --optical     "$OPTICAL" \
        --sipm-hits   "$SIPM_HITS" \
        --run-id      $i \
        --physics-list "$PHYSICS_LIST" \
        --output-dir  "$OUT_DIR"
done

echo "------------------------------------------------------------"
echo " Simulation runs complete. Launching analysis workflow..."
echo "------------------------------------------------------------"

if [ -f "analyze.py" ]; then
    python3 analyze.py --batch-dir "$OUT_DIR"
else
    echo " [INFO] analyze.py script not found in root directory. Skipping analysis."
fi

if [ -f "plot_3d.py" ]; then
    echo " [INFO] Rendering 3D visualization primitives..."
    python3 plot_3d.py --batch-dir "$OUT_DIR"
fi

echo "============================================================"
echo " Pipeline Complete. Output data stored in: $OUT_DIR"
echo "================================────────────────============"