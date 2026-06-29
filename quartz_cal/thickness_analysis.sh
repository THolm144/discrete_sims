#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_sim.sh — OpenGATE Simulation Launcher for Iron-Quartz Array
# ─────────────────────────────────────────────────────────────────────────────

WORLD="quartz_cal"         # Target the NxN Quartz crystal / SiPM scanner array
PARTICLE="proton"          # Primary beam particle type
ENERGY_KEV=1000000000      # Beam energy in keV (1000000000 keV = 1 TeV)
N_PARTICLES=100            # Total primary particles per run
THREADS=8                 # CPU threads per run execution
N_RUNS=5                   # Number of independent simulation runs per thickness
BEAM_RADIUS=0.01           # Beam disc radius in cm
OPTICAL="on"               # ENABLES full optical photon transportation tracking
SIPM_HITS="on"             # ENABLES PhaseSpace tracking inside active SiPM volumes
PHYSICS_LIST="QGSP_BERT_EMV" # Hadronic + standard EM physics base

# --- Iron Thickness Loop Configurations ---
IRON_START=1               # Starting thickness in cm
IRON_END=70                # Ending thickness in cm
IRON_STEP=1                # Step size in cm
# ─────────────────────────────────────────────────────────────────────────────

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_OUT_DIR="runs/${WORLD}/${ENERGY_KEV}keV_${TIMESTAMP}"

echo "============================================================"
echo " Launching Modular OpenGATE Simulation Stack (Parameter Sweep)"
echo "============================================================"
echo " Target Geometry   : ${WORLD}"
echo " Iron Sweep        : ${IRON_START}cm to ${IRON_END}cm (Step: ${IRON_STEP}cm)"
echo " Primaries/Run     : ${N_PARTICLES} (across ${THREADS} threads)"
echo " Runs / Thickness  : ${N_RUNS}"
echo " Master Batch Dir  : ${MASTER_OUT_DIR}"
echo "============================================================"

# Outer loop: Iterate over iron thicknesses
for IRON_CM in $(seq $IRON_START $IRON_STEP $IRON_END); do
    
    # Create a specific output directory for this thickness
    OUT_DIR="${MASTER_OUT_DIR}/iron_${IRON_CM}cm"
    
    echo ""
    echo " ------------------------------------------------------------"
    echo " [+] Starting Sweep Iteration: Iron Shield = ${IRON_CM} cm"
    echo " ------------------------------------------------------------"

    # Inner loop: Execute sequential runs with distinct random seeds
    for i in $(seq 0 $((N_RUNS - 1))); do
        echo "   -> Executing Run [${i}/${N_RUNS}] for ${IRON_CM}cm shield..."
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

    echo "   -> Simulation runs for ${IRON_CM}cm complete. Running analysis..."

    if [ -f "analyze.py" ]; then
        python3 analyze.py --batch-dir "$OUT_DIR"
    else
        echo "   [INFO] analyze.py script not found. Skipping per-run analysis."
    fi

    if [ -f "plot_3d.py" ]; then
        echo "   [INFO] Rendering 3D visualization primitives..."
        python3 plot_3d.py --batch-dir "$OUT_DIR"
    fi

done

echo ""
echo "============================================================"
echo " Cross-Sweep Analysis"
echo "============================================================"

if [ -f "thickness_analysis.py" ]; then
    echo " [INFO] Running cross-thickness hit analysis..."
    python3 thickness_analysis.py --master-dir "$MASTER_OUT_DIR"
else
    echo " [INFO] thickness_analysis.py not found. Skipping histogram generation."
fi

echo "============================================================"
echo " Pipeline Complete. All sweep data stored in:"
echo " $MASTER_OUT_DIR"
echo "============================================================"