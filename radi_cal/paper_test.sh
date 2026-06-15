#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_paper_validation.sh — Automated RADiCAL Beam Test Replication Sweep
# Sweeps over electron beam energies: 25, 50, 75, 100, and 150 GeV
# ─────────────────────────────────────────────────────────────────────────────

WORLD=radi_cal            # World module name matching your setup
PARTICLE=e-               # Primary beam particle type used in the paper
N_PARTICLES=100          # Primary events per run
THREADS=8                 # CPU threads per run
N_RUNS=1                  # Number of sub-runs to average over
BEAM_RADIUS=0.2           # Beam radius in cm
OPTICAL="on"              # REQUIRED: Set to "on" to generate tracking & timing data
PHYSICS_LIST="FTFP_BERT"  # Base hadronic/EM physics template

# Array of the exact primary electron beam energies used in the paper
ENERGIES_GEV=(25 50 75 100 150)
# Converted to keV for simulator.py (e.g., 25 GeV = 25,000,000 keV)
ENERGIES_KEV=(25000000 50000000 75000000 100000000 150000000)

# Initialize a clean global text summary matrix
SUMMARY_FILE="paper_comparison_summary.txt"
echo "========================================================================" > $SUMMARY_FILE
echo "             RADiCAL SIMULATION TO PAPER COMPARISON REPORT             " >> $SUMMARY_FILE
echo "========================================================================" >> $SUMMARY_FILE
echo "Functional Target Scaling Fit: sigma_t = (a / sqrt(E)) (+) b" >> $SUMMARY_FILE
echo "Paper Constant Fit Parameters: a ~ 256 ps*sqrt(GeV), b ~ 17.5 ps" >> $SUMMARY_FILE
echo "------------------------------------------------------------------------" >> $SUMMARY_FILE
printf "%-12s | %-16s | %-16s | %-14s\n" "Energy (GeV)" "Sim sigma_t (ps)" "Paper Target (ps)" "Status" >> $SUMMARY_FILE
echo "------------------------------------------------------------------------" >> $SUMMARY_FILE

echo "Starting automated paper validation sweep..."
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Iterate through each energy point specified in the paper
for idx in "${!ENERGIES_GEV[@]}"; do
    GEV=${ENERGIES_GEV[$idx]}
    KEV=${ENERGIES_KEV[$idx]}
    
    # Isolate each energy into its own clean batch directory structure
    OUT_DIR="runs/${WORLD}/${KEV}keV_${TIMESTAMP}"
    
    echo ""
    echo "=================================================================="
    echo " RUNNING SWEEP POINT: ${GEV} GeV (${KEV} keV) | Optical: ${OPTICAL}"
    echo "=================================================================="
    
    # Execute loop across the sub-runs matching your verified framework
    for i in $(seq 0 $((N_RUNS - 1))); do
        echo "  -> Processing Run ${i} of ${N_RUNS}..."
        python3 simulator.py \
            --world       $WORLD \
            --particle    $PARTICLE \
            --energy-kev  $KEV \
            --n           $N_PARTICLES \
            --threads     $THREADS \
            --beam-radius $BEAM_RADIUS \
            --optical     $OPTICAL \
            --physics-list $PHYSICS_LIST \
            --run-id      $i \
            --output-dir  $OUT_DIR
    done

    echo "  Processing complete for ${GEV} GeV. Running analysis pipeline..."
    python3 analyze.py --batch-dir $OUT_DIR
    
    # Extract the simulated timing value from the newly built batch analysis report
    REPORT_PATH="${OUT_DIR}/batch_analysis.txt"
    if [ -f "$REPORT_PATH" ]; then
        SIM_SIGMA=$(grep "Timing resolution" "$REPORT_PATH" | awk -F':' '{print $2}' | sed 's/ ps//g' | xargs)
        
        # Pull static experimental values derived directly from the paper's timing curve fit
        if [ "$GEV" -eq 25 ]; then PAPER_TARGET="54.1";
        elif [ "$GEV" -eq 50 ]; then PAPER_TARGET="39.8";
        elif [ "$GEV" -eq 75 ]; then PAPER_TARGET="33.7";
        elif [ "$GEV" -eq 100 ]; then PAPER_TARGET="30.2";
        elif [ "$GEV" -eq 150 ]; then PAPER_TARGET="26.3";
        else PAPER_TARGET="N/A"; fi
        
        # Clean up empty or missing results string parsing
        if [ -z "$SIM_SIGMA" ] || [ "$SIM_SIGMA" = "N/A" ]; then
            printf "%-12s | %-16s | %-16s | %-14s\n" "${GEV}" "No Opt/Time Data" "${PAPER_TARGET}" "Check ROOT" >> $SUMMARY_FILE
        else
            printf "%-12s | %-16s | %-16s | %-14s\n" "${GEV}" "${SIM_SIGMA}" "${PAPER_TARGET}" "Logged" >> $SUMMARY_FILE
            echo "Successfully parsed ${GEV} GeV point: Sim=${SIM_SIGMA} ps vs. Paper ~${PAPER_TARGET} ps"
        fi
    else
        printf "%-12s | %-16s | %-16s | %-14s\n" "${GEV}" "ERROR" "${PAPER_TARGET}" "Report Missing" >> $SUMMARY_FILE
    fi
done

echo ""
echo "========================================================================"
echo " All beam test simulation configurations executed successfully!"
echo " Combined metric report saved to: ./${SUMMARY_FILE}"
echo "========================================================================"
cat $SUMMARY_FILE