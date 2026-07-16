#!/bin/bash
WORLD="luagce_rc_hex_triple"
PARTICLE="e-"
BEAM_RADIUS=0.01
OPTICAL="on"
CHERENKOV="off"
PHYSICS_LIST="QGSP_BERT_EMV"

# Dynamically balanced totals calculated for a 16-hour execution safety window
# Spawns 1 process per energy x 10 internal C++ threads = 40 cores maxed out
ENERGIES=(25000000 50000000 70000000 90000000)
COUNTS=(1850 935 660 510)
THREADS=10

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_BATCH_DIR="runs/${WORLD}/sweep_${TIMESTAMP}"
mkdir -p "$MASTER_BATCH_DIR/logs"

echo "========================================================================"
echo " Launching Dynamic Optical Sweep (MT Mode) | Resource Cap: 40 CPUs"
echo "========================================================================"

for i in "${!ENERGIES[@]}"; do
    ENERGY=${ENERGIES[$i]}
    N_EVENTS=${COUNTS[$i]}
    ENERGY_GBS=$(( ENERGY / 1000000 ))
    ENERGY_DIR="${MASTER_BATCH_DIR}/${ENERGY_GBS}GeV"
    mkdir -p "$ENERGY_DIR"
    LOG_FILE="${MASTER_BATCH_DIR}/logs/${ENERGY_GBS}GeV_production.log"

    echo " [+] Launching [${ENERGY_GBS}GeV] -> Target: ${N_EVENTS} events on 10 threads..."

    python3 simulator.py --beam-x -0.37032 --beam-y 0.37032 \
        --world "$WORLD" --particle "$PARTICLE" --energy-kev "$ENERGY" \
        --n "$N_EVENTS" --threads "$THREADS" --beam-radius "$BEAM_RADIUS" \
        --optical "$OPTICAL" --cherenkov "$CHERENKOV" --hits-optical-only on \
        --physics-list "$PHYSICS_LIST" --run-id 0 --output-dir "$ENERGY_DIR" > "$LOG_FILE" 2>&1 &
done

echo " [+] All 4 channels running. Monitoring workloads..."
wait
echo " [✓] Simulation sweep complete. Running sequential analysis pipeline..."
# (Post processing cleanup block goes here...)