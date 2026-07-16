#!/bin/bash
WORLD=luagce_rc_hex_triple
PARTICLE=e-
ENERGY_KEV=20000000
N_PARTICLES=100
N_RUNS=10
BEAM_RADIUS=0.01
OPTICAL="on"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="runs/${WORLD}/${ENERGY_KEV}keV_${TIMESTAMP}"
LOG_DIR="logs/${WORLD}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Starting ${N_RUNS} runs → ${OUT_DIR}"
echo "Logs → ${LOG_DIR}"

pids=()
for i in $(seq 0 $((N_RUNS - 1))); do
 python3 simulator.py --beam-x 0.00000 --beam-y 0.35000 \
        --world       $WORLD \
        --particle    $PARTICLE \
        --energy-kev  $ENERGY_KEV \
        --n           $N_PARTICLES \
        --threads     1 \
        --beam-radius $BEAM_RADIUS \
        --optical     $OPTICAL \
        --run-id      $i \
        --output-dir  $OUT_DIR \
        > "${LOG_DIR}/run_${i}.log" 2>&1 &
    pids+=($!)
    echo "  Run ${i} started (PID ${pids[-1]})"
done

echo "Waiting for all ${N_RUNS} runs..."
for pid in "${pids[@]}"; do wait $pid; done

echo "Done. Analysing..."
python3 analyze.py     --batch-dir $OUT_DIR
python3 energy_calc.py --batch-dir $OUT_DIR
python3 plot_3d.py     --batch-dir $OUT_DIR
echo "Results in: $OUT_DIR"
