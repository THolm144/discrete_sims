#!/bin/bash
# Parallel simulation sweeping the Z-axis safely within active boundaries
OFFSETS=(-10.0 -8.5 -7.0 -5.5 -4.0 -2.5 -1.0 0.5 2.0 3.5 5.0 6.5 8.0 9.5)

echo "Launching 14 parallel OpenGATE calibration runs for world: dsb1_radi_cal_triple"
echo "Targeting Capillary coordinate: X = 0.0 cm, Y = 0.0 cm"

for offset in "${OFFSETS[@]}"; do
    OUT_DIR="./calib_runs/offset_${offset}"
    mkdir -p "$OUT_DIR"
    
    python3 simulator.py \
        --world dsb1_radi_cal_triple \
        --particle opticalphoton \
        --energy-kev 0.003 \
        --n 50000 \
        --threads 1 \
        --beam-radius 0.0 \
        --beam-offset "$offset" \
        --beam-x 0.0 \
        --beam-y 0.0 \
        --optical on \
        --hits-optical-only on \
        --output-dir "$OUT_DIR" > "${OUT_DIR}/sim.log" 2>&1 &
done

wait
echo "All calibration runs complete!"
