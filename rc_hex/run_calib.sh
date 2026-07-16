#!/bin/bash
# Parallel simulation sweeping the Z-axis safely within active boundaries
OFFSETS=(-6.0 -5.0 -4.0 -3.0 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0)

echo "Launching 13 parallel OpenGATE calibration runs for world: rc_hex"
echo "Targeting Capillary coordinate: X = 0.0 cm, Y = 0.35 cm"

for offset in "${OFFSETS[@]}"; do
    OUT_DIR="./calib_runs/offset_${offset}"
    mkdir -p "$OUT_DIR"
    
    python3 simulator.py \
        --world rc_hex \
        --particle opticalphoton \
        --energy-kev 0.003 \
        --n 50000 \
        --threads 1 \
        --beam-radius 0.0 \
        --beam-offset "$offset" \
        --beam-x 0.0 \
        --beam-y 0.35 \
        --optical on \
        --hits-optical-only on \
        --output-dir "$OUT_DIR" > "${OUT_DIR}/sim.log" 2>&1 &
done

wait
echo "All calibration runs complete!"
