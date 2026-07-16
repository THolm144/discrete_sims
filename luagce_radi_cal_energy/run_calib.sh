#!/bin/bash
# Parallel simulation sweeping the Z-axis
OFFSETS=(-14.0 -12.5 -11.0 -9.5 -8.0 -6.5 -5.0 -3.5 -2.0 -0.5 1.0 2.5 4.0 5.5 7.0 8.5 10.0 11.5 13.0 14.5)

echo "Launching 20 parallel OpenGATE calibration runs for world: luagce_radi_cal_energy"

for offset in "${OFFSETS[@]}"; do
    OUT_DIR="./calib_runs/offset_${offset}"
    mkdir -p "$OUT_DIR"
    
    python simulator.py \
        --world luagce_radi_cal_energy \
        --particle opticalphoton \
        --energy-kev 0.000003 \
        --n 50000 \
        --threads 1 \
        --beam-radius 0.0 \
        --beam-offset "$offset" \
        --optical on \
        --hits-optical-only on \
        --output-dir "$OUT_DIR" > "${OUT_DIR}/sim.log" 2>&1 &
done

wait
echo "All calibration runs complete!"
