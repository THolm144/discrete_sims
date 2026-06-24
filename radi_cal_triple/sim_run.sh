setsid bash -c '
WORLD=radi_cal_triple
PARTICLE=e-
ENERGY_KEV=20000000
N_PARTICLES=100
THREADS=1
N_RUNS=10
BEAM_RADIUS=0.01
OPTICAL="on"
PHYSICS_LIST="QGSP_BERT_EMV"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="runs/${WORLD}/${ENERGY_KEV}keV_${TIMESTAMP}"
echo "Starting ${N_RUNS} runs → ${OUT_DIR}"
pids=()
for i in $(seq 0 $((N_RUNS - 1))); do
    python3 simulator.py \
        --world $WORLD --particle $PARTICLE \
        --energy-kev $ENERGY_KEV --n $N_PARTICLES \
        --threads $THREADS --beam-radius $BEAM_RADIUS \
        --optical $OPTICAL --run-id $i \
        --output-dir $OUT_DIR > ${OUT_DIR}_run${i}.log 2>&1 &
    pids+=($!)
    echo "  Run ${i} started (PID ${pids[-1]})"
done
echo "Waiting for all runs..."
for pid in "${pids[@]}"; do wait $pid; done
echo "Done. Analysing..."
python3 analyze.py --batch-dir $OUT_DIR
python3 energy_calc.py --batch-dir $OUT_DIR
python3 plot_3d.py --batch-dir $OUT_DIR
echo "Results in: $OUT_DIR"
' > sim_run.log 2>&1 &
echo "Master PID: $!"