#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_calibration_discovery.sh — Parallel Multi-World Calibration (20 Cores)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=gate_calib_20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --partition=express
#SBATCH --output=calib_pool_%j.log

ROOT_DIR=$(pwd)
JOB_FILE="${ROOT_DIR}/calib_job_list.txt"
> "$JOB_FILE"

MAX_CONCURRENT_SIMS=20
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "========================================================================"
echo " Preparing Execution Matrix for Multi-World Calibration Queue"
echo "========================================================================"

# --- Queue setup for dsb1_radi_cal_energy ---
WORLD_DIR="${ROOT_DIR}/dsb1_radi_cal_energy"
if [ -d "$WORLD_DIR" ]; then
    MASTER_BATCH_DIR="${WORLD_DIR}/runs/calib_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    Z_SWEEP=(-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00)
    for Z_VAL in "${Z_SWEEP[@]}"; do
        RUN_OUT_DIR="${MASTER_BATCH_DIR}/z_${Z_VAL}"
        mkdir -p "$RUN_OUT_DIR"
        LOG_FILE="${MASTER_BATCH_DIR}/logs/z_${Z_VAL}.log"

        # Parameters written to job list (9 arguments per task row):
        # 0:WorldDir 1:WorldPath 2:X 3:Y 4:Z 5:OutputDir 6:LogFile 7:SensorZ 8:WorldName
        echo "${WORLD_DIR} dsb1_radi_cal_energy/worlds/dsb1_radi_cal_energy.py -0.37032 0.37032 ${Z_VAL} ${RUN_OUT_DIR} ${LOG_FILE} 9.16500 dsb1_radi_cal_energy" >> "$JOB_FILE"
    done
fi

# --- Queue setup for dsb1_radi_cal_triple ---
WORLD_DIR="${ROOT_DIR}/dsb1_radi_cal_triple"
if [ -d "$WORLD_DIR" ]; then
    MASTER_BATCH_DIR="${WORLD_DIR}/runs/calib_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    Z_SWEEP=(-8.50 -6.80 -5.10 -3.40 -1.70 0.00 1.70 3.40 5.10 6.80 8.50)
    for Z_VAL in "${Z_SWEEP[@]}"; do
        RUN_OUT_DIR="${MASTER_BATCH_DIR}/z_${Z_VAL}"
        mkdir -p "$RUN_OUT_DIR"
        LOG_FILE="${MASTER_BATCH_DIR}/logs/z_${Z_VAL}.log"

        # Parameters written to job list (9 arguments per task row):
        # 0:WorldDir 1:WorldPath 2:X 3:Y 4:Z 5:OutputDir 6:LogFile 7:SensorZ 8:WorldName
        echo "${WORLD_DIR} dsb1_radi_cal_triple/worlds/dsb1_radi_cal_triple.py -0.37032 0.37032 ${Z_VAL} ${RUN_OUT_DIR} ${LOG_FILE} 13.51500 dsb1_radi_cal_triple" >> "$JOB_FILE"
    done
fi

# --- Queue setup for dsb1_rc_hex ---
WORLD_DIR="${ROOT_DIR}/dsb1_rc_hex"
if [ -d "$WORLD_DIR" ]; then
    MASTER_BATCH_DIR="${WORLD_DIR}/runs/calib_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    Z_SWEEP=(-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00)
    for Z_VAL in "${Z_SWEEP[@]}"; do
        RUN_OUT_DIR="${MASTER_BATCH_DIR}/z_${Z_VAL}"
        mkdir -p "$RUN_OUT_DIR"
        LOG_FILE="${MASTER_BATCH_DIR}/logs/z_${Z_VAL}.log"

        # Parameters written to job list (9 arguments per task row):
        # 0:WorldDir 1:WorldPath 2:X 3:Y 4:Z 5:OutputDir 6:LogFile 7:SensorZ 8:WorldName
        echo "${WORLD_DIR} dsb1_rc_hex/worlds/dsb1_rc_hex.py 0.00000 0.35000 ${Z_VAL} ${RUN_OUT_DIR} ${LOG_FILE} 9.16500 dsb1_rc_hex" >> "$JOB_FILE"
    done
fi

# --- Queue setup for dsb1_rc_hex_triple ---
WORLD_DIR="${ROOT_DIR}/dsb1_rc_hex_triple"
if [ -d "$WORLD_DIR" ]; then
    MASTER_BATCH_DIR="${WORLD_DIR}/runs/calib_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    Z_SWEEP=(-8.50 -6.80 -5.10 -3.40 -1.70 0.00 1.70 3.40 5.10 6.80 8.50)
    for Z_VAL in "${Z_SWEEP[@]}"; do
        RUN_OUT_DIR="${MASTER_BATCH_DIR}/z_${Z_VAL}"
        mkdir -p "$RUN_OUT_DIR"
        LOG_FILE="${MASTER_BATCH_DIR}/logs/z_${Z_VAL}.log"

        # Parameters written to job list (9 arguments per task row):
        # 0:WorldDir 1:WorldPath 2:X 3:Y 4:Z 5:OutputDir 6:LogFile 7:SensorZ 8:WorldName
        echo "${WORLD_DIR} dsb1_rc_hex_triple/worlds/dsb1_rc_hex_triple.py 0.00000 0.35000 ${Z_VAL} ${RUN_OUT_DIR} ${LOG_FILE} 11.01500 dsb1_rc_hex_triple" >> "$JOB_FILE"
    done
fi

# --- Queue setup for luagce_radi_cal_energy ---
WORLD_DIR="${ROOT_DIR}/luagce_radi_cal_energy"
if [ -d "$WORLD_DIR" ]; then
    MASTER_BATCH_DIR="${WORLD_DIR}/runs/calib_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    Z_SWEEP=(-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00)
    for Z_VAL in "${Z_SWEEP[@]}"; do
        RUN_OUT_DIR="${MASTER_BATCH_DIR}/z_${Z_VAL}"
        mkdir -p "$RUN_OUT_DIR"
        LOG_FILE="${MASTER_BATCH_DIR}/logs/z_${Z_VAL}.log"

        # Parameters written to job list (9 arguments per task row):
        # 0:WorldDir 1:WorldPath 2:X 3:Y 4:Z 5:OutputDir 6:LogFile 7:SensorZ 8:WorldName
        echo "${WORLD_DIR} luagce_radi_cal_energy/worlds/luagce_radi_cal_energy.py -0.37032 0.37032 ${Z_VAL} ${RUN_OUT_DIR} ${LOG_FILE} 9.16500 luagce_radi_cal_energy" >> "$JOB_FILE"
    done
fi

# --- Queue setup for luagce_radi_cal_triple ---
WORLD_DIR="${ROOT_DIR}/luagce_radi_cal_triple"
if [ -d "$WORLD_DIR" ]; then
    MASTER_BATCH_DIR="${WORLD_DIR}/runs/calib_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    Z_SWEEP=(-8.50 -6.80 -5.10 -3.40 -1.70 0.00 1.70 3.40 5.10 6.80 8.50)
    for Z_VAL in "${Z_SWEEP[@]}"; do
        RUN_OUT_DIR="${MASTER_BATCH_DIR}/z_${Z_VAL}"
        mkdir -p "$RUN_OUT_DIR"
        LOG_FILE="${MASTER_BATCH_DIR}/logs/z_${Z_VAL}.log"

        # Parameters written to job list (9 arguments per task row):
        # 0:WorldDir 1:WorldPath 2:X 3:Y 4:Z 5:OutputDir 6:LogFile 7:SensorZ 8:WorldName
        echo "${WORLD_DIR} luagce_radi_cal_triple/worlds/luagce_radi_cal_triple.py -0.37032 0.37032 ${Z_VAL} ${RUN_OUT_DIR} ${LOG_FILE} 13.51500 luagce_radi_cal_triple" >> "$JOB_FILE"
    done
fi

# --- Queue setup for luagce_rc_hex ---
WORLD_DIR="${ROOT_DIR}/luagce_rc_hex"
if [ -d "$WORLD_DIR" ]; then
    MASTER_BATCH_DIR="${WORLD_DIR}/runs/calib_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    Z_SWEEP=(-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00)
    for Z_VAL in "${Z_SWEEP[@]}"; do
        RUN_OUT_DIR="${MASTER_BATCH_DIR}/z_${Z_VAL}"
        mkdir -p "$RUN_OUT_DIR"
        LOG_FILE="${MASTER_BATCH_DIR}/logs/z_${Z_VAL}.log"

        # Parameters written to job list (9 arguments per task row):
        # 0:WorldDir 1:WorldPath 2:X 3:Y 4:Z 5:OutputDir 6:LogFile 7:SensorZ 8:WorldName
        echo "${WORLD_DIR} luagce_rc_hex/worlds/luagce_rc_hex.py 0.00000 0.35000 ${Z_VAL} ${RUN_OUT_DIR} ${LOG_FILE} 9.16500 luagce_rc_hex" >> "$JOB_FILE"
    done
fi

# --- Queue setup for luagce_rc_hex_triple ---
WORLD_DIR="${ROOT_DIR}/luagce_rc_hex_triple"
if [ -d "$WORLD_DIR" ]; then
    MASTER_BATCH_DIR="${WORLD_DIR}/runs/calib_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    Z_SWEEP=(-8.50 -6.80 -5.10 -3.40 -1.70 0.00 1.70 3.40 5.10 6.80 8.50)
    for Z_VAL in "${Z_SWEEP[@]}"; do
        RUN_OUT_DIR="${MASTER_BATCH_DIR}/z_${Z_VAL}"
        mkdir -p "$RUN_OUT_DIR"
        LOG_FILE="${MASTER_BATCH_DIR}/logs/z_${Z_VAL}.log"

        # Parameters written to job list (9 arguments per task row):
        # 0:WorldDir 1:WorldPath 2:X 3:Y 4:Z 5:OutputDir 6:LogFile 7:SensorZ 8:WorldName
        echo "${WORLD_DIR} luagce_rc_hex_triple/worlds/luagce_rc_hex_triple.py 0.00000 0.35000 ${Z_VAL} ${RUN_OUT_DIR} ${LOG_FILE} 11.01500 luagce_rc_hex_triple" >> "$JOB_FILE"
    done
fi

# --- Queue setup for radi_cal ---
WORLD_DIR="${ROOT_DIR}/radi_cal"
if [ -d "$WORLD_DIR" ]; then
    MASTER_BATCH_DIR="${WORLD_DIR}/runs/calib_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    Z_SWEEP=(-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00)
    for Z_VAL in "${Z_SWEEP[@]}"; do
        RUN_OUT_DIR="${MASTER_BATCH_DIR}/z_${Z_VAL}"
        mkdir -p "$RUN_OUT_DIR"
        LOG_FILE="${MASTER_BATCH_DIR}/logs/z_${Z_VAL}.log"

        # Parameters written to job list (9 arguments per task row):
        # 0:WorldDir 1:WorldPath 2:X 3:Y 4:Z 5:OutputDir 6:LogFile 7:SensorZ 8:WorldName
        echo "${WORLD_DIR} radi_cal/worlds/radi_cal.py -0.37032 0.37032 ${Z_VAL} ${RUN_OUT_DIR} ${LOG_FILE} 9.16500 radi_cal" >> "$JOB_FILE"
    done
fi

# --- Queue setup for radi_cal_energy ---
WORLD_DIR="${ROOT_DIR}/radi_cal_energy"
if [ -d "$WORLD_DIR" ]; then
    MASTER_BATCH_DIR="${WORLD_DIR}/runs/calib_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    Z_SWEEP=(-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00)
    for Z_VAL in "${Z_SWEEP[@]}"; do
        RUN_OUT_DIR="${MASTER_BATCH_DIR}/z_${Z_VAL}"
        mkdir -p "$RUN_OUT_DIR"
        LOG_FILE="${MASTER_BATCH_DIR}/logs/z_${Z_VAL}.log"

        # Parameters written to job list (9 arguments per task row):
        # 0:WorldDir 1:WorldPath 2:X 3:Y 4:Z 5:OutputDir 6:LogFile 7:SensorZ 8:WorldName
        echo "${WORLD_DIR} radi_cal_energy/worlds/radi_cal_energy.py -0.37032 0.37032 ${Z_VAL} ${RUN_OUT_DIR} ${LOG_FILE} 9.16500 radi_cal_energy" >> "$JOB_FILE"
    done
fi

# --- Queue setup for radi_cal_triple ---
WORLD_DIR="${ROOT_DIR}/radi_cal_triple"
if [ -d "$WORLD_DIR" ]; then
    MASTER_BATCH_DIR="${WORLD_DIR}/runs/calib_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    Z_SWEEP=(-8.50 -6.80 -5.10 -3.40 -1.70 0.00 1.70 3.40 5.10 6.80 8.50)
    for Z_VAL in "${Z_SWEEP[@]}"; do
        RUN_OUT_DIR="${MASTER_BATCH_DIR}/z_${Z_VAL}"
        mkdir -p "$RUN_OUT_DIR"
        LOG_FILE="${MASTER_BATCH_DIR}/logs/z_${Z_VAL}.log"

        # Parameters written to job list (9 arguments per task row):
        # 0:WorldDir 1:WorldPath 2:X 3:Y 4:Z 5:OutputDir 6:LogFile 7:SensorZ 8:WorldName
        echo "${WORLD_DIR} radi_cal_triple/worlds/radi_cal_triple.py -0.37032 0.37032 ${Z_VAL} ${RUN_OUT_DIR} ${LOG_FILE} 13.51500 radi_cal_triple" >> "$JOB_FILE"
    done
fi

# --- Queue setup for rc_hex ---
WORLD_DIR="${ROOT_DIR}/rc_hex"
if [ -d "$WORLD_DIR" ]; then
    MASTER_BATCH_DIR="${WORLD_DIR}/runs/calib_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    Z_SWEEP=(-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00)
    for Z_VAL in "${Z_SWEEP[@]}"; do
        RUN_OUT_DIR="${MASTER_BATCH_DIR}/z_${Z_VAL}"
        mkdir -p "$RUN_OUT_DIR"
        LOG_FILE="${MASTER_BATCH_DIR}/logs/z_${Z_VAL}.log"

        # Parameters written to job list (9 arguments per task row):
        # 0:WorldDir 1:WorldPath 2:X 3:Y 4:Z 5:OutputDir 6:LogFile 7:SensorZ 8:WorldName
        echo "${WORLD_DIR} rc_hex/worlds/rc_hex.py 0.00000 0.35000 ${Z_VAL} ${RUN_OUT_DIR} ${LOG_FILE} 9.16500 rc_hex" >> "$JOB_FILE"
    done
fi

# --- Queue setup for rc_hex_triple ---
WORLD_DIR="${ROOT_DIR}/rc_hex_triple"
if [ -d "$WORLD_DIR" ]; then
    MASTER_BATCH_DIR="${WORLD_DIR}/runs/calib_${TIMESTAMP}"
    mkdir -p "${MASTER_BATCH_DIR}/logs"

    Z_SWEEP=(-8.50 -6.80 -5.10 -3.40 -1.70 0.00 1.70 3.40 5.10 6.80 8.50)
    for Z_VAL in "${Z_SWEEP[@]}"; do
        RUN_OUT_DIR="${MASTER_BATCH_DIR}/z_${Z_VAL}"
        mkdir -p "$RUN_OUT_DIR"
        LOG_FILE="${MASTER_BATCH_DIR}/logs/z_${Z_VAL}.log"

        # Parameters written to job list (9 arguments per task row):
        # 0:WorldDir 1:WorldPath 2:X 3:Y 4:Z 5:OutputDir 6:LogFile 7:SensorZ 8:WorldName
        echo "${WORLD_DIR} rc_hex_triple/worlds/rc_hex_triple.py 0.00000 0.35000 ${Z_VAL} ${RUN_OUT_DIR} ${LOG_FILE} 11.01500 rc_hex_triple" >> "$JOB_FILE"
    done
fi


TOTAL_JOBS=$(wc -l < "$JOB_FILE")
echo " [✓] Generated ${TOTAL_JOBS} calibration runs inside calib_job_list.txt."
echo "------------------------------------------------------------------------"
echo " [+] Handing execution queue to xargs pool (Concurrency = 20 Cores)..."

(
    sleep 2 
    while [ -f "$JOB_FILE" ] || [ $(pgrep -f "simulator.py" | wc -l) -gt 0 ]; do
        COMPLETED_JOBS=$(find ${ROOT_DIR}/*/runs/calib_${TIMESTAMP}/logs -name "*.log" 2>/dev/null | wc -l)
        ACTIVE_CORES=$(pgrep -f "simulator.py" | wc -l)
        printf "\r     -> Queue Progress: %4d / %d completed | [%2d Cores Occupied]" "$COMPLETED_JOBS" "$TOTAL_JOBS" "$ACTIVE_CORES"
        sleep 1
    done
) &
TRACKER_PID=$!

# Execute tasks. We pull exactly 9 arguments per row.
xargs -P "$MAX_CONCURRENT_SIMS" -n 9 bash -c '
    WORLD_DIR="$0"
    WORLD_PATH="$1"
    BEAM_X="$2"
    BEAM_Y="$3"
    BEAM_Z="$4"
    OUT_DIR="$5"
    LOG_FILE="$6"
    SENSOR_Z="$7"
    WORLD_NAME="$8"

    # 1. Execute OpenGATE simulation targeting the specific capillary fiber path
    python3 "${WORLD_DIR}/simulator.py" \
        --world "$WORLD_PATH" \
        --particle "e-" \
        --energy-kev 50000000 \
        --n 24 \
        --threads 1 \
        --beam-radius 0.01 \
        --optical "on" \
        --cherenkov "off" \
        --hits-optical-only on \
        --physics-list "QGSP_BERT_EMV" \
        --beam_x "$BEAM_X" \
        --beam_y "$BEAM_Y" \
        --beam_z "$BEAM_Z" \
        --run-id 0 \
        --output-dir "$OUT_DIR" > "$LOG_FILE" 2>&1

    # 2. Immediately execute analysis within the isolated thread environment
    if [ $? -eq 0 ]; then
        python3 extract_prompt_attenuation.py \
            --run_dir "$OUT_DIR" \
            --z_offset "$BEAM_Z" \
            --sensor_z "$SENSOR_Z" >> "$LOG_FILE" 2>&1
    else
        echo "[-] Run failed for ${WORLD_NAME} at Z=${BEAM_Z}" >> "$LOG_FILE"
    fi
' < "$JOB_FILE"

rm -f "$JOB_FILE"
wait $TRACKER_PID 2>/dev/null

echo ""
echo "========================================================================"
echo " [✓] ALL CALIBRATION SIMULATIONS AND ANALYSIS PILES COMPLETE."
echo "========================================================================"
