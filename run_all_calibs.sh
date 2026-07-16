#!/bin/bash
# =====================================================================
# AUTO-GENERATED MASTER MULTI-WORLD CALIBRATION RUNNER
# Runs sequential Z-sweeps with dynamically scaled boundaries
# =====================================================================

set -e

# ============================================================================
# PIPELINE FOR WORLD: dsb1_radi_cal_energy (Square)
# ============================================================================
cat << 'EOF' > calibration_config.py
# =====================================================================
# AUTO-GENERATED CONFIGURATION - DO NOT EDIT MANUALLY
# =====================================================================
WORLD_PATH     = "dsb1_radi_cal_energy/worlds/dsb1_radi_cal_energy.py"
WORLD_NAME     = "dsb1_radi_cal_energy"
GEOMETRY_TYPE  = "Square"
ACTIVE_INDEX   = 2

# Beam Alignment (Capillary Coordinates in cm)
BEAM_X_CM      = -0.37032
BEAM_Y_CM      = 0.37032

# Physical Boundaries & Sensor Placement (cm)
CALOR_THICK_CM = 12.52856
SENSOR_Z_CM    = 9.16500
EOF
echo '[+] Applied configuration for world: dsb1_radi_cal_energy'

Z_SWEEP_VALUES=(-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00)
for z_val in "${Z_SWEEP_VALUES[@]}"; do
    echo "[*] Sweeping dsb1_radi_cal_energy at Z = ${z_val} cm..."
    mkdir -p "output/dsb1_radi_cal_energy/calib_z_${z_val}"
    # python3 run_simulation.py \
    #     --world "dsb1_radi_cal_energy/worlds/dsb1_radi_cal_energy.py" \
    #     --beam_x "-0.37032" \
    #     --beam_y "0.37032" \
    #     --beam_z "${z_val}" \
    #     --output_dir "output/dsb1_radi_cal_energy/calib_z_${z_val}"
    
    # python3 extract_prompt_attenuation.py --run_dir "output/dsb1_radi_cal_energy/calib_z_${z_val}" --z_offset "${z_val}"
done

# ============================================================================
# PIPELINE FOR WORLD: dsb1_radi_cal_triple (Square)
# ============================================================================
cat << 'EOF' > calibration_config.py
# =====================================================================
# AUTO-GENERATED CONFIGURATION - DO NOT EDIT MANUALLY
# =====================================================================
WORLD_PATH     = "dsb1_radi_cal_triple/worlds/dsb1_radi_cal_triple.py"
WORLD_NAME     = "dsb1_radi_cal_triple"
GEOMETRY_TYPE  = "Square"
ACTIVE_INDEX   = 2

# Beam Alignment (Capillary Coordinates in cm)
BEAM_X_CM      = -0.37032
BEAM_Y_CM      = 0.37032

# Physical Boundaries & Sensor Placement (cm)
CALOR_THICK_CM = 21.22856
SENSOR_Z_CM    = 13.51500
EOF
echo '[+] Applied configuration for world: dsb1_radi_cal_triple'

Z_SWEEP_VALUES=(-8.50 -6.80 -5.10 -3.40 -1.70 0.00 1.70 3.40 5.10 6.80 8.50)
for z_val in "${Z_SWEEP_VALUES[@]}"; do
    echo "[*] Sweeping dsb1_radi_cal_triple at Z = ${z_val} cm..."
    mkdir -p "output/dsb1_radi_cal_triple/calib_z_${z_val}"
    # python3 run_simulation.py \
    #     --world "dsb1_radi_cal_triple/worlds/dsb1_radi_cal_triple.py" \
    #     --beam_x "-0.37032" \
    #     --beam_y "0.37032" \
    #     --beam_z "${z_val}" \
    #     --output_dir "output/dsb1_radi_cal_triple/calib_z_${z_val}"
    
    # python3 extract_prompt_attenuation.py --run_dir "output/dsb1_radi_cal_triple/calib_z_${z_val}" --z_offset "${z_val}"
done

# ============================================================================
# PIPELINE FOR WORLD: dsb1_rc_hex (Hexagonal)
# ============================================================================
cat << 'EOF' > calibration_config.py
# =====================================================================
# AUTO-GENERATED CONFIGURATION - DO NOT EDIT MANUALLY
# =====================================================================
WORLD_PATH     = "dsb1_rc_hex/worlds/dsb1_rc_hex.py"
WORLD_NAME     = "dsb1_rc_hex"
GEOMETRY_TYPE  = "Hexagonal"
ACTIVE_INDEX   = 0

# Beam Alignment (Capillary Coordinates in cm)
BEAM_X_CM      = 0.00000
BEAM_Y_CM      = 0.35000

# Physical Boundaries & Sensor Placement (cm)
CALOR_THICK_CM = 12.52856
SENSOR_Z_CM    = 9.16500
EOF
echo '[+] Applied configuration for world: dsb1_rc_hex'

Z_SWEEP_VALUES=(-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00)
for z_val in "${Z_SWEEP_VALUES[@]}"; do
    echo "[*] Sweeping dsb1_rc_hex at Z = ${z_val} cm..."
    mkdir -p "output/dsb1_rc_hex/calib_z_${z_val}"
    # python3 run_simulation.py \
    #     --world "dsb1_rc_hex/worlds/dsb1_rc_hex.py" \
    #     --beam_x "0.00000" \
    #     --beam_y "0.35000" \
    #     --beam_z "${z_val}" \
    #     --output_dir "output/dsb1_rc_hex/calib_z_${z_val}"
    
    # python3 extract_prompt_attenuation.py --run_dir "output/dsb1_rc_hex/calib_z_${z_val}" --z_offset "${z_val}"
done

# ============================================================================
# PIPELINE FOR WORLD: dsb1_rc_hex_triple (Hexagonal)
# ============================================================================
cat << 'EOF' > calibration_config.py
# =====================================================================
# AUTO-GENERATED CONFIGURATION - DO NOT EDIT MANUALLY
# =====================================================================
WORLD_PATH     = "dsb1_rc_hex_triple/worlds/dsb1_rc_hex_triple.py"
WORLD_NAME     = "dsb1_rc_hex_triple"
GEOMETRY_TYPE  = "Hexagonal"
ACTIVE_INDEX   = 0

# Beam Alignment (Capillary Coordinates in cm)
BEAM_X_CM      = 0.00000
BEAM_Y_CM      = 0.35000

# Physical Boundaries & Sensor Placement (cm)
CALOR_THICK_CM = 21.22856
SENSOR_Z_CM    = 11.01500
EOF
echo '[+] Applied configuration for world: dsb1_rc_hex_triple'

Z_SWEEP_VALUES=(-8.50 -6.80 -5.10 -3.40 -1.70 0.00 1.70 3.40 5.10 6.80 8.50)
for z_val in "${Z_SWEEP_VALUES[@]}"; do
    echo "[*] Sweeping dsb1_rc_hex_triple at Z = ${z_val} cm..."
    mkdir -p "output/dsb1_rc_hex_triple/calib_z_${z_val}"
    # python3 run_simulation.py \
    #     --world "dsb1_rc_hex_triple/worlds/dsb1_rc_hex_triple.py" \
    #     --beam_x "0.00000" \
    #     --beam_y "0.35000" \
    #     --beam_z "${z_val}" \
    #     --output_dir "output/dsb1_rc_hex_triple/calib_z_${z_val}"
    
    # python3 extract_prompt_attenuation.py --run_dir "output/dsb1_rc_hex_triple/calib_z_${z_val}" --z_offset "${z_val}"
done

# ============================================================================
# PIPELINE FOR WORLD: luagce_radi_cal_energy (Square)
# ============================================================================
cat << 'EOF' > calibration_config.py
# =====================================================================
# AUTO-GENERATED CONFIGURATION - DO NOT EDIT MANUALLY
# =====================================================================
WORLD_PATH     = "luagce_radi_cal_energy/worlds/luagce_radi_cal_energy.py"
WORLD_NAME     = "luagce_radi_cal_energy"
GEOMETRY_TYPE  = "Square"
ACTIVE_INDEX   = 2

# Beam Alignment (Capillary Coordinates in cm)
BEAM_X_CM      = -0.37032
BEAM_Y_CM      = 0.37032

# Physical Boundaries & Sensor Placement (cm)
CALOR_THICK_CM = 12.52856
SENSOR_Z_CM    = 9.16500
EOF
echo '[+] Applied configuration for world: luagce_radi_cal_energy'

Z_SWEEP_VALUES=(-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00)
for z_val in "${Z_SWEEP_VALUES[@]}"; do
    echo "[*] Sweeping luagce_radi_cal_energy at Z = ${z_val} cm..."
    mkdir -p "output/luagce_radi_cal_energy/calib_z_${z_val}"
    # python3 run_simulation.py \
    #     --world "luagce_radi_cal_energy/worlds/luagce_radi_cal_energy.py" \
    #     --beam_x "-0.37032" \
    #     --beam_y "0.37032" \
    #     --beam_z "${z_val}" \
    #     --output_dir "output/luagce_radi_cal_energy/calib_z_${z_val}"
    
    # python3 extract_prompt_attenuation.py --run_dir "output/luagce_radi_cal_energy/calib_z_${z_val}" --z_offset "${z_val}"
done

# ============================================================================
# PIPELINE FOR WORLD: luagce_radi_cal_triple (Square)
# ============================================================================
cat << 'EOF' > calibration_config.py
# =====================================================================
# AUTO-GENERATED CONFIGURATION - DO NOT EDIT MANUALLY
# =====================================================================
WORLD_PATH     = "luagce_radi_cal_triple/worlds/luagce_radi_cal_triple.py"
WORLD_NAME     = "luagce_radi_cal_triple"
GEOMETRY_TYPE  = "Square"
ACTIVE_INDEX   = 2

# Beam Alignment (Capillary Coordinates in cm)
BEAM_X_CM      = -0.37032
BEAM_Y_CM      = 0.37032

# Physical Boundaries & Sensor Placement (cm)
CALOR_THICK_CM = 21.22856
SENSOR_Z_CM    = 13.51500
EOF
echo '[+] Applied configuration for world: luagce_radi_cal_triple'

Z_SWEEP_VALUES=(-8.50 -6.80 -5.10 -3.40 -1.70 0.00 1.70 3.40 5.10 6.80 8.50)
for z_val in "${Z_SWEEP_VALUES[@]}"; do
    echo "[*] Sweeping luagce_radi_cal_triple at Z = ${z_val} cm..."
    mkdir -p "output/luagce_radi_cal_triple/calib_z_${z_val}"
    # python3 run_simulation.py \
    #     --world "luagce_radi_cal_triple/worlds/luagce_radi_cal_triple.py" \
    #     --beam_x "-0.37032" \
    #     --beam_y "0.37032" \
    #     --beam_z "${z_val}" \
    #     --output_dir "output/luagce_radi_cal_triple/calib_z_${z_val}"
    
    # python3 extract_prompt_attenuation.py --run_dir "output/luagce_radi_cal_triple/calib_z_${z_val}" --z_offset "${z_val}"
done

# ============================================================================
# PIPELINE FOR WORLD: luagce_rc_hex (Hexagonal)
# ============================================================================
cat << 'EOF' > calibration_config.py
# =====================================================================
# AUTO-GENERATED CONFIGURATION - DO NOT EDIT MANUALLY
# =====================================================================
WORLD_PATH     = "luagce_rc_hex/worlds/luagce_rc_hex.py"
WORLD_NAME     = "luagce_rc_hex"
GEOMETRY_TYPE  = "Hexagonal"
ACTIVE_INDEX   = 0

# Beam Alignment (Capillary Coordinates in cm)
BEAM_X_CM      = 0.00000
BEAM_Y_CM      = 0.35000

# Physical Boundaries & Sensor Placement (cm)
CALOR_THICK_CM = 12.52856
SENSOR_Z_CM    = 9.16500
EOF
echo '[+] Applied configuration for world: luagce_rc_hex'

Z_SWEEP_VALUES=(-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00)
for z_val in "${Z_SWEEP_VALUES[@]}"; do
    echo "[*] Sweeping luagce_rc_hex at Z = ${z_val} cm..."
    mkdir -p "output/luagce_rc_hex/calib_z_${z_val}"
    # python3 run_simulation.py \
    #     --world "luagce_rc_hex/worlds/luagce_rc_hex.py" \
    #     --beam_x "0.00000" \
    #     --beam_y "0.35000" \
    #     --beam_z "${z_val}" \
    #     --output_dir "output/luagce_rc_hex/calib_z_${z_val}"
    
    # python3 extract_prompt_attenuation.py --run_dir "output/luagce_rc_hex/calib_z_${z_val}" --z_offset "${z_val}"
done

# ============================================================================
# PIPELINE FOR WORLD: luagce_rc_hex_triple (Hexagonal)
# ============================================================================
cat << 'EOF' > calibration_config.py
# =====================================================================
# AUTO-GENERATED CONFIGURATION - DO NOT EDIT MANUALLY
# =====================================================================
WORLD_PATH     = "luagce_rc_hex_triple/worlds/luagce_rc_hex_triple.py"
WORLD_NAME     = "luagce_rc_hex_triple"
GEOMETRY_TYPE  = "Hexagonal"
ACTIVE_INDEX   = 0

# Beam Alignment (Capillary Coordinates in cm)
BEAM_X_CM      = 0.00000
BEAM_Y_CM      = 0.35000

# Physical Boundaries & Sensor Placement (cm)
CALOR_THICK_CM = 21.22856
SENSOR_Z_CM    = 11.01500
EOF
echo '[+] Applied configuration for world: luagce_rc_hex_triple'

Z_SWEEP_VALUES=(-8.50 -6.80 -5.10 -3.40 -1.70 0.00 1.70 3.40 5.10 6.80 8.50)
for z_val in "${Z_SWEEP_VALUES[@]}"; do
    echo "[*] Sweeping luagce_rc_hex_triple at Z = ${z_val} cm..."
    mkdir -p "output/luagce_rc_hex_triple/calib_z_${z_val}"
    # python3 run_simulation.py \
    #     --world "luagce_rc_hex_triple/worlds/luagce_rc_hex_triple.py" \
    #     --beam_x "0.00000" \
    #     --beam_y "0.35000" \
    #     --beam_z "${z_val}" \
    #     --output_dir "output/luagce_rc_hex_triple/calib_z_${z_val}"
    
    # python3 extract_prompt_attenuation.py --run_dir "output/luagce_rc_hex_triple/calib_z_${z_val}" --z_offset "${z_val}"
done

# ============================================================================
# PIPELINE FOR WORLD: radi_cal (Square)
# ============================================================================
cat << 'EOF' > calibration_config.py
# =====================================================================
# AUTO-GENERATED CONFIGURATION - DO NOT EDIT MANUALLY
# =====================================================================
WORLD_PATH     = "radi_cal/worlds/radi_cal.py"
WORLD_NAME     = "radi_cal"
GEOMETRY_TYPE  = "Square"
ACTIVE_INDEX   = 2

# Beam Alignment (Capillary Coordinates in cm)
BEAM_X_CM      = -0.37032
BEAM_Y_CM      = 0.37032

# Physical Boundaries & Sensor Placement (cm)
CALOR_THICK_CM = 12.52856
SENSOR_Z_CM    = 9.16500
EOF
echo '[+] Applied configuration for world: radi_cal'

Z_SWEEP_VALUES=(-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00)
for z_val in "${Z_SWEEP_VALUES[@]}"; do
    echo "[*] Sweeping radi_cal at Z = ${z_val} cm..."
    mkdir -p "output/radi_cal/calib_z_${z_val}"
    # python3 run_simulation.py \
    #     --world "radi_cal/worlds/radi_cal.py" \
    #     --beam_x "-0.37032" \
    #     --beam_y "0.37032" \
    #     --beam_z "${z_val}" \
    #     --output_dir "output/radi_cal/calib_z_${z_val}"
    
    # python3 extract_prompt_attenuation.py --run_dir "output/radi_cal/calib_z_${z_val}" --z_offset "${z_val}"
done

# ============================================================================
# PIPELINE FOR WORLD: radi_cal_energy (Square)
# ============================================================================
cat << 'EOF' > calibration_config.py
# =====================================================================
# AUTO-GENERATED CONFIGURATION - DO NOT EDIT MANUALLY
# =====================================================================
WORLD_PATH     = "radi_cal_energy/worlds/radi_cal_energy.py"
WORLD_NAME     = "radi_cal_energy"
GEOMETRY_TYPE  = "Square"
ACTIVE_INDEX   = 2

# Beam Alignment (Capillary Coordinates in cm)
BEAM_X_CM      = -0.37032
BEAM_Y_CM      = 0.37032

# Physical Boundaries & Sensor Placement (cm)
CALOR_THICK_CM = 12.52856
SENSOR_Z_CM    = 9.16500
EOF
echo '[+] Applied configuration for world: radi_cal_energy'

Z_SWEEP_VALUES=(-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00)
for z_val in "${Z_SWEEP_VALUES[@]}"; do
    echo "[*] Sweeping radi_cal_energy at Z = ${z_val} cm..."
    mkdir -p "output/radi_cal_energy/calib_z_${z_val}"
    # python3 run_simulation.py \
    #     --world "radi_cal_energy/worlds/radi_cal_energy.py" \
    #     --beam_x "-0.37032" \
    #     --beam_y "0.37032" \
    #     --beam_z "${z_val}" \
    #     --output_dir "output/radi_cal_energy/calib_z_${z_val}"
    
    # python3 extract_prompt_attenuation.py --run_dir "output/radi_cal_energy/calib_z_${z_val}" --z_offset "${z_val}"
done

# ============================================================================
# PIPELINE FOR WORLD: radi_cal_triple (Square)
# ============================================================================
cat << 'EOF' > calibration_config.py
# =====================================================================
# AUTO-GENERATED CONFIGURATION - DO NOT EDIT MANUALLY
# =====================================================================
WORLD_PATH     = "radi_cal_triple/worlds/radi_cal_triple.py"
WORLD_NAME     = "radi_cal_triple"
GEOMETRY_TYPE  = "Square"
ACTIVE_INDEX   = 2

# Beam Alignment (Capillary Coordinates in cm)
BEAM_X_CM      = -0.37032
BEAM_Y_CM      = 0.37032

# Physical Boundaries & Sensor Placement (cm)
CALOR_THICK_CM = 21.22856
SENSOR_Z_CM    = 13.51500
EOF
echo '[+] Applied configuration for world: radi_cal_triple'

Z_SWEEP_VALUES=(-8.50 -6.80 -5.10 -3.40 -1.70 0.00 1.70 3.40 5.10 6.80 8.50)
for z_val in "${Z_SWEEP_VALUES[@]}"; do
    echo "[*] Sweeping radi_cal_triple at Z = ${z_val} cm..."
    mkdir -p "output/radi_cal_triple/calib_z_${z_val}"
    # python3 run_simulation.py \
    #     --world "radi_cal_triple/worlds/radi_cal_triple.py" \
    #     --beam_x "-0.37032" \
    #     --beam_y "0.37032" \
    #     --beam_z "${z_val}" \
    #     --output_dir "output/radi_cal_triple/calib_z_${z_val}"
    
    # python3 extract_prompt_attenuation.py --run_dir "output/radi_cal_triple/calib_z_${z_val}" --z_offset "${z_val}"
done

# ============================================================================
# PIPELINE FOR WORLD: rc_hex (Hexagonal)
# ============================================================================
cat << 'EOF' > calibration_config.py
# =====================================================================
# AUTO-GENERATED CONFIGURATION - DO NOT EDIT MANUALLY
# =====================================================================
WORLD_PATH     = "rc_hex/worlds/rc_hex.py"
WORLD_NAME     = "rc_hex"
GEOMETRY_TYPE  = "Hexagonal"
ACTIVE_INDEX   = 0

# Beam Alignment (Capillary Coordinates in cm)
BEAM_X_CM      = 0.00000
BEAM_Y_CM      = 0.35000

# Physical Boundaries & Sensor Placement (cm)
CALOR_THICK_CM = 12.52856
SENSOR_Z_CM    = 9.16500
EOF
echo '[+] Applied configuration for world: rc_hex'

Z_SWEEP_VALUES=(-5.00 -4.00 -3.00 -2.00 -1.00 0.00 1.00 2.00 3.00 4.00 5.00)
for z_val in "${Z_SWEEP_VALUES[@]}"; do
    echo "[*] Sweeping rc_hex at Z = ${z_val} cm..."
    mkdir -p "output/rc_hex/calib_z_${z_val}"
    # python3 run_simulation.py \
    #     --world "rc_hex/worlds/rc_hex.py" \
    #     --beam_x "0.00000" \
    #     --beam_y "0.35000" \
    #     --beam_z "${z_val}" \
    #     --output_dir "output/rc_hex/calib_z_${z_val}"
    
    # python3 extract_prompt_attenuation.py --run_dir "output/rc_hex/calib_z_${z_val}" --z_offset "${z_val}"
done

# ============================================================================
# PIPELINE FOR WORLD: rc_hex_triple (Hexagonal)
# ============================================================================
cat << 'EOF' > calibration_config.py
# =====================================================================
# AUTO-GENERATED CONFIGURATION - DO NOT EDIT MANUALLY
# =====================================================================
WORLD_PATH     = "rc_hex_triple/worlds/rc_hex_triple.py"
WORLD_NAME     = "rc_hex_triple"
GEOMETRY_TYPE  = "Hexagonal"
ACTIVE_INDEX   = 0

# Beam Alignment (Capillary Coordinates in cm)
BEAM_X_CM      = 0.00000
BEAM_Y_CM      = 0.35000

# Physical Boundaries & Sensor Placement (cm)
CALOR_THICK_CM = 21.22856
SENSOR_Z_CM    = 11.01500
EOF
echo '[+] Applied configuration for world: rc_hex_triple'

Z_SWEEP_VALUES=(-8.50 -6.80 -5.10 -3.40 -1.70 0.00 1.70 3.40 5.10 6.80 8.50)
for z_val in "${Z_SWEEP_VALUES[@]}"; do
    echo "[*] Sweeping rc_hex_triple at Z = ${z_val} cm..."
    mkdir -p "output/rc_hex_triple/calib_z_${z_val}"
    # python3 run_simulation.py \
    #     --world "rc_hex_triple/worlds/rc_hex_triple.py" \
    #     --beam_x "0.00000" \
    #     --beam_y "0.35000" \
    #     --beam_z "${z_val}" \
    #     --output_dir "output/rc_hex_triple/calib_z_${z_val}"
    
    # python3 extract_prompt_attenuation.py --run_dir "output/rc_hex_triple/calib_z_${z_val}" --z_offset "${z_val}"
done
