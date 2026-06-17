#!/bin/bash
set -e

VALUES=(0.0 0.05 0.1 0.2 0.4)
N=100
THREADS=8
ENERGY_KEV=9000

TEMPLATE="SurfaceProperties_template.xml"
TARGET="SurfaceProperties.xml"

cp "$TARGET" "$TEMPLATE.bak"  # backup current

for SA in "${VALUES[@]}"; do
    echo "=== sigmaalpha = $SA ==="

    python3 - "$SA" <<'PYEOF'
import sys, re
sa = sys.argv[1]
with open("SurfaceProperties.xml") as f:
    xml = f.read()
xml = re.sub(r'name="ScintX_air_surface"(.*?)sigmaalpha="[0-9.]+"',
             f'name="ScintX_air_surface"\\1sigmaalpha="{sa}"',
             xml, flags=re.DOTALL)
with open("SurfaceProperties.xml", "w") as f:
    f.write(xml)
PYEOF

    OUTDIR="runs/scintx_sipm_array/sweep_sigmaalpha_${SA}"
    python3 simulator.py \
        --world scintx_sipm_array \
        --n $N --threads $THREADS \
        --particle e- \
        --energy-kev $ENERGY_KEV \
        --output-dir "$OUTDIR" \
        --optical on --sipm-hits on --track-optical on

    python3 analyze.py --batch-dir "$OUTDIR" --world scintx_sipm_array

    echo "  Rendering 3D plot..."
    python3 plot_3d.py --batch-dir "$OUTDIR"
done

cp "$TEMPLATE.bak" "$TARGET"  # restore original
rm "$TEMPLATE.bak"
echo "Sweep complete."