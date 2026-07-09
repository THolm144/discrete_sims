#!/bin/bash

# 1. Define the core source folders
sources=("radi_cal_energy" "radi_cal_triple" "rc_hex" "rc_hex_triple")

# 2. Loop through each source folder
for src in "${sources[@]}"; do
    # Ensure the source directory actually exists before trying to sync
    if [ ! -d "$src" ]; then
        echo "Warning: Source folder '$src' not found in current directory. Skipping."
        continue
    fi

    # Define the two new destination folder names
    dest1="dsb1_${src}"
    dest2="luagce_${src}"

    # Create the destination directories on the same level if they don't exist
    mkdir -p "$dest1" "$dest2"

    echo "=================================================="
    echo "Syncing: $src/ -> $dest1/ & $dest2/"
    echo "=================================================="

    # Sync to dsb1_* folder
    rsync -avm \
      --include="*/" \
      --include="*.py" --include="*.xml" --include="*.sh" --include="*.db" \
      --exclude="*" \
      "$src/" "$dest1/"

    # Sync to luagce_* folder
    rsync -avm \
      --include="*/" \
      --include="*.py" --include="*.xml" --include="*.sh" --include="*.db" \
      --exclude="*" \
      "$src/" "$dest2/"
done

echo "=================================================="
echo "Sync job complete!"