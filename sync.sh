#!/bin/bash

# Check if a commit message was provided
if [ -z "$1" ]; then
    echo "Error: Please provide a commit message."
    echo "Usage: ./sync.sh \"your commit message\""
    exit 1
fi

# 1. Push code changes to GitHub from your laptop
git add . && git commit -m "$1" && git push

# 2. Safely mirror the code directory to the server
# The trailing slashes on the excludes ensure the directories themselves and their contents are protected.
rsync -avz --delete \
    --exclude='.git/' \
    --exclude='runs/' \
    --exclude='analysis/' \
    --exclude='**/analysis/' \
    --exclude='*.root' \
    --exclude='*.json' \
    --exclude='*.mhd' \
    --exclude='*.raw' \
    --exclude='*.npy' \
    ./ uakgun@172.16.17.251:/home/uakgun/gate_sims/discrete_sims/

echo "Sync complete! Server code updated, data folders safely preserved."