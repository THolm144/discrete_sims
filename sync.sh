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
    --exclude='profile_analysis/' \
    --exclude='**/profile_analysis/' \
    --exclude='test_analysis/' \
    --exclude='**/test_analysis/' \
    --exclude='unfolded_profile_analysis/' \
    --exclude='**/unfolded_profile_analysis/' \
    --exclude='tikhonov_profile_analysis/' \
    --exclude='**/tikhonov_profile_analysis/' \
    --exclude='response_matrices/' \
    --exclude='**/response_matrices/' \
    --exclude='12_sweep_analysis/' \
    --exclude='**/12_sweep_analysis/' \
    --exclude='analysis_output/' \
    --exclude='**/analysis_output/' \
    --exclude='*.root' \
    --exclude='*.json' \
    --exclude='*.mhd' \
    --exclude='*.raw' \
    --exclude='*.npy' \
    ./ uakgun@172.16.17.251:/home/uakgun/env/THOMAS/discrete_sims/
    

echo "Sync complete! Server code updated in uakgun/env, data folders safely preserved."