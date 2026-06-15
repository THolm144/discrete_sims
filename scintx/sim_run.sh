#!/bin/bash

# Ensure a commit message was provided
if [ -z "$1" ]; then
    echo "Error: Please provide a commit message."
    echo "Usage: bash sync.sh 'your commit message'"
    exit 1
fi

echo "================────────────────────────"
echo " 1. Pushing code to GitHub..."
echo "================────────────────────────"
# Git will now automatically skip everything listed in your .gitignore
git add .
git commit -m "$1"
git push

echo -e "\n================────────────────────────"
echo " 2. Mirroring clean codebase to Server..."
echo "================────────────────────────"
# --filter=':- .gitignore' tells rsync to respect your gitignore file rules
# We keep manual exclusion of the .git/ folder itself.
rsync -avz --delete \
    --exclude='.git/' \
    --filter=':- .gitignore' \
    ./ uakgun@172.16.17.251:~/THOMAS/discrete_sims/

echo -e "\nSync complete!"