#!/bin/bash
# 1. Push everything to GitHub from your laptop
git add . && git commit -m "$1" && git push

# 2. Instantly mirror the folder to the server
rsync -avz --delete --exclude='.git/' ./ uakgun@discovery:~/THOMAS/discrete_sims/
