#!/bin/bash

# ----------------------------------------
# Domains you want to evaluate
# ----------------------------------------
DOMAINS=(
    animals
    vehicles
    arts_and_works
    landscapes
    food_and_drinks
    clothing
    interior_spaces
    household_items
    buildings
    people
)

# Ensure logs folder exists
mkdir -p logs

D1="interior_spaces"
D2="vehicles"
# ----------------------------------------
# Submit all D1 -> D2 jobs
# ----------------------------------------
#for D1 in "${DOMAINS[@]}"; do
#    for D2 in "${DOMAINS[@]}"; do

        echo "Submitting job: $D1 → $D2"

        sbatch \
            --export=ALL,D1=$D1,D2=$D2 \
            --job-name=EVAL_${D1}_${D2} \
            --output=logs/%j/${D1}_${D2}.out \
            --error=logs/%j/${D1}_${D2}.err \
            eval_cross.sh

#    done
#done
