#!/bin/bash

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

for D in "${DOMAINS[@]}"; do
    sbatch --job-name=$D train_ood.sh $D
done
