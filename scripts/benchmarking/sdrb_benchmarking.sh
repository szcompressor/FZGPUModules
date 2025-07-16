#!/bin/bash

# SDRB Benchmarking Script
# Author: Skyler Ruiter (sruiter@iu.edu) 07/15/2025
# This script benchmarks the SDRB compressor with various configurations.
#############################################################

# For example:
# run_compressor "$data_dir/datafile.f32" $some_size "-t f32 -m rel -e 1e-4 -d $dimensions" true

# Load the compressor run script
source scripts/benchmarking/run_compressor.sh

# test data file
run_compressor data/CLDHGH.f32 6480000 "-t f32 -m rel -e 1e-4" true

CESM_DATA="data/CESM_1800x3600"
EXAALT_DATA="data/EXAALT_2869440"
HACC_DATA="data/HACC_280953867"
HURR_DATA="data/HURR_100x500x500"

DATA_DIRS=(
    "$CESM_DATA"
    "$EXAALT_DATA"
    "$HACC_DATA"
    "$HURR_DATA"
)

eb_modes=("abs" "rel")
eb_values=("1e-2" "1e-3" "1e-4" "1e-5" "1e-6" "1e-7")

for data_dir in "${DATA_DIRS[@]}"; do
    echo "Benchmarking SDRB on data directory: $data_dir"
    
    # Extract dataset name (everything before the first underscore)
    dir_basename=$(basename "$data_dir")
    dataset_name=$(echo "$dir_basename" | cut -d'_' -f1)
    
    # Extract dimensions
    dimensions=$(echo "$dir_basename" | cut -d'_' -f2-)
    
    # Reverse dimensions if they contain 'x'
    if [[ "$dimensions" == *x* ]]; then
        # Split dimensions by 'x' and reverse order
        IFS='x' read -r -a dims_array <<< "$dimensions"
        reversed_dims=""
        for ((i=${#dims_array[@]}-1; i>=0; i--)); do
            if [[ $i -eq ${#dims_array[@]}-1 ]]; then
                reversed_dims="${dims_array[i]}"
            else
                reversed_dims+="x${dims_array[i]}"
            fi
        done
        dimensions="$reversed_dims"
    fi
    
    echo "Dataset: $dataset_name"
    echo "Dimensions: $dimensions"
    


    echo ""
done