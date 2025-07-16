#!/bin/bash

## Compressor Benchmarking Function
## Author: Skyler Ruiter (sruiter@iu.edu) 07/15/2025
####################################################################

# run_compressor: Run compressor and save output to a text file
# Usage: run_compressor <input_file> <data_size> <compressor_config> [stf]

FZMOD_BIN=./release_build/fzmod

mkdir -p scripts/benchmarking/results
OUTPUT=scripts/benchmarking/results

run_compressor() {
    local input_file="$1"
    local data_size="$2"
    local compressor_config="$3"
    local is_stf="${4:-false}"
    
    # Extract error bound and mode from compressor_config
    local eb=$(echo "$compressor_config" | grep -oP '(?<=-e )[^ ]*')
    local mode=$(echo "$compressor_config" | grep -oP '(?<=-m )[^ ]*')
    # Fallback if not found
    eb=${eb:-unknown}
    mode=${mode:-unknown}

    # Set file paths based on compression mode
    if [ "$is_stf" = "true" ]; then
        local compressed_file="$input_file.stf_compressed"
        local decompressed_file="$input_file.stf_compressed.stf_decompressed"
        # Add STF flag to compressor config if not already present
        if ! echo "$compressor_config" | grep -q -- "--stf"; then
            compressor_config="$compressor_config --stf"
        fi
    else
        local compressed_file="$input_file.fzmod"
        local decompressed_file="$input_file.fzmod.fzmodx"
    fi

    local txt_output="${input_file##*/}.out_${eb}_${mode}.txt"
    # Add STF suffix if STF mode is enabled
    if [ "$is_stf" = "true" ]; then
        txt_output="${txt_output%.txt}_stf.txt"
    fi

    COMP_CMD="${FZMOD_BIN} -i $input_file -l $data_size $compressor_config -z --verbose --report"
    echo "Running command: $COMP_CMD"
    
    # Capture both stdout and stderr
    output=$(eval ${COMP_CMD} 2>&1)
    
    # Check for the outlier buffer error
    if echo "$output" | grep -q "Number of outliers exceeds reserved buffer size"; then
        echo "    Error: Too many outliers detected. Try increasing outlier_buffer_ratio or lowering error bound."
        echo "$output" > ${OUTPUT}/${txt_output}
        return 1
    else
        # If no error, save output to file
        echo "$output" > ${OUTPUT}/${txt_output}
        
        # Customize decompression command based on compression mode
        if [ "$is_stf" = "true" ]; then
            DECOMP_CMD="${FZMOD_BIN} -i $compressed_file -x --compare $input_file --verbose --report --stf"
        else
            DECOMP_CMD="${FZMOD_BIN} -i $compressed_file -x --compare $input_file --verbose --report"
        fi
        
        echo "Running command: $DECOMP_CMD"
        eval ${DECOMP_CMD} >> ${OUTPUT}/${txt_output}
    fi

    # Clean up compressed and decompressed files to save storage space
    if [ -f "$compressed_file" ]; then
        echo "Removing compressed file: $compressed_file"
        rm "$compressed_file"
    fi
    
    if [ -f "$decompressed_file" ]; then
        echo "Removing decompressed file: $decompressed_file"
        rm "$decompressed_file"
    fi

    echo ""
}

####################################################################