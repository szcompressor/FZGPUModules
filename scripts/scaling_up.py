#!/usr/bin/env python3
"""
scaling_up.py - A utility script to create larger test files by concatenating a source file multiple times.

Usage:
    python scaling_up.py input_file multiplier output_file

Arguments:
    input_file:  Path to the source data file
    multiplier:  Number of times to concatenate the data (positive integer)
    output_file: Path to the output file where the scaled data will be written

Example:
    python scaling_up.py data.bin 5 data_5x.bin
"""

import sys
import os
import shutil


def scale_up_file(input_file, multiplier, output_file):
    """
    Concatenate a file multiple times to create a larger file.
    
    Args:
        input_file (str): Path to the source file
        multiplier (int): Number of times to concatenate the data
        output_file (str): Path to the output file
    
    Returns:
        bool: True if operation was successful, False otherwise
    """
    try:
        # Validate multiplier
        multiplier = int(multiplier)
        if multiplier <= 0:
            print(f"Error: Multiplier must be a positive integer, got {multiplier}")
            return False
            
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' does not exist")
            return False
            
        # Get file size for progress reporting
        file_size = os.path.getsize(input_file)
        total_size = file_size * multiplier
        print(f"Input file size: {file_size / (1024*1024):.2f} MB")
        print(f"Expected output size: {total_size / (1024*1024):.2f} MB")
        
        # Create output file and copy contents multiplier times
        with open(input_file, 'rb') as in_file:
            data = in_file.read()
            
        with open(output_file, 'wb') as out_file:
            for i in range(multiplier):
                progress = (i / multiplier) * 100
                print(f"Progress: {progress:.1f}% - Writing chunk {i+1}/{multiplier}", end='\r')
                out_file.write(data)
                
        print(f"\nCompleted: Created '{output_file}' with {multiplier}x the data from '{input_file}'")
        print(f"Output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        return True
        
    except Exception as e:
        print(f"Error during scaling operation: {str(e)}")
        return False


def main():
    """Parse command line arguments and run the scaling operation."""
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
        
    input_file = sys.argv[1]
    multiplier = sys.argv[2]
    output_file = sys.argv[3]
    
    success = scale_up_file(input_file, multiplier, output_file)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
