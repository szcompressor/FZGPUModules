#!/usr/bin/env python3
"""
Data testing script for comparing decompressed files against original data.
Calculates PSNR, MSE, and other quality metrics for compression evaluation.
"""

import numpy as np
import struct
import os
import sys
from typing import Tuple, Dict, Any

def load_f32_file(filepath: str) -> np.ndarray:
    """
    Load binary float32 data from file.
    
    Args:
        filepath: Path to the .f32 file
        
    Returns:
        numpy array containing the float32 data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = f.read()
    
    # Convert binary data to float32 array
    num_floats = len(data) // 4
    float_data = struct.unpack(f'{num_floats}f', data)
    return np.array(float_data, dtype=np.float32)

def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between original and reconstructed data.
    
    Args:
        original: Original data array
        reconstructed: Reconstructed/decompressed data array
        
    Returns:
        PSNR value in dB
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}")
    
    # Calculate MSE
    mse = np.mean((original - reconstructed) ** 2)
    
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    
    # Calculate max value in original data
    max_val = np.max(np.abs(original))
    if max_val == 0:
        max_val = 1.0  # Avoid division by zero
    
    # PSNR = 20 * log10(MAX / sqrt(MSE))
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr

def calculate_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive quality metrics between original and reconstructed data.
    
    Args:
        original: Original data array
        reconstructed: Reconstructed/decompressed data array
        
    Returns:
        Dictionary containing various quality metrics
    """
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}")
    
    # Calculate differences
    diff = original - reconstructed
    abs_diff = np.abs(diff)
    
    # Basic statistics
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_diff)
    max_error = np.max(abs_diff)
    
    # PSNR
    psnr = calculate_psnr(original, reconstructed)
    
    # Relative errors
    original_range = np.max(original) - np.min(original)
    relative_rmse = rmse / original_range if original_range != 0 else 0
    relative_max_error = max_error / original_range if original_range != 0 else 0
    
    # Correlation coefficient
    correlation = np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1]
    
    # Data range information
    orig_min, orig_max = np.min(original), np.max(original)
    recon_min, recon_max = np.min(reconstructed), np.max(reconstructed)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'psnr': psnr,
        'relative_rmse': relative_rmse,
        'relative_max_error': relative_max_error,
        'correlation': correlation,
        'original_range': (orig_min, orig_max),
        'reconstructed_range': (recon_min, recon_max),
        'data_points': len(original)
    }

def analyze_differences(original: np.ndarray, reconstructed: np.ndarray, name: str) -> None:
    """
    Perform detailed analysis of differences between original and reconstructed data.
    
    Args:
        original: Original data array
        reconstructed: Reconstructed data array
        name: Name identifier for the comparison
    """
    print(f"\n=== Detailed Analysis for {name} ===")
    
    diff = original - reconstructed
    abs_diff = np.abs(diff)
    
    # Histogram of differences
    print(f"Difference statistics:")
    print(f"  Mean difference: {np.mean(diff):.6e}")
    print(f"  Std dev of differences: {np.std(diff):.6e}")
    print(f"  Min difference: {np.min(diff):.6e}")
    print(f"  Max difference: {np.max(diff):.6e}")
    
    # Percentiles of absolute differences
    percentiles = [50, 90, 95, 99, 99.9]
    print(f"Absolute difference percentiles:")
    for p in percentiles:
        val = np.percentile(abs_diff, p)
        print(f"  {p}th percentile: {val:.6e}")
    
    # Count of exact matches
    exact_matches = np.sum(diff == 0)
    total_points = len(diff)
    print(f"Exact matches: {exact_matches}/{total_points} ({100*exact_matches/total_points:.2f}%)")
    
    # Check for systematic bias
    if len(diff) > 1000:  # Only for reasonably sized datasets
        # Check if differences follow a pattern
        mean_diff = np.mean(diff)
        if abs(mean_diff) > 1e-10:
            print(f"Potential systematic bias detected: mean difference = {mean_diff:.6e}")

def compare_decompressed_files(decompressed_data_dict: Dict[str, np.ndarray]) -> None:
    """
    Compare decompressed files against each other to find differences.
    
    Args:
        decompressed_data_dict: Dictionary mapping method names to data arrays
    """
    methods = list(decompressed_data_dict.keys())
    if len(methods) < 2:
        return
    
    print(f"\n{'='*80}")
    print(f"INTER-METHOD COMPARISON")
    print(f"{'='*80}")
    
    # Compare each pair of methods
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1, method2 = methods[i], methods[j]
            data1, data2 = decompressed_data_dict[method1], decompressed_data_dict[method2]
            
            print(f"\n--- Comparing {method1} vs {method2} ---")
            
            if data1.shape != data2.shape:
                print(f"ERROR: Shape mismatch! {method1}: {data1.shape}, {method2}: {data2.shape}")
                continue
            
            # Find differences
            diff = data1 - data2
            diff_indices = np.where(diff != 0)[0]
            
            if len(diff_indices) == 0:
                print(f"Files are identical!")
                continue
            
            print(f"Total differing values: {len(diff_indices)} out of {len(data1)} ({100*len(diff_indices)/len(data1):.2f}%)")
            print(f"Max absolute difference: {np.max(np.abs(diff)):.6e}")
            print(f"RMS difference: {np.sqrt(np.mean(diff**2)):.6e}")
            
            # Show first 100 differences
            num_to_show = min(100, len(diff_indices))
            print(f"\nFirst {num_to_show} differences:")
            print(f"{'Index':<10} {'FZG Value':<15} {'PFPL Value':<15} {'Difference':<15}")
            print(f"{'-'*65}")
            
            for k in range(num_to_show):
                idx = diff_indices[k]
                val1 = data1[idx]
                val2 = data2[idx]
                diff_val = val1 - val2
                print(f"{idx:<10} {val1:<15.6e} {val2:<15.6e} {diff_val:<15.6e}")

def compare_files(original_path: str, decompressed_paths: Dict[str, str]) -> None:
    """
    Compare original file against multiple decompressed versions.
    
    Args:
        original_path: Path to original file
        decompressed_paths: Dictionary mapping method names to file paths
    """
    print(f"Loading original file: {original_path}")
    try:
        original_data = load_f32_file(original_path)
        print(f"Original data shape: {original_data.shape}")
        print(f"Original data range: [{np.min(original_data):.6e}, {np.max(original_data):.6e}]")
    except Exception as e:
        print(f"Error loading original file: {e}")
        return
    
    print(f"\n{'='*80}")
    print(f"COMPRESSION QUALITY COMPARISON RESULTS")
    print(f"{'='*80}")
    
    results = {}
    decompressed_data_dict = {}
    
    for method_name, decompressed_path in decompressed_paths.items():
        print(f"\n--- Analyzing {method_name} ---")
        print(f"File: {decompressed_path}")
        
        try:
            # Load decompressed data
            decompressed_data = load_f32_file(decompressed_path)
            print(f"Decompressed data shape: {decompressed_data.shape}")
            
            # Store for inter-method comparison
            decompressed_data_dict[method_name] = decompressed_data
            
            # Check if shapes match
            if original_data.shape != decompressed_data.shape:
                print(f"ERROR: Shape mismatch! Original: {original_data.shape}, Decompressed: {decompressed_data.shape}")
                continue
            
            # Calculate metrics
            metrics = calculate_metrics(original_data, decompressed_data)
            results[method_name] = metrics
            
            # Display results
            print(f"PSNR: {metrics['psnr']:.2f} dB")
            print(f"MSE: {metrics['mse']:.6e}")
            print(f"RMSE: {metrics['rmse']:.6e}")
            print(f"MAE: {metrics['mae']:.6e}")
            print(f"Max Error: {metrics['max_error']:.6e}")
            print(f"Relative RMSE: {metrics['relative_rmse']:.6f}")
            print(f"Correlation: {metrics['correlation']:.6f}")
            print(f"Reconstructed range: [{metrics['reconstructed_range'][0]:.6e}, {metrics['reconstructed_range'][1]:.6e}]")
            
            # Detailed analysis for problematic cases
            if metrics['psnr'] < 60:  # Arbitrary threshold for "problematic"
                analyze_differences(original_data, decompressed_data, method_name)
                
        except Exception as e:
            print(f"Error processing {method_name}: {e}")
    
    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*80}")
        print(f"SUMMARY COMPARISON")
        print(f"{'='*80}")
        
        # Sort by PSNR
        sorted_methods = sorted(results.items(), key=lambda x: x[1]['psnr'], reverse=True)
        
        print(f"{'Method':<15} {'PSNR (dB)':<12} {'RMSE':<12} {'Max Error':<12} {'Correlation':<12}")
        print(f"{'-'*70}")
        
        for method_name, metrics in sorted_methods:
            print(f"{method_name:<15} {metrics['psnr']:<12.2f} {metrics['rmse']:<12.2e} "
                  f"{metrics['max_error']:<12.2e} {metrics['correlation']:<12.6f}")
    
    # Compare decompressed files against each other
    if len(decompressed_data_dict) > 1:
        compare_decompressed_files(decompressed_data_dict)

def main():
    """Main function to run the comparison analysis."""
    # Define file paths
    data_dir = "/home/skyler/sz_compression/FZModules/data"
    original_file = os.path.join(data_dir, "CLDHGH.f32")
    
    decompressed_files = {
        "FZG": os.path.join(data_dir, "CLDHGH.f32.fzmod_fzg.fzmodx"),
        "PFPL": os.path.join(data_dir, "CLDHGH.f32.fzmod_pfpl.fzmodx")
    }
    
    # Filter out non-existent files
    existing_files = {}
    for method, path in decompressed_files.items():
        if os.path.exists(path):
            existing_files[method] = path
        else:
            print(f"Warning: File not found: {path}")
    
    if not existing_files:
        print("No decompressed files found!")
        return
    
    # Run comparison
    compare_files(original_file, existing_files)

if __name__ == "__main__":
    main()
