#!/usr/bin/env python3
"""
R² Analysis: Correlation between Causal Variables and Latent Dimensions

This script analyzes the correlation between causal variables and BISCUIT latent dimensions
by computing R² scores and visualizing them as a heatmap.

Usage:
    python r2_concepts_vs_latents.py --data_dir ../src/data/ithor_indep --max_files 10
    
Requirements:
    - Activate cbmbiscuit environment first: conda activate cbmbiscuit
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Setup path for clean project imports
# Repo imports
from causality.data.ithor_loader import load_causality_data
from causality.utils.utils import compute_r2, pretty_concept_name
from causality.visualization.plots import plot_r2_matrix


def main():
    parser = argparse.ArgumentParser(description='Analyze R² correlation between causal variables and latent dimensions')
    parser.add_argument('--data_dir', type=str, default='../src/data/ithor',
                        help='Path to iTHOR dataset directory')
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Output directory for plots')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to analyze (default: test)')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of sequence files to load (None for all)')
    parser.add_argument('--fig_width', type=int, default=12,
                        help='Figure width')
    parser.add_argument('--max_height', type=int, default=14,
                        help='Maximum figure height')
    parser.add_argument('--output_name', type=str, default='r2_concepts_vs_latents.pdf',
                        help='Output filename')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all causality data using the all-inclusive function
    try:
        biscuit_vars, sim_vars, all_vars, sim_var_name_to_data = load_causality_data(
            data_dir=args.data_dir,
            split=args.split,
            max_files=args.max_files,
            use_cache=True
        )
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
    
    # Build R² matrix: rows=dims, cols=variables
    print("Computing R² correlations...")
    n_vars = len(all_vars)
    n_dims = biscuit_vars.shape[1]

    r2_matrix = np.zeros((n_dims, n_vars))
    for j, var_name in enumerate(all_vars):
        if j % 5 == 0:
            print(f"  Processing variable {j+1}/{n_vars}: {pretty_concept_name(var_name)}")
        y = sim_var_name_to_data[var_name].astype(float)
        for d in range(n_dims):
            r2_matrix[d, j] = compute_r2(biscuit_vars[:, d], y)

    # Sort variables by their best dim (keeps previous ordering logic)
    best_dims = np.argmax(r2_matrix, axis=0)
    order = np.argsort(best_dims)
    r2_matrix = r2_matrix[:, order]
    all_vars = [all_vars[i] for i in order]
    
    # Transpose for plotting: concepts on Y, dims on X
    r2_plot = r2_matrix.T  # shape: (n_vars, n_dims)
    
    # Create output path and plot
    output_path = os.path.join(args.output_dir, args.output_name)
    print(f"Creating visualization...")
    plot_r2_matrix(r2_plot, all_vars, output_path, args.fig_width, args.max_height)
    
    # Print some statistics
    print("\n" + "="*50)
    print("R² Analysis Results:")
    print("="*50)
    print(f"  Number of latent dimensions: {n_dims}")
    print(f"  Number of causal variables: {n_vars}")
    print(f"  Max R² value: {np.max(r2_matrix):.3f}")
    print(f"  Mean R² value: {np.mean(r2_matrix):.3f}")
    
    # Print best correlations
    print("\nTop 5 variable-dimension correlations:")
    flat_indices = np.argsort(r2_matrix.ravel())[-5:]
    for idx in reversed(flat_indices):
        dim_idx, var_idx = np.unravel_index(idx, r2_matrix.shape)
        var_name = all_vars[var_idx]
        r2_val = r2_matrix[dim_idx, var_idx]
        print(f"  {pretty_concept_name(var_name)} ↔ Dim {dim_idx}: R² = {r2_val:.3f}")
    
    print("="*50)
    print("Analysis complete!")


if __name__ == "__main__":
    main()