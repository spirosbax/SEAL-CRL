#!/usr/bin/env python3
"""
Grouped R² Analysis for iTHOR Dataset

This script performs a grouped variable analysis to compute R²-diag and R²-sep metrics
that measure how well BISCUIT latent variables capture causal variable groups vs 
individual variables.

The analysis:
1. Groups related causal variables (e.g., egg position components, stove knobs)
2. Finds the best latent dimension for each variable
3. Computes R²-diag (average R² within matched groups)
4. Computes R²-sep (average R² for non-matched latents)
5. Creates a heatmap visualization
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Repo imports
from causality.data.ithor_loader import load_causality_data
from causality.utils.utils import setup_output_dir, compute_r2
from causality.utils.r2_analysis import create_variable_groups, compute_grouped_r2_metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Grouped R² Analysis for iTHOR Dataset")
    parser.add_argument("--data_dir", type=str, default="src/data/ithor",
                       help="Directory containing the iTHOR data")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to use (test, train, val)")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to load (None for all)")
    parser.add_argument("--output_dir", type=str, default="results/grouped_r2",
                       help="Directory to save results")
    parser.add_argument("--no_cache", action="store_true",
                       help="Disable caching of loaded data")
    parser.add_argument("--figsize", type=int, nargs=2, default=[18, 6],
                       help="Figure size for heatmap (width height)")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for saved figures")
    
    return parser.parse_args()


def analyze_grouped_r2(biscuit_vars, sim_var_name_to_data, group_map, output_dir):
    """
    Perform the grouped R² analysis and create visualizations.
    
    Args:
        biscuit_vars: Array of BISCUIT latent variables (n_samples, n_dims)
        sim_var_name_to_data: Dict mapping variable names to data arrays
        group_map: Dict mapping group names to lists of variable names
        output_dir: Directory to save results
        
    Returns:
        Dict with analysis results
    """
    print("Starting grouped R² analysis...")
    
    # Get all individual variables in order
    groups = list(group_map.keys())
    x_vars = sum(group_map.values(), [])
    
    print(f"Analyzing {len(groups)} groups with {len(x_vars)} total variables")
    print(f"BISCUIT dimensions: {biscuit_vars.shape[1]}")
    
    # Compute full R² matrix (latent dimensions → real variables)
    Z = biscuit_vars
    n_dims = Z.shape[1]
    full_R2 = np.empty((n_dims, len(x_vars)))
    
    print("Computing R² matrix...")
    for d in range(n_dims):
        if d % 10 == 0:
            print(f"  Processing latent dimension {d+1}/{n_dims}")
        
        z = Z[:, d]
        for j, v in enumerate(x_vars):
            y = sim_var_name_to_data[v].astype(float)
            full_R2[d, j] = compute_r2(z, y)
    
    # Find best latent dimension for each variable
    var_to_lat = {}
    for j, v in enumerate(x_vars):
        var_to_lat[v] = np.argmax(full_R2[:, j])
    
    print("\nBest latent dimensions for each variable:")
    for v, lat_dim in var_to_lat.items():
        var_idx = x_vars.index(v)
        r2_val = full_R2[lat_dim, var_idx]
        print(f"  {v}: latent dim {lat_dim} (R² = {r2_val:.4f})")
    
    # Compute grouped metrics
    results = compute_grouped_r2_metrics(full_R2, group_map, x_vars, var_to_lat)
    
    print(f"\nGrouped R² Results:")
    print(f"  R²-diag: {results['R2_diag']:.4f}")
    print(f"  R²-sep:  {results['R2_sep']:.4f}")
    
    # Build heat matrix for visualization
    R2_heat = np.zeros((len(group_map), len(x_vars)))
    for i, (g, vars_in_g) in enumerate(group_map.items()):
        # For single variable groups, use its latent dim for all variables
        if len(vars_in_g) == 1:
            v = vars_in_g[0]
            lat_dim = var_to_lat[v]
            R2_heat[i, :] = full_R2[lat_dim, :]  # Fill entire row
        else:
            # For multi-variable groups, just fill in the grouped variables
            for v in vars_in_g:
                lat_dim = var_to_lat[v]
                var_idx = x_vars.index(v)
                R2_heat[i, var_idx] = full_R2[lat_dim, var_idx]
    
    # Save results
    results.update({
        'full_R2': full_R2,
        'R2_heat': R2_heat,
        'var_to_lat': var_to_lat,
        'groups': groups,
        'x_vars': x_vars,
        'group_map': group_map
    })
    
    # Save numerical results
    results_file = os.path.join(output_dir, 'grouped_r2_results.npz')
    np.savez_compressed(
        results_file,
        full_R2=full_R2,
        R2_heat=R2_heat,
        R2_diag=results['R2_diag'],
        R2_sep=results['R2_sep'],
        diag_vals=results['diag_vals'],
        sep_vals=results['sep_vals'],
        groups=groups,
        x_vars=x_vars
    )
    print(f"Saved numerical results to {results_file}")
    
    return results


def create_grouped_heatmap(results, output_dir, figsize=(18, 6), dpi=300):
    """Create and save the grouped R² heatmap"""
    print("Creating grouped R² heatmap...")
    
    R2_heat = results['R2_heat']
    groups = results['groups']
    x_vars = results['x_vars']
    R2_diag = results['R2_diag']
    R2_sep = results['R2_sep']
    
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        R2_heat,
        xticklabels=x_vars,
        yticklabels=groups,
        cmap="rocket",  # Reversed colormap so black=1, white=0
        annot=True,
        fmt=".2f", 
        vmin=0,  # Force range from 0 to 1
        vmax=1,
        cbar_kws={"label": "Absolute R²"},
    )
    ax.set_xticklabels(x_vars, rotation=45, ha='right')
    ax.set_xlabel("Ground Truth Causal Variables")
    ax.set_ylabel("Learned Latent Variables (one per group)")
    ax.set_title(f"Grouped R² Heatmap —  R²-diag={R2_diag:.3f},  R²-sep={R2_sep:.3f}")
    plt.tight_layout()
    
    # Save figure
    heatmap_file = os.path.join(output_dir, 'grouped_r2_heatmap.png')
    plt.savefig(heatmap_file, dpi=dpi, bbox_inches='tight')
    print(f"Saved heatmap to {heatmap_file}")
    
    # Also save as PDF for papers
    heatmap_pdf = os.path.join(output_dir, 'grouped_r2_heatmap.pdf')
    plt.savefig(heatmap_pdf, bbox_inches='tight')
    print(f"Saved heatmap to {heatmap_pdf}")
    
    plt.show()


def main():
    """Main function"""
    args = parse_args()
    
    print("Grouped R² Analysis for iTHOR Dataset")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Max files: {args.max_files}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Setup output directory
    setup_output_dir(args.output_dir)
    
    # Load data
    print("Loading causality data...")
    biscuit_vars, sim_vars, var_names, sim_var_name_to_data = load_causality_data(
        data_dir=args.data_dir,
        split=args.split,
        max_files=args.max_files,
        use_cache=not args.no_cache
    )
    
    print(f"Loaded data shapes:")
    print(f"  BISCUIT variables: {biscuit_vars.shape}")
    print(f"  Simulator variables: {sim_vars.shape}")
    print(f"  Variable names: {len(var_names)}")
    print()
    
    # Create variable groups
    group_map = create_variable_groups()
    
    # Verify all variables exist in the dataset
    all_group_vars = sum(group_map.values(), [])
    missing_vars = [v for v in all_group_vars if v not in sim_var_name_to_data]
    if missing_vars:
        print(f"Warning: Missing variables in dataset: {missing_vars}")
        # Filter out missing variables
        filtered_group_map = {}
        for group_name, vars_in_group in group_map.items():
            filtered_vars = [v for v in vars_in_group if v in sim_var_name_to_data]
            if filtered_vars:  # Only include groups with at least one variable
                filtered_group_map[group_name] = filtered_vars
        group_map = filtered_group_map
        print(f"Using filtered groups: {list(group_map.keys())}")
    
    # Perform analysis
    results = analyze_grouped_r2(biscuit_vars, sim_var_name_to_data, group_map, args.output_dir)
    
    # Create visualization
    create_grouped_heatmap(results, args.output_dir, 
                          figsize=tuple(args.figsize), dpi=args.dpi)
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
