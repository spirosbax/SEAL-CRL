#!/usr/bin/env python3
"""
Latent vs Binary Variable Analysis

This experiment creates publication-ready scatter plots showing relationships
between latent variables and binary concept variables. The plots help visualize
which latent dimensions capture specific binary concepts.

Usage:
    python experiments/latent_vs_binary_analysis.py --var_name "Plate_49b95a7a_pickedup" --latent_dims 0 1 8
    python experiments/latent_vs_binary_analysis.py --var_name "Egg_afaaaca3_pickedup" --latent_dims 3 17 25
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Import from installed package (use pip install -e . to install)
# Repo imports
from causality.data.ithor_loader import load_causality_data
from causality.utils.utils import setup_output_dir, pretty_concept_name
from causality.visualization.plots import plot_latents_vs_binary_vertical


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Latent vs Binary Variable Analysis"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="src/data/ithor",
        help="Directory containing the iTHOR data",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use (test, train, val)",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to load (None for all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/latent_vs_binary",
        help="Directory to save results",
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Disable caching of loaded data"
    )

    # Analysis parameters
    parser.add_argument(
        "--var_name",
        type=str,
        required=True,
        help="Name of the binary variable to analyze (e.g., 'Plate_49b95a7a_pickedup')",
    )
    parser.add_argument(
        "--latent_dims",
        type=int,
        nargs="+",
        default=None,
        help="Latent dimensions to plot (default: all dimensions)",
    )
    parser.add_argument(
        "--jitter", type=float, default=0.06, help="Amount of vertical jitter for visualization"
    )
    parser.add_argument(
        "--point_size", type=float, default=4, help="Size of scatter plot points"
    )
    parser.add_argument(
        "--height_per_plot", type=float, default=2.6, help="Height of each subplot in inches"
    )
    parser.add_argument(
        "--subsample", type=int, default=None, help="Number of samples to randomly subsample"
    )
    parser.add_argument(
        "--suptitle", action="store_true", help="Add figure title"
    )
    parser.add_argument(
        "--same_xlim", action="store_true", default=True, help="Use consistent x-limits across subplots"
    )
    parser.add_argument(
        "--xlim_percentiles",
        type=float,
        nargs=2,
        default=[1, 99],
        help="Percentile range for robust x-limits",
    )
    parser.add_argument("--dpi", type=int, default=600, help="DPI for saved figures")
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed for reproducibility")

    # Output naming
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Optional suffix for output filename (e.g., '_subset')",
    )

    return parser.parse_args()


def generate_output_filename(var_name: str, suffix: str = "") -> str:
    """
    Generate output filename based on variable name.
    
    Args:
        var_name: Binary variable name (e.g., 'Plate_49b95a7a_pickedup')
        suffix: Optional suffix to add to filename
        
    Returns:
        Filename suitable for saving (e.g., 'plate_pickedup.pdf')
    """
    # Extract the main object and state from variable name
    # e.g., 'Plate_49b95a7a_pickedup' -> 'plate_pickedup'
    parts = var_name.split('_')
    
    if len(parts) >= 3:
        # Standard format: ObjectName_hex_state
        object_name = parts[0].lower()
        state = '_'.join(parts[2:]).lower()
        base_name = f"{object_name}_{state}"
    else:
        # Fallback: use the whole name
        base_name = var_name.lower().replace('_', '_')
    
    return f"{base_name}{suffix}.pdf"


def create_latent_binary_plots(
    biscuit_vars: np.ndarray,
    sim_var_name_to_data: dict,
    var_name: str,
    latent_dims: list = None,
    output_dir: str = "results/paper_plots",
    **plot_kwargs
) -> None:
    """
    Create and save latent vs binary variable plots.
    
    Args:
        biscuit_vars: Latent variables array of shape (N, D)
        sim_var_name_to_data: Dictionary mapping variable names to data arrays
        var_name: Name of the binary variable to plot
        latent_dims: List of latent dimensions to plot (default: all)
        output_dir: Directory to save plots
        **plot_kwargs: Additional arguments for plot_latents_vs_binary_vertical
    """
    print(f"Creating plots for variable: {var_name}")
    print(f"Pretty name: {pretty_concept_name(var_name)}")
    
    if var_name not in sim_var_name_to_data:
        available_vars = list(sim_var_name_to_data.keys())
        print(f"Error: Variable '{var_name}' not found in dataset.")
        print(f"Available variables: {available_vars[:10]}...")  # Show first 10
        return
    
    # Check if variable is binary
    var_data = sim_var_name_to_data[var_name]
    unique_vals = np.unique(var_data)
    if not set(unique_vals).issubset({0, 1}):
        print(f"Warning: Variable '{var_name}' is not binary. Unique values: {unique_vals}")
        print("Converting to binary (non-zero -> 1)")
        # Convert to binary if needed
        binary_data = (var_data != 0).astype(int)
        sim_var_name_to_data_copy = sim_var_name_to_data.copy()
        sim_var_name_to_data_copy[var_name] = binary_data
    else:
        sim_var_name_to_data_copy = sim_var_name_to_data
    
    # Generate output filename
    output_suffix = plot_kwargs.pop('output_suffix', '')
    output_filename = generate_output_filename(var_name, output_suffix)
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Latent dimensions: {latent_dims if latent_dims else 'all'}")
    print(f"Output path: {output_path}")
    
    # Create the plot
    fig, axes = plot_latents_vs_binary_vertical(
        Z=biscuit_vars,
        vars_df=sim_var_name_to_data_copy,
        var_name=var_name,
        latent_dims=latent_dims,
        output_path=output_path,
        **plot_kwargs
    )
    
    print(f"Plot saved successfully!")
    
    # Print some summary statistics
    var_values = sim_var_name_to_data_copy[var_name]
    n_samples = len(var_values)
    n_positive = np.sum(var_values)
    print(f"Variable statistics:")
    print(f"  Total samples: {n_samples}")
    print(f"  Positive samples (1): {n_positive} ({100*n_positive/n_samples:.1f}%)")
    print(f"  Negative samples (0): {n_samples - n_positive} ({100*(n_samples-n_positive)/n_samples:.1f}%)")


def main():
    """Main function"""
    args = parse_args()

    print("Latent vs Binary Variable Analysis")
    print("=" * 40)
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Max files: {args.max_files}")
    print(f"Output directory: {args.output_dir}")
    print(f"Variable: {args.var_name}")
    print(f"Latent dimensions: {args.latent_dims}")
    print()

    # Setup output directory
    setup_output_dir(args.output_dir)

    # Load data
    print("Loading causality data...")
    biscuit_vars, sim_vars, var_names, sim_var_name_to_data = load_causality_data(
        data_dir=args.data_dir,
        split=args.split,
        max_files=args.max_files,
        use_cache=not args.no_cache,
    )

    print(f"Loaded data shapes:")
    print(f"  BISCUIT variables: {biscuit_vars.shape}")
    print(f"  Simulator variables: {sim_vars.shape}")
    print(f"  Variable names: {len(var_names)}")
    print()

    # Prepare plot parameters
    plot_params = {
        'jitter': args.jitter,
        'point_size': args.point_size,
        'height_per_plot': args.height_per_plot,
        'subsample': args.subsample,
        'suptitle': args.suptitle,
        'same_xlim': args.same_xlim,
        'xlim_percentiles': tuple(args.xlim_percentiles),
        'dpi': args.dpi,
        'random_seed': args.random_seed,
        'output_suffix': args.output_suffix,
    }

    # Create the plots
    create_latent_binary_plots(
        biscuit_vars=biscuit_vars,
        sim_var_name_to_data=sim_var_name_to_data,
        var_name=args.var_name,
        latent_dims=args.latent_dims,
        output_dir=args.output_dir,
        **plot_params
    )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
