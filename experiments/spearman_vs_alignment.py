#!/usr/bin/env python3
"""
Spearman vs Alignment Methods Comparison

This experiment compares different methods for aligning latent variables with ground truth concepts:
1. Baseline: Spearman correlation + Hungarian assignment + Linear Regression
2. Linear: Feature Permutation Estimator with Lasso regularization
3. Logistic: Feature Permutation Estimator with Logistic Regression
4. Kernel RBF: Kernelized Permutation Estimator with RBF kernel
5. Kernel Laplacian: Kernelized Permutation Estimator with Laplacian kernel

The analysis evaluates performance across different training set sizes and multiple random seeds,
providing insights into sample efficiency and method robustness.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Repo imports
from causality.data.ithor_loader import load_causality_data
from causality.utils.utils import setup_output_dir
from causality.utils.model_evaluation import (
    create_concept_order,
    create_data_split,
    run_model_comparison,
)
from causality.utils.visualization import plot_r2_grid


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Spearman vs Alignment Methods Comparison"
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
        default="results/spearman_vs_alignment",
        help="Directory to save results",
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Disable caching of loaded data"
    )

    # Experiment parameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.002154,
        help="Regularization parameter for permutation estimators",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds for experiments",
    )
    parser.add_argument(
        "--n_train_values",
        type=int,
        nargs="+",
        default=[5, 10, 20, 40, 80, 160, 320, 640, 1280],
        help="Training set sizes to evaluate",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing",
    )
    parser.add_argument(
        "--split_seed", type=int, default=42, help="Random seed for train/test split"
    )

    # Model selection
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["baseline", "linear", "logistic", "kernel_rbf"],
        choices=["baseline", "linear", "logistic", "kernel_rbf", "kernel_laplacian"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--kernel_gamma",
        type=float,
        default=0.5,
        help="Gamma parameter for kernel methods",
    )

    # Variable selection
    parser.add_argument(
        "--variable_types",
        type=str,
        default="binary",
        choices=["binary", "continuous", "all"],
        help="Types of variables to include (binary, continuous, all)",
    )

    # Visualization options
    parser.add_argument(
        "--figsize_per_ax",
        type=float,
        nargs=2,
        default=[3.4, 2.8],
        help="Figure size per subplot (width height)",
    )
    parser.add_argument(
        "--n_cols", type=int, default=None, help="Number of columns in grid plot (auto-detect if None)"
    )
    parser.add_argument("--dpi", type=int, default=600, help="DPI for saved figures")

    return parser.parse_args()


def create_models_config(models_to_run: list, kernel_gamma: float) -> dict:
    """
    Create model configuration dictionary.

    Args:
        models_to_run: List of model names to include
        kernel_gamma: Gamma parameter for kernel methods

    Returns:
        Dict mapping model names to (kind, kernel, parameter) tuples
    """
    all_models = {
        "baseline": ("baseline", None, None),
        "linear": ("linear", None, None),
        "logistic": ("logistic", None, None),
        "kernel_rbf": ("kernel", "rbf", kernel_gamma),
        "kernel_laplacian": ("kernel", "laplacian", kernel_gamma),
    }

    # Map friendly names to full names
    name_mapping = {
        "baseline": "Baseline (Hungarian LR)",
        "linear": "Linear",
        "logistic": "Logistic",
        "kernel_rbf": "Kernel RBF",
        "kernel_laplacian": "Kernel Laplacian",
    }

    models_config = {}
    for model_key in models_to_run:
        if model_key in all_models:
            full_name = name_mapping[model_key]
            models_config[full_name] = all_models[model_key]

    return models_config


def main():
    """Main function"""
    args = parse_args()

    print("Spearman vs Alignment Methods Comparison")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Max files: {args.max_files}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models: {args.models}")
    print(f"Variable types: {args.variable_types}")
    print(f"Alpha: {args.alpha}")
    print(f"Seeds: {args.seeds}")
    print(f"Training sizes: {args.n_train_values}")
    print()

    # Setup output directory with variable type suffix
    output_dir = f"{args.output_dir}_{args.variable_types}"
    setup_output_dir(output_dir)

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

    # Create concept order and prepare data
    concept_order = create_concept_order(variable_types=args.variable_types)

    # Filter concepts that exist in the dataset
    available_concepts = [c for c in concept_order if c in sim_var_name_to_data]
    if len(available_concepts) < len(concept_order):
        missing = set(concept_order) - set(available_concepts)
        print(f"Warning: Missing concepts in dataset: {missing}")
        concept_order = available_concepts

    print(f"Using {len(concept_order)} {args.variable_types} concepts:")

    # Prepare data matrices
    X_all = biscuit_vars  # (N_total, d_vars)
    # Handle different data types appropriately
    if args.variable_types == "binary":
        Y_all = np.vstack([sim_var_name_to_data[v] for v in concept_order]).astype(
            int
        )  # (n_concepts, N_total)
    else:
        Y_all = np.vstack([sim_var_name_to_data[v] for v in concept_order]).astype(
            float
        )  # (n_concepts, N_total)

    print(f"Data matrix shapes:")
    print(f"  X_all: {X_all.shape}")
    print(f"  Y_all: {Y_all.shape}")
    print()

    # Create train/test split
    split_indices = create_data_split(
        total_samples=X_all.shape[0],
        test_fraction=args.test_fraction,
        random_seed=args.split_seed,
    )

    print(f"Data split:")
    print(f"  Test samples: {len(split_indices['test_idx'])}")
    print(f"  Training pool: {len(split_indices['train_pool_perm'])}")
    print()

    # Create model configuration
    models_config = create_models_config(args.models, args.kernel_gamma)
    # Skip logistic for non-binary targets
    if args.variable_types != "binary":
        if "Logistic" in models_config:
            print("Skipping Logistic model for non-binary variable types.")
        models_config = {k: v for k, v in models_config.items() if k != "Logistic"}
    model_names = list(models_config.keys())

    print(f"Model configuration:")
    for name, (kind, kernel, param) in models_config.items():
        print(f"  {name}: {kind}" + (f" ({kernel}, γ={param})" if kernel else ""))
    print()

    # Run model comparison
    print("Running model comparison...")
    results_df = run_model_comparison(
        X_all=X_all,
        Y_all=Y_all,
        concept_names=concept_order,
        split_indices=split_indices,
        n_train_values=args.n_train_values,
        alpha=args.alpha,
        seeds=args.seeds,
        models_config=models_config,
    )

    print(f"Collected {len(results_df)} result records")

    # Save raw results
    results_file = os.path.join(output_dir, "comparison_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")

    # Create main grid plot
    print("Creating main comparison plot...")
    grid_plot_path = os.path.join(
        output_dir, f"spearman_vs_alignment_{args.variable_types}_grid.pdf"
    )
    fig_grid = plot_r2_grid(
        records_df=results_df,
        concepts_order=concept_order,
        N_values=args.n_train_values,
        model_order=model_names,
        n_cols=args.n_cols,
        figsize_per_ax=tuple(args.figsize_per_ax),
        output_path=grid_plot_path,
    )

    # Also save as PNG
    png_path = grid_plot_path.replace(".pdf", ".png")
    fig_grid.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved grid plot to {grid_plot_path} and {png_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 30)

    summary_stats = results_df.groupby("model")["r2"].agg(["mean", "std", "count"])
    print(summary_stats)

    print(f"\nBest performing model (mean R²): {summary_stats['mean'].idxmax()}")
    print(f"Best mean R²: {summary_stats['mean'].max():.4f}")

    # Find best concept-model combinations
    best_combo = results_df.loc[results_df["r2"].idxmax()]
    print(f"\nBest individual result:")
    print(f"  Model: {best_combo['model']}")
    print(f"  Concept: {best_combo['concept']}")
    print(f"  N_train: {best_combo['n_train']}")
    print(f"  R²: {best_combo['r2']:.4f}")

    print("\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
