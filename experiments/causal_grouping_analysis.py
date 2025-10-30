#!/usr/bin/env python3
"""
Causal Grouping Analysis

This experiment evaluates different models on aggregated causal predicates.
It includes both baseline models (Linear/Logistic Regression, MLP) and
advanced permutation estimators (Linear/Logistic/Kernel with group-lasso).

The analysis supports both baseline evaluation and N→1 alignment with
capacitated matching for understanding which latent variables capture
specific causal concepts.

Usage:
    # Run baseline analysis with single predicate
    python experiments/causal_grouping_analysis.py --mode baseline --predicates all_stoves_on

    # Run with multiple non-overlapping predicates
    python experiments/causal_grouping_analysis.py --mode baseline --predicates at_least_two_stoves_on egg_intact
    python experiments/causal_grouping_analysis.py --mode both --predicates at_least_three_stoves_on egg_intact --other_concept Microwave_d8b935e4_open

    # Run estimator analysis only  
    python experiments/causal_grouping_analysis.py --mode estimators --predicates all_stoves_on --other_concept Microwave_d8b935e4_open

    # Multiple predicates with custom output
    python experiments/causal_grouping_analysis.py --mode both --predicates at_least_three_stoves_on egg_intact --other_concept Microwave_d8b935e4_open --output_dir results/causal_grouping_multiple
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

# Repo imports
from causality.data.ithor_loader import load_causality_data
from causality.utils.utils import setup_output_dir, pretty_concept_name
from causality.utils.baseline_models import (
    create_train_test_split, create_baseline_models, evaluate_baseline_models,
    summarize_baseline_results, inspect_logistic_weights, create_aggregate_predicate,
    get_stove_configuration, get_predicate_configuration, get_supported_predicates,
    validate_predicate_combination
)
from causality.utils.estimator_models import (
    evaluate_estimator_models, summarize_estimator_results, evaluate_estimator_models_multiple,
    capacitated_match_multiple
)
from causality.visualization.plots import (
    plot_baseline_comparison, plot_estimator_comparison, plot_latent_heatmap_grouped
)
from causality.utils.visualization import plot_r2_grid
from causality.utils.math_utils import r2_safe


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Causal Grouping Analysis"
    )
    
    # Data parameters
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
        default="results/causal_grouping",
        help="Directory to save results",
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="Disable caching of loaded data"
    )

    # Experiment mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "estimators", "both"],
        default="both",
        help="Which analysis to run",
    )

    # Predicate configuration
    parser.add_argument(
        "--predicates",
        type=str,
        nargs="+",
        default=["all_stoves_on"],
        choices=get_supported_predicates(),
        help="Which aggregate predicates to evaluate (can specify multiple)",
    )
    parser.add_argument(
        "--other_concept",
        type=str,
        default="Microwave_d8b935e4_open",
        help="Second concept for N→1 alignment (required for estimator mode)",
    )

    # Model parameters
    parser.add_argument(
        "--mlp_hidden_sizes",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Hidden layer sizes for MLP models",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.154e-3,
        help="Regularization parameter for permutation estimators",
    )
    parser.add_argument(
        "--kernel_gamma",
        type=float,
        default=0.5,
        help="Gamma parameter for RBF kernel",
    )
    parser.add_argument(
        "--max_nystrom",
        type=int,
        default=64,
        help="Maximum Nyström components per latent",
    )

    # Experiment parameters
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
        default=[5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240],
        help="Training set sizes to evaluate",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing",
    )
    parser.add_argument(
        "--split_seed", 
        type=int, 
        default=0, 
        help="Random seed for train/test split"
    )

    # Class balancing parameters
    parser.add_argument(
        "--balance_classes",
        action="store_true",
        help="Balance classes by undersampling majority class for closer to 50-50 split",
    )
    parser.add_argument(
        "--target_ratio",
        type=float,
        default=0.5,
        help="Target ratio for minority class when balancing (0.5 = 50-50 split)",
    )

    # Visualization parameters
    parser.add_argument("--dpi", type=int, default=600, help="DPI for saved figures")
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress information"
    )

    return parser.parse_args()


def run_baseline_analysis_multiple(
    predicates: List[str],
    biscuit_vars: np.ndarray,
    sim_var_name_to_data: dict,
    args: argparse.Namespace,
    output_dir: str
) -> pd.DataFrame:
    """
    Capacitated baseline (Spearman + capacitated assignment + multi-variate LR) for
    multiple predicates plus the other concept. Publication-ready plotting included.
    """
    print("Running capacitated baseline for multiple predicates and other concept...")

    # Build concept list and capacities (same as estimators)
    chosen_concepts = []
    K_assignments = {}
    for predicate in predicates:
        predicate_config = get_predicate_configuration(predicate)
        predicate_vars = predicate_config['variables']
        agg_name = predicate.replace('_', ' ').title().replace(' ', '_')
        chosen_concepts.append(agg_name)
        if predicate in ["all_stoves_on", "at_least_two_stoves_on", "at_least_three_stoves_on"]:
            K_assignments[agg_name] = len(predicate_vars)
        else:
            K_assignments[agg_name] = 2
        y_all_predicate = create_aggregate_predicate(sim_var_name_to_data, predicate, predicate_vars)
        sim_var_name_to_data[agg_name] = y_all_predicate.astype(int)

    other_name = args.other_concept
    chosen_concepts.append(other_name)
    K_assignments[other_name] = 1

    # Unified split based on first predicate
    first_predicate = predicates[0]
    first_cfg = get_predicate_configuration(first_predicate)
    y_first = create_aggregate_predicate(sim_var_name_to_data, first_predicate, first_cfg['variables'])
    split_dict = create_train_test_split(
        y_all=y_first,
        test_size=args.test_fraction,
        random_state=args.split_seed,
        balance_classes=args.balance_classes,
        target_ratio=args.target_ratio,
        verbose=False
    )

    # Prepare matrices
    X_all = biscuit_vars  # (N, d)
    Y_all = np.vstack([sim_var_name_to_data[c] for c in chosen_concepts]).astype(int)  # (m, N)

    test_idx = split_dict["test_idx"]
    train_pool = split_dict["train_pool_perm"]

    # Fixed test matrices (dims-first)
    X_test = X_all[test_idx].T  # (d, N_test)
    Y_test = Y_all[:, test_idx]

    rows = []
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        permuted_pool = rng.permutation(train_pool)
        Ns = [N for N in args.n_train_values if N <= len(permuted_pool)]

        for N in Ns:
            subset = permuted_pool[:N]
            X_train = X_all[subset].T  # (d, N)
            Y_train = Y_all[:, subset]  # (m, N)

            # Standardize X for correlation and LR
            scaler = StandardScaler().fit(X_train.T)
            Xtr_std = scaler.transform(X_train.T).T
            Xte_std = scaler.transform(X_test.T).T

            # Spearman correlation matrix (abs)
            m, d = Y_train.shape[0], Xtr_std.shape[0]
            C = np.zeros((m, d), dtype=float)
            for i in range(m):
                y = Y_train[i]
                if np.unique(y).size < 2:
                    continue
                for j in range(d):
                    rho, _ = spearmanr(y, Xtr_std[j, :])
                    C[i, j] = 0.0 if np.isnan(rho) else abs(rho)

            # Capacitated assignment
            assignments = capacitated_match_multiple(C, chosen_concepts, K_assignments)

            # Predict per concept using assigned latents (multi-variate LR)
            for i, concept in enumerate(chosen_concepts):
                cols = assignments.get(concept, [])
                if len(cols) == 0 or np.unique(Y_train[i]).size < 2 or np.std(Y_test[i]) == 0:
                    r2v = np.nan
                else:
                    lm = LinearRegression().fit(Xtr_std[cols, :].T, Y_train[i])
                    y_hat = lm.predict(Xte_std[cols, :].T)
                    r2v = r2_safe(Y_test[i], y_hat)

                rows.append({
                    "model": "Baseline (cap-LR)",
                    "concept": concept,
                    "n_train": int(N),
                    "seed": int(seed),
                    "r2": float(r2v) if r2v is not None else np.nan,
                })

    df_baseline = pd.DataFrame(rows)

    # Save raw results
    baseline_results_file = os.path.join(output_dir, "baseline_results.csv")
    df_baseline.to_csv(baseline_results_file, index=False)
    print(f"Saved baseline results to {baseline_results_file}")

    # Publication-ready plot
    baseline_plot_path = os.path.join(output_dir, "baseline_comparison.pdf")
    plot_r2_grid(
        records_df=df_baseline,
        concepts_order=chosen_concepts,
        N_values=args.n_train_values,
        model_order=["Baseline (cap-LR)"],
        n_cols=len(chosen_concepts),
        output_path=baseline_plot_path,
    )

    return df_baseline


def run_estimator_analysis(
    biscuit_vars: np.ndarray,
    sim_var_name_to_data: dict,
    split_dict: dict,
    args: argparse.Namespace,
    output_dir: str
) -> tuple:
    """
    Run permutation estimator analysis with N→1 alignment.
    
    Args:
        biscuit_vars: Full latent variable matrix
        sim_var_name_to_data: Dictionary of simulator variables
        split_dict: Train/test split indices (already balanced if balancing was applied)
        args: Command line arguments
        output_dir: Output directory path
        
    Returns:
        Tuple of (results DataFrame, picks_by_seed dictionary)
    """
    print("Running permutation estimator analysis...")
    
    # Prepare data for N→1 alignment based on predicate type
    predicate_config = get_predicate_configuration(args.predicate)
    predicate_vars = predicate_config['variables']
    K_agg = len(predicate_vars)
    
    # Create a display name for the predicate
    agg_name = args.predicate.replace('_', ' ').title().replace(' ', '_')
    other_name = args.other_concept
    chosen_concepts = [agg_name, other_name]
    
    # Create aggregate predicate
    y_all_predicate = create_aggregate_predicate(sim_var_name_to_data, args.predicate, predicate_vars)
    
    # Prepare data matrices (split_dict already contains balanced indices if balancing was applied)
    X_all = biscuit_vars.copy()  # Full latent space
    sim_var_name_to_data[agg_name] = y_all_predicate.astype(int)
    Y_all = np.vstack([sim_var_name_to_data[c] for c in chosen_concepts]).astype(float)
    
    print(f"Data shapes for estimator analysis:")
    print(f"  X_all: {X_all.shape}")  
    print(f"  Y_all: {Y_all.shape}")
    print(f"  Concepts: {chosen_concepts}")
    
    # Evaluate estimator models
    df_estimators, picks_by_seed = evaluate_estimator_models(
        X_all=X_all,
        Y_all=Y_all,
        concept_names=chosen_concepts,
        agg_name=agg_name,
        K_agg=K_agg,
        split_dict=split_dict,
        N_train_values=args.n_train_values,
        alpha_fixed=args.alpha,
        kernel_gamma=args.kernel_gamma,
        max_nystrom=args.max_nystrom,
        seeds=args.seeds,
        verbose=args.verbose
    )
    
    # Save raw results
    estimator_results_file = os.path.join(output_dir, "estimator_results.csv")
    df_estimators.to_csv(estimator_results_file, index=False)
    print(f"Saved estimator results to {estimator_results_file}")
    
    # Summarize results
    summary_estimators = summarize_estimator_results(df_estimators)
    
    # Create publication-ready estimator comparison grid using shared color map
    estimator_plot_path = os.path.join(output_dir, "estimator_comparison.pdf")
    model_order = ["Baseline (cap-LR)", "Linear", "Logistic", "Kernel RBF"]
    plot_r2_grid(
        records_df=df_estimators,
        concepts_order=chosen_concepts,
        N_values=args.n_train_values,
        model_order=model_order,
        n_cols=len(chosen_concepts),
        output_path=estimator_plot_path,
    )
    
    # Create latent assignment heatmap
    heatmap_plot_path = os.path.join(output_dir, "latent_assignment_heatmap.pdf")
    plot_latent_heatmap_grouped(
        picks_by_seed=picks_by_seed,
        concept=agg_name,
        N_vals=args.n_train_values,
        seeds=args.seeds,
        K=K_agg,
        output_path=heatmap_plot_path,
        dpi=args.dpi
    )
    
    return df_estimators, picks_by_seed


def run_estimator_analysis_multiple(
    predicates: List[str],
    biscuit_vars: np.ndarray,
    sim_var_name_to_data: dict,
    args: argparse.Namespace,
    output_dir: str
) -> tuple:
    """
    Run permutation estimator analysis with unified capacitated matching for multiple predicates.
    
    Args:
        predicates: List of predicate names to analyze
        biscuit_vars: Full latent variable matrix
        sim_var_name_to_data: Dictionary of simulator variables
        args: Command line arguments
        output_dir: Output directory path
        
    Returns:
        Tuple of (results DataFrame, picks_by_seed dictionary for each predicate)
    """
    print("Running permutation estimator analysis with unified capacitated matching...")
    
    # Prepare concept list: all predicates + other concept
    chosen_concepts = []
    K_assignments = {}  # Track K values for each predicate
    
    # Add all predicates as concepts
    for predicate in predicates:
        predicate_config = get_predicate_configuration(predicate)
        predicate_vars = predicate_config['variables']
        
        # Create display name
        agg_name = predicate.replace('_', ' ').title().replace(' ', '_')
        chosen_concepts.append(agg_name)
        
        # Set K value for capacitated matching
        if predicate in ["all_stoves_on", "at_least_two_stoves_on", "at_least_three_stoves_on"]:
            K_assignments[agg_name] = len(predicate_vars)  # Number of stove knobs (4)
        else:
            K_assignments[agg_name] = 2  # Default for other predicates
        
        # Create aggregate predicate and add to data
        y_all_predicate = create_aggregate_predicate(sim_var_name_to_data, predicate, predicate_vars)
        sim_var_name_to_data[agg_name] = y_all_predicate.astype(int)
    
    # Add other concept
    other_name = args.other_concept
    chosen_concepts.append(other_name)
    K_assignments[other_name] = 1  # Other concept gets 1 assignment
    
    print(f"Unified capacitated matching for ALL concepts: {chosen_concepts}")
    print(f"K assignments: {K_assignments}")
    
    # Create unified train/test split based on first predicate for consistency
    first_predicate = predicates[0]
    first_predicate_config = get_predicate_configuration(first_predicate)
    first_predicate_vars = first_predicate_config['variables']
    y_first_predicate = create_aggregate_predicate(sim_var_name_to_data, first_predicate, first_predicate_vars)
    
    split_dict = create_train_test_split(
        y_all=y_first_predicate,
        test_size=args.test_fraction,
        random_state=args.split_seed,
        balance_classes=args.balance_classes,
        target_ratio=args.target_ratio,
        verbose=False
    )
    
    # Prepare data matrices for ALL concepts in unified analysis
    X_all = biscuit_vars.copy()  # Full latent space
    Y_all = np.vstack([sim_var_name_to_data[c] for c in chosen_concepts]).astype(float)
    
    print(f"Data shapes for unified estimator analysis:")
    print(f"  X_all: {X_all.shape}")  
    print(f"  Y_all: {Y_all.shape}")
    print(f"  Concepts: {chosen_concepts}")
    
    # Use the new unified capacitated matching function
    df_estimators, picks_by_seed = evaluate_estimator_models_multiple(
        X_all=X_all,
        Y_all=Y_all,
        concept_names=chosen_concepts,
        K_assignments=K_assignments,
        split_dict=split_dict,
        N_train_values=args.n_train_values,
        alpha_fixed=args.alpha,
        kernel_gamma=args.kernel_gamma,
        max_nystrom=args.max_nystrom,
        max_n_value=1280,
        seeds=args.seeds,
        verbose=args.verbose
    )
    
    # Save raw results
    estimator_results_file = os.path.join(output_dir, "estimator_results.csv")
    df_estimators.to_csv(estimator_results_file, index=False)
    print(f"Saved estimator results to {estimator_results_file}")
    
    # Summarize results
    summary_estimators = summarize_estimator_results(df_estimators)
    
    # Create publication-ready estimator comparison grid using shared color map
    estimator_plot_path = os.path.join(output_dir, "estimator_comparison.pdf")
    model_order = ["Baseline (cap-LR)", "Linear", "Logistic", "Kernel RBF"]
    plot_r2_grid(
        records_df=df_estimators,
        concepts_order=chosen_concepts,
        N_values=args.n_train_values,
        model_order=model_order,
        n_cols=len(chosen_concepts),
        output_path=estimator_plot_path,
    )
    
    # Create individual heatmaps for each predicate with K > 1
    for predicate in predicates:
        agg_name = predicate.replace('_', ' ').title().replace(' ', '_')
        if K_assignments[agg_name] > 1:  # Only create heatmaps for grouped predicates
            heatmap_plot_path = os.path.join(output_dir, f"latent_assignment_heatmap_{predicate}.pdf")
            plot_latent_heatmap_grouped(
                picks_by_seed=picks_by_seed,
                concept=agg_name,
                N_vals=args.n_train_values,
                seeds=args.seeds,
                K=K_assignments[agg_name],
                output_path=heatmap_plot_path,
                dpi=args.dpi
            )
    
    # Create heatmap for the single other concept (if it exists)
    if args.other_concept:
        # Find the matching concept key (handles case variations)
        matching_key = None
        for key in K_assignments.keys():
            if key.lower() == args.other_concept.lower():
                matching_key = key
                break
        
        if matching_key:
            heatmap_plot_path = os.path.join(output_dir, f"latent_assignment_heatmap_{args.other_concept}.pdf")
            plot_latent_heatmap_grouped(
                picks_by_seed=picks_by_seed,
                concept=matching_key,
                N_vals=args.n_train_values,
                seeds=args.seeds,
                K=K_assignments[matching_key],
                output_path=heatmap_plot_path,
                dpi=args.dpi
            )
    
    return df_estimators, picks_by_seed


def main():
    """Main function"""
    args = parse_args()

    print("Causal Grouping Analysis")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Max files: {args.max_files}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Predicates: {args.predicates}")
    if args.mode in ["estimators", "both"]:
        print(f"Other concept: {args.other_concept}")
    print(f"Seeds: {args.seeds}")
    print(f"Training sizes: {args.n_train_values}")
    print(f"Balance classes: {args.balance_classes}")
    if args.balance_classes:
        print(f"Target ratio: {args.target_ratio}")
    print()

    # Validate predicate combination
    try:
        validate_predicate_combination(args.predicates)
        print(f"✓ Predicate combination validated: {args.predicates}")
    except ValueError as e:
        print(f"✗ Error: {e}")
        return
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

    # Run analyses based on mode
    results = {}
    
    if args.mode in ["baseline", "both"]:
        results['baseline'] = run_baseline_analysis_multiple(
            predicates=args.predicates,
            biscuit_vars=biscuit_vars,
            sim_var_name_to_data=sim_var_name_to_data,
            args=args,
            output_dir=args.output_dir
        )
    
    if args.mode in ["estimators", "both"]:
        if args.other_concept not in sim_var_name_to_data:
            print(f"Error: Other concept '{args.other_concept}' not found in dataset.")
            print(f"Available concepts: {list(sim_var_name_to_data.keys())[:10]}...")
            return
        
        results['estimators'], results['picks'] = run_estimator_analysis_multiple(
            predicates=args.predicates,
            biscuit_vars=biscuit_vars,
            sim_var_name_to_data=sim_var_name_to_data,
            args=args,
            output_dir=args.output_dir
        )
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 30)
    
    if 'baseline' in results:
        print("\nBaseline Model Performance:")
        summary_stats = results['baseline'].groupby("model")["r2"].agg(["mean", "std", "count"])
        print(summary_stats)
        
        best_model = summary_stats['mean'].idxmax()
        best_r2 = summary_stats['mean'].max()
        print(f"Best baseline model: {best_model} (mean R²: {best_r2:.4f})")
    
    if 'estimators' in results:
        print("\nPermutation Estimator Performance:")
        estimator_stats = results['estimators'].groupby("model")["r2"].agg(["mean", "std", "count"])
        print(estimator_stats)
        
        best_estimator = estimator_stats['mean'].idxmax()
        best_est_r2 = estimator_stats['mean'].max()
        print(f"Best estimator model: {best_estimator} (mean R²: {best_est_r2:.4f})")

    print(f"\nAnalysis complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
