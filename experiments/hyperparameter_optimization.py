#!/usr/bin/env python3
"""
Hyperparameter Optimization Script for Permutation Estimator

This script performs comprehensive hyperparameter optimization for the alpha parameter
across all available variables (both binary and continuous) using cross-validation.

Usage:
    python experiments/hyperparameter_optimization.py --data_dir src/data/ithor --split test
    python experiments/hyperparameter_optimization.py --variable_types all --alpha_range -4 1 --n_alpha 20
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, roc_auc_score, accuracy_score, mean_squared_error
from sklearn.model_selection import KFold
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Repo imports
from causality.data.ithor_loader import load_causality_data
from causality.utils.utils import setup_output_dir
from causality.utils.model_evaluation import get_all_variable_info, create_concept_order, create_data_split

# External permutation estimators
from permutation_estimator.estimator import FeaturePermutationEstimator

# =============================================================
# Argument parsing
# =============================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization for Permutation Estimator"
    )
    
    # Data parameters
    parser.add_argument(
        "--data_dir", type=str, default="src/data/ithor",
        help="Directory containing the iTHOR data"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "test", "val"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--max_files", type=int, default=None,
        help="Maximum number of files to load (None for all)"
    )
    parser.add_argument(
        "--no_cache", action="store_true",
        help="Force reload from raw data instead of using cache"
    )
    
    # Hyperparameter search parameters
    parser.add_argument(
        "--alpha_range", type=float, nargs=2, default=[-4, 1],
        help="Log10 range for alpha values (min max)"
    )
    parser.add_argument(
        "--n_alpha", type=int, default=20,
        help="Number of alpha values to test"
    )
    
    # Cross-validation parameters
    parser.add_argument(
        "--n_folds", type=int, default=5,
        help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--n_seeds", type=int, default=3,
        help="Number of random seeds for robust evaluation"
    )
    parser.add_argument(
        "--test_fraction", type=float, default=0.2,
        help="Fraction of data to use for final test set"
    )
    parser.add_argument(
        "--split_seed", type=int, default=42,
        help="Random seed for train/test split"
    )
    
    # Variable selection
    parser.add_argument(
        "--variable_types", type=str, default="all",
        choices=["binary", "continuous", "all"],
        help="Types of variables to include in optimization"
    )
    
    # Training efficiency analysis
    parser.add_argument(
        "--n_train_values", type=int, nargs="*", default=[40, 80, 160, 320],
        help="Training sizes for efficiency analysis"
    )
    parser.add_argument(
    "--skip_efficiency", action="store_true",
        help="Skip training efficiency analysis (faster execution)"
    )
    
    # Output parameters
    parser.add_argument(
        "--output_dir", type=str, default="results/hyperparameter_optimization",
        help="Output directory for results"
    )
    parser.add_argument(
        "--save_plots", action="store_true",
        help="Save visualization plots"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for saved plots"
    )
    
    return parser.parse_args()

def get_variable_type(variable_name):
    """Determine variable type from the variable info."""
    var_info = get_all_variable_info()
    if variable_name in var_info:
        var_type = var_info[variable_name]
        if var_type == 'categ_2':
            return 'binary'
        elif var_type.startswith('continuous'):
            return 'continuous'
        else:
            return 'continuous'  # Default to continuous for unknown types
    return 'continuous'  # Default fallback

def determine_regularizer(variable_type):
    """Determine appropriate regularizer based on variable type."""
    if variable_type == 'binary':
        return "logistic_group"
    else:  # continuous
        return "group"

def evaluate_model(est, X_test, Y_test, variable_type, is_binary_target=False):
    """Evaluate model performance with appropriate metrics."""
    Y_hat = est.predict_match(X_test)
    
    metrics = {}
    
    if is_binary_target:
        # Binary target metrics
        Y_test_binary = (Y_test > 0).astype(int) if not np.all(np.isin(Y_test, [0, 1])) else Y_test.astype(int)
        Y_hat_binary = (Y_hat > 0).astype(int)
        
        # R² for continuous predictions
        metrics['r2'] = r2_score(Y_test, Y_hat)
        
        # Binary classification metrics
        try:
            metrics['roc_auc'] = roc_auc_score(Y_test_binary, Y_hat)
        except:
            metrics['roc_auc'] = np.nan
            
        metrics['accuracy'] = accuracy_score(Y_test_binary, Y_hat_binary)
        
    else:
        # Continuous target metrics
        metrics['r2'] = r2_score(Y_test, Y_hat)
        metrics['mse'] = mean_squared_error(Y_test, Y_hat)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
    return metrics

def cross_validate_alpha(X, Y, variable_name, variable_type, alpha, n_folds, n_train=None):
    """Perform cross-validation for a given alpha value."""
    
    if n_train is not None and n_train < X.shape[1]:
        # Subsample for efficiency analysis
        idx = np.random.choice(X.shape[1], n_train, replace=False)
        X = X[:, idx]
        Y = Y[:, idx]
    
    regularizer = determine_regularizer(variable_type)
    is_binary_target = variable_type == 'binary'
    
    # Prepare Y for logistic regression if binary
    if is_binary_target and regularizer == "logistic_group":
        Y_processed = 2 * Y.astype(int) - 1  # Convert 0/1 to -1/+1
    else:
        Y_processed = Y.astype(float)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X.T)):
        try:
            X_train = X[:, train_idx]
            X_val = X[:, val_idx]
            Y_train = Y_processed[:, train_idx] if Y_processed.ndim > 1 else Y_processed[train_idx]
            Y_val = Y_processed[:, val_idx] if Y_processed.ndim > 1 else Y_processed[val_idx]
            
            # Ensure Y_train has correct shape for estimator
            if Y_train.ndim == 1:
                Y_train = Y_train.reshape(1, -1)
            if Y_val.ndim == 1:
                Y_val = Y_val.reshape(1, -1)
            
            est = FeaturePermutationEstimator(
                regularizer=regularizer,
                optim_kwargs={"alpha": alpha},
                feature_transform=None,
                d_variables=X_train.shape[0],
                n_features=1
            )
            
            est.fit(X_train, Y_train)
            fold_metrics.append(evaluate_model(est, X_val, Y_val, variable_type, is_binary_target))
            
        except Exception as e:
            print(f"Error in fold {fold} for {variable_name}, alpha={alpha}: {e}")
            fold_metrics.append({})
    
    # Average metrics across folds
    if fold_metrics and any(fold_metrics):
        avg_metrics = {}
        for key in fold_metrics[0].keys():
            values = [m.get(key, np.nan) for m in fold_metrics if key in m]
            avg_metrics[key] = np.nanmean(values)
            avg_metrics[f'{key}_std'] = np.nanstd(values)
    else:
        avg_metrics = {}
    
    return avg_metrics

def run_hyperparameter_optimization(args):
    """Main hyperparameter optimization function."""
    
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
    
    # Create alpha values
    alpha_values = np.logspace(args.alpha_range[0], args.alpha_range[1], args.n_alpha)
    print(f"Testing {len(alpha_values)} alpha values with {args.n_folds}-fold CV")
    print(f"Alpha range: {alpha_values.min():.6f} to {alpha_values.max():.6f}")
    print()
    
    # Create concept order and prepare data
    concept_order = create_concept_order(variable_types=args.variable_types)
    
    # Filter concepts that exist in the dataset
    available_concepts = [c for c in concept_order if c in sim_var_name_to_data]
    if len(available_concepts) < len(concept_order):
        missing = set(concept_order) - set(available_concepts)
        print(f"Warning: Missing concepts in dataset: {missing}")
        concept_order = available_concepts
    
    print(f"Using {len(concept_order)} {args.variable_types} concepts")
    print()
    
    # Create train/test split for efficiency analysis
    split_indices = create_data_split(
        total_samples=biscuit_vars.shape[0],
        test_fraction=args.test_fraction,
        random_seed=args.split_seed,
    )
    
    print(f"Data split:")
    print(f"  Test samples: {len(split_indices['test_idx'])}")
    print(f"  Training pool: {len(split_indices['train_pool_perm'])}")
    print()
    
    all_results = []
    
    # Test each variable
    for var_idx, variable_name in enumerate(concept_order):
        print(f"\nProcessing variable {var_idx+1}/{len(concept_order)}: {variable_name}")
        
        variable_type = get_variable_type(variable_name)
        Y = sim_var_name_to_data[variable_name]
        
        # Use training pool for hyperparameter optimization
        X_train_pool = biscuit_vars[split_indices['train_pool_perm']].T
        Y_train_pool = Y[split_indices['train_pool_perm']]
        
        # Ensure Y is 2D for the estimator
        if Y_train_pool.ndim == 1:
            Y_train_pool = Y_train_pool.reshape(1, -1)
        
        # Test each alpha value
        for alpha_idx, alpha in enumerate(alpha_values):
            print(f"  Alpha {alpha_idx+1}/{len(alpha_values)}: {alpha:.6f}")
            
            # Test on full training pool
            metrics = cross_validate_alpha(
                X_train_pool, Y_train_pool, variable_name, variable_type, alpha, args.n_folds
            )
            
            result = {
                'variable': variable_name,
                'variable_type': variable_type,
                'alpha': alpha,
                'n_train': 'full',
                **metrics
            }
            all_results.append(result)
            
            # Test on different training sizes for efficiency analysis (if enabled)
            if not args.skip_efficiency:
                for n_train in args.n_train_values:
                    if n_train < X_train_pool.shape[1]:  # Only if we have enough data
                        metrics_subset = cross_validate_alpha(
                            X_train_pool, Y_train_pool, variable_name, variable_type, 
                            alpha, args.n_folds, n_train=n_train
                        )
                        
                        result_subset = {
                            'variable': variable_name,
                            'variable_type': variable_type,
                            'alpha': alpha,
                            'n_train': n_train,
                            **metrics_subset
                        }
                        all_results.append(result_subset)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(args.output_dir, f"hyperopt_results_{timestamp}.csv")
    results_df.to_csv(results_file, index=False)
    
    print(f"\nResults saved to: {results_file}")
    
    return results_df

def analyze_results(results_df):
    """Analyze and visualize hyperparameter optimization results."""
    
    print("\nAnalyzing results...")
    
    # Find best alpha for each variable
    best_alphas = {}
    
    for variable in results_df['variable'].unique():
        var_data = results_df[
            (results_df['variable'] == variable) & 
            (results_df['n_train'] == 'full')
        ].copy()
        
        if len(var_data) == 0:
            continue
            
        variable_type = var_data['variable_type'].iloc[0]
        
        # Choose primary metric based on variable type
        if variable_type == 'binary':
            primary_metric = 'roc_auc'
            # Fallback to accuracy if ROC AUC not available
            if var_data[primary_metric].isna().all():
                primary_metric = 'accuracy'
        else:
            primary_metric = 'r2'
        
        if not var_data[primary_metric].isna().all():
            best_idx = var_data[primary_metric].idxmax()
            best_alpha = var_data.loc[best_idx, 'alpha']
            best_score = var_data.loc[best_idx, primary_metric]
            
            best_alphas[variable] = {
                'alpha': best_alpha,
                'score': best_score,
                'metric': primary_metric,
                'type': variable_type
            }
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Best alpha distribution by variable type
    binary_alphas = [v['alpha'] for v in best_alphas.values() if v['type'] == 'binary']
    continuous_alphas = [v['alpha'] for v in best_alphas.values() if v['type'] == 'continuous']
    
    axes[0, 0].hist(binary_alphas, bins=10, alpha=0.7, label='Binary', color='blue')
    axes[0, 0].hist(continuous_alphas, bins=10, alpha=0.7, label='Continuous', color='red')
    axes[0, 0].set_xlabel('Best Alpha')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Distribution of Best Alpha Values')
    axes[0, 0].legend()
    axes[0, 0].set_xscale('log')
    
    # Plot 2: Performance vs Alpha for a representative variable
    if len(results_df) > 0:
        repr_var = results_df['variable'].value_counts().index[0]  # Most common variable
        repr_data = results_df[
            (results_df['variable'] == repr_var) & 
            (results_df['n_train'] == 'full')
        ].sort_values('alpha')
        
        if len(repr_data) > 0:
            primary_metric = 'roc_auc' if repr_data['variable_type'].iloc[0] == 'binary' else 'r2'
            if not repr_data[primary_metric].isna().all():
                axes[0, 1].plot(repr_data['alpha'], repr_data[primary_metric], 'o-')
                axes[0, 1].set_xlabel('Alpha')
                axes[0, 1].set_ylabel(primary_metric.upper())
                axes[0, 1].set_title(f'Performance vs Alpha: {repr_var}')
                axes[0, 1].set_xscale('log')
    
    # Plot 3: Training size efficiency
    efficiency_data = results_df[results_df['n_train'] != 'full'].copy()
    if len(efficiency_data) > 0:
        # Group by training size and compute median performance
        efficiency_summary = efficiency_data.groupby(['n_train', 'variable_type']).agg({
            'r2': 'median',
            'roc_auc': 'median'
        }).reset_index()
        
        for var_type in efficiency_summary['variable_type'].unique():
            type_data = efficiency_summary[efficiency_summary['variable_type'] == var_type]
            metric = 'roc_auc' if var_type == 'binary' else 'r2'
            if not type_data[metric].isna().all():
                axes[1, 0].plot(type_data['n_train'], type_data[metric], 'o-', label=f'{var_type} ({metric})')
        
        axes[1, 0].set_xlabel('Training Size')
        axes[1, 0].set_ylabel('Performance')
        axes[1, 0].set_title('Performance vs Training Size')
        axes[1, 0].legend()
    
    # Plot 4: Best scores distribution
    best_scores = [v['score'] for v in best_alphas.values() if not np.isnan(v['score'])]
    if best_scores:
        axes[1, 1].hist(best_scores, bins=15, alpha=0.7, color='green')
        axes[1, 1].set_xlabel('Best Performance Score')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Distribution of Best Performance Scores')
    
    plt.tight_layout()
    
    # Save plots and summary
    if len(best_alphas) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = results_df.attrs.get('output_dir', '.')
        plot_file = os.path.join(output_dir, f"hyperopt_analysis_{timestamp}.pdf")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Analysis plots saved to: {plot_file}")
        
        # Save best alphas summary
        best_alphas_df = pd.DataFrame(best_alphas).T
        summary_file = os.path.join(output_dir, f"best_alphas_{timestamp}.csv")
        best_alphas_df.to_csv(summary_file)
        print(f"Best alphas summary saved to: {summary_file}")
    
    # Print summary
    print("\nSummary of Best Alpha Values:")
    print("=" * 50)
    for variable, info in best_alphas.items():
        print(f"{variable:30s} | α={info['alpha']:.6f} | {info['metric']}={info['score']:.4f}")
    
    return best_alphas_df

def main():
    """Main entry point"""
    args = parse_args()
    
    print("Hyperparameter Optimization for Permutation Estimator")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Variable types: {args.variable_types}")
    print(f"Alpha range: 10^{args.alpha_range[0]} to 10^{args.alpha_range[1]}")
    print(f"Number of alphas: {args.n_alpha}")
    print(f"Cross-validation folds: {args.n_folds}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Setup output directory with variable type suffix
    output_dir = f"{args.output_dir}_{args.variable_types}"
    setup_output_dir(output_dir)
    args.output_dir = output_dir
    
    # Run optimization
    results_df = run_hyperparameter_optimization(args)
    
    if len(results_df) > 0:
        # Store output directory for plotting
        results_df.attrs['output_dir'] = output_dir
        
        # Analyze results
        best_alphas_df = analyze_results(results_df)
        
        print("\nHyperparameter optimization completed successfully!")
        print(f"Results available in: {output_dir}")
        
        if args.save_plots:
            plt.show()
    else:
        print("No results generated. Check data availability and error messages.")

if __name__ == "__main__":
    main()
