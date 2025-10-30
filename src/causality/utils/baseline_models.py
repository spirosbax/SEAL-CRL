"""
Baseline model evaluation utilities for causal grouping experiments.

This module provides functions for evaluating baseline models (LinearRegression, 
LogisticRegression, MLPClassifier) on aggregated causal predicates.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.base import clone
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

from .utils import setup_output_dir
from .math_utils import r2_safe


## use shared r2_safe from math_utils


def model_scores(pipe: Any, X: np.ndarray) -> np.ndarray:
    """
    Return continuous scores for X from a fitted pipeline.
    
    Prefers predict_proba[:,1] (probabilities) for classification,
    then decision_function (margins), then predict (e.g., linear regression).
    
    Args:
        pipe: Fitted sklearn pipeline
        X: Input features
        
    Returns:
        Continuous scores/predictions
    """
    if hasattr(pipe, "predict_proba"):
        try:
            return pipe.predict_proba(X)[:, 1]
        except Exception:
            pass
    if hasattr(pipe, "decision_function"):
        try:
            return pipe.decision_function(X)
        except Exception:
            pass
    return pipe.predict(X)


def stratified_subset_from_pool(
    pool_idx: np.ndarray, 
    y: np.ndarray, 
    n: int, 
    random_state: int = 0,
    min_per_class: int = 1, 
    require_both_classes: bool = True
) -> np.ndarray:
    """
    Create a stratified subset of size n from pool_idx.
    
    This function ensures balanced class representation while handling edge cases
    like small sample sizes and missing classes.
    
    Args:
        pool_idx: Indices of the available pool
        y: Target labels for all samples
        n: Desired subset size
        random_state: Random seed for reproducibility
        min_per_class: Minimum samples per class (if feasible)
        require_both_classes: Whether to ensure both classes are represented
        
    Returns:
        Array of selected indices
    """
    rng = np.random.RandomState(random_state + int(n * 9973))
    y_pool = y[pool_idx]
    classes, counts = np.unique(y_pool, return_counts=True)

    # If only one class exists in the pool, we can't do better
    if len(classes) == 1:
        return pool_idx[:min(n, len(pool_idx))]

    # Desired proportional allocation
    p = counts / counts.sum()
    desired = np.round(p * n).astype(int)

    # Enforce min_per_class if required
    if require_both_classes:
        desired = np.maximum(desired, min_per_class)

    # Cap by availability (no replacement baseline)
    desired = np.minimum(desired, counts)

    # Adjust to sum <= n
    total = desired.sum()
    if total > n:
        # Reduce from the class(es) with the largest residual first
        residual = desired - np.floor(p * n).astype(int)
        order = np.argsort(-residual)
        for j in order:
            take = min(desired[j], total - n)
            desired[j] -= take
            total -= take
            if total == n:
                break

    # If total < n, try to add more from classes that still have capacity
    total = desired.sum()
    if total < n:
        need = n - total
        capacity = counts - desired
        order = np.argsort(-capacity)
        for j in order:
            add = min(capacity[j], need)
            desired[j] += add
            need -= add
            if need == 0:
                break
        total = desired.sum()

    # If still short, fill with replacement (rare, only for tiny samples)
    take_with_replacement = max(0, n - total)

    # Materialize indices without replacement per class
    chosen = []
    for cls, want in zip(classes, desired):
        cls_pool = pool_idx[y_pool == cls]
        if want > 0:
            chosen.append(rng.choice(cls_pool, size=want, replace=False))
    chosen = np.concatenate(chosen) if chosen else np.array([], dtype=int)

    if take_with_replacement > 0:
        # Pull remaining with replacement
        extra = rng.choice(pool_idx, size=take_with_replacement, replace=True)
        chosen = np.concatenate([chosen, extra])

    rng.shuffle(chosen)
    return chosen[:n]


def balance_classes_subsample(
    y_all: np.ndarray,
    target_ratio: float = 0.5,
    random_state: int = 0,
    verbose: bool = True
) -> np.ndarray:
    """
    Subsample data to achieve balanced classes close to target ratio.
    
    This function balances classes by undersampling the majority class
    to match the minority class size, achieving close to 50-50 balance.
    
    Args:
        y_all: Target labels (0/1)
        target_ratio: Target ratio for minority class (default: 0.5 for 50-50)
        random_state: Random seed for reproducibility
        verbose: Whether to print balancing information
        
    Returns:
        Array of indices for balanced subset
    """
    rng = np.random.RandomState(random_state)
    
    # Count class frequencies
    classes, counts = np.unique(y_all, return_counts=True)
    class_counts = dict(zip(classes, counts))
    
    if verbose:
        print(f"Original class distribution:")
        for cls, count in class_counts.items():
            ratio = count / len(y_all)
            print(f"  Class {cls}: {count} samples ({ratio:.1%})")
    
    # Identify minority and majority classes
    min_class = classes[np.argmin(counts)]
    maj_class = classes[np.argmax(counts)]
    min_count = np.min(counts)
    maj_count = np.max(counts)
    
    # Calculate target counts for balanced dataset
    # We'll undersample majority to create balance
    target_min_count = min_count  # Keep all minority samples
    target_maj_count = int(target_min_count / target_ratio - target_min_count)
    target_maj_count = min(target_maj_count, maj_count)  # Can't exceed available
    
    # Get indices for each class
    min_indices = np.where(y_all == min_class)[0]
    maj_indices = np.where(y_all == maj_class)[0]
    
    # Sample from majority class
    sampled_maj_indices = rng.choice(maj_indices, size=target_maj_count, replace=False)
    
    # Combine indices
    balanced_indices = np.concatenate([min_indices, sampled_maj_indices])
    rng.shuffle(balanced_indices)
    
    # Report balanced distribution
    if verbose:
        balanced_y = y_all[balanced_indices]
        balanced_classes, balanced_counts = np.unique(balanced_y, return_counts=True)
        balanced_class_counts = dict(zip(balanced_classes, balanced_counts))
        
        print(f"\nBalanced class distribution:")
        for cls, count in balanced_class_counts.items():
            ratio = count / len(balanced_indices)
            print(f"  Class {cls}: {count} samples ({ratio:.1%})")
        print(f"Total samples: {len(balanced_indices)} (reduced from {len(y_all)})")
    
    return balanced_indices


def create_train_test_split(
    y_all: np.ndarray, 
    test_size: float = 0.2, 
    random_state: int = 0,
    balance_classes: bool = False,
    target_ratio: float = 0.5,
    verbose: bool = True
) -> Dict[str, np.ndarray]:
    """
    Create reproducible stratified train/test split with optional class balancing.
    
    Args:
        y_all: Target labels for stratification
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        balance_classes: Whether to balance classes before splitting
        target_ratio: Target ratio for minority class (when balancing)
        verbose: Whether to print balancing information
        
    Returns:
        Dictionary with 'test_idx' and 'train_pool_perm' keys
    """
    if balance_classes:
        if verbose:
            print("Balancing classes before train/test split...")
        balanced_indices = balance_classes_subsample(
            y_all, target_ratio=target_ratio, random_state=random_state, verbose=verbose
        )
        # Apply balancing to get balanced subset
        y_balanced = y_all[balanced_indices]
        
        # Create stratified split on balanced data
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        (train_pool_rel, test_rel), = sss.split(np.zeros_like(y_balanced), y_balanced)
        
        # Map relative indices back to original indices
        train_pool_idx = balanced_indices[train_pool_rel]
        test_idx = balanced_indices[test_rel]
    else:
        # Standard stratified split without balancing
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        (train_pool_idx, test_idx), = sss.split(np.zeros_like(y_all), y_all)
    
    # Create a permutation of the training pool for reproducible subsampling
    rng = np.random.RandomState(random_state)
    train_pool_perm = rng.permutation(train_pool_idx)
    
    return {'test_idx': test_idx, 'train_pool_perm': train_pool_perm}


def create_baseline_models(mlp_hidden_sizes: List[int] = None) -> List[Tuple[str, Any]]:
    """
    Create list of baseline models for evaluation.
    
    Args:
        mlp_hidden_sizes: List of hidden layer sizes for MLP models
        
    Returns:
        List of (name, pipeline) tuples
    """
    if mlp_hidden_sizes is None:
        mlp_hidden_sizes = [2, 4, 8]
    
    models = [
        ("LinReg", make_pipeline(
            StandardScaler(),
            LinearRegression()
        )),
        ("LogReg", make_pipeline(
            StandardScaler(),
            LogisticRegression()
        ))
    ]
    
    # Add MLP models with different hidden layer sizes
    for n_units in mlp_hidden_sizes:
        models.append((
            f"MLP-1x{n_units}",
            make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(n_units,),
                    activation='relu',
                    max_iter=300,
                    learning_rate_init=1e-3,
                    random_state=0
                )
            )
        ))
    
    return models


def evaluate_baseline_models(
    X_all: np.ndarray,
    y_all: np.ndarray,
    split_dict: Dict[str, np.ndarray],
    target_n_trains: List[int],
    models: List[Tuple[str, Any]],
    seeds: List[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Evaluate baseline models across different training set sizes and seeds.
    
    Args:
        X_all: Feature matrix (N, D)
        y_all: Target labels (N,)
        split_dict: Dictionary with 'test_idx' and 'train_pool_perm'
        target_n_trains: List of training set sizes to evaluate
        models: List of (name, pipeline) tuples
        seeds: List of random seeds for evaluation
        verbose: Whether to print progress
        
    Returns:
        DataFrame with evaluation results
    """
    if seeds is None:
        seeds = [0, 1, 2]
    
    train_pool = split_dict['train_pool_perm']
    test_idx = split_dict['test_idx']
    
    # Filter training sizes that are feasible
    total_train_pool = len(train_pool)
    target_n_trains = [n for n in target_n_trains if n <= total_train_pool]
    
    if verbose:
        cls, cnt = np.unique(y_all[train_pool], return_counts=True)
        print("Train-pool class counts (y=0, y=1):", dict(zip(cls.tolist(), cnt.tolist())))
    
    rows = []
    
    for seed in seeds:
        if verbose:
            print(f"\n*** Seed: {seed} ***")
        
        for name, pipe in models:
            # Clone the pipeline to avoid state issues across seeds
            pipe_seed = clone(pipe)
            
            # Set random state for MLP if applicable
            if hasattr(pipe_seed.named_steps.get('mlpclassifier', None), 'random_state'):
                pipe_seed.named_steps['mlpclassifier'].random_state = seed
            
            for n_train in target_n_trains:
                # Skip if we can't get both classes with this sample size
                if len(np.unique(y_all[train_pool])) == 2 and n_train < 2:
                    continue
                
                train_subset_idx = stratified_subset_from_pool(
                    train_pool, y_all, n_train,
                    random_state=seed, min_per_class=1, require_both_classes=True
                )
                
                # Skip if subset collapsed to one class
                if len(np.unique(y_all[train_subset_idx])) < 2:
                    continue
                
                X_tr, X_te = X_all[train_subset_idx], X_all[test_idx]
                y_tr, y_te = y_all[train_subset_idx], y_all[test_idx]
                
                try:
                    pipe_seed.fit(X_tr, y_tr)
                    scores_te = model_scores(pipe_seed, X_te)
                    
                    # Compute metrics
                    auc = roc_auc_score(y_te, scores_te) if len(np.unique(y_te)) == 2 else np.nan
                    r2 = r2_safe(y_te, scores_te)
                    
                    rows.append({
                        'model': name, 
                        'seed': seed, 
                        'n_train': n_train, 
                        'auc': auc, 
                        'r2': r2
                    })
                
                except Exception as e:
                    if verbose:
                        print(f"Warning: Failed to evaluate {name} with n_train={n_train}, seed={seed}: {e}")
                    rows.append({
                        'model': name, 
                        'seed': seed, 
                        'n_train': n_train, 
                        'auc': np.nan, 
                        'r2': np.nan
                    })
    
    return pd.DataFrame(rows)


def summarize_baseline_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize baseline model results across seeds.
    
    Args:
        df: DataFrame with individual results
        
    Returns:
        DataFrame with mean, min, max statistics
    """
    # Determine groupby columns based on whether predicate column exists
    groupby_cols = ['model', 'n_train']
    if 'predicate' in df.columns:
        groupby_cols = ['model', 'n_train', 'predicate']
    
    summary = (df
               .groupby(groupby_cols)
               .agg({'auc': ['mean', 'min', 'max'], 'r2': ['mean', 'min', 'max']})
               .reset_index())
    
    # Flatten column names based on groupby columns
    if 'predicate' in df.columns:
        summary.columns = ['model', 'n_train', 'predicate', 'auc_mean', 'auc_min', 'auc_max', 'r2_mean', 'r2_min', 'r2_max']
    else:
        summary.columns = ['model', 'n_train', 'auc_mean', 'auc_min', 'auc_max', 'r2_mean', 'r2_min', 'r2_max']
    
    return summary


def inspect_logistic_weights(
    X_all: np.ndarray, 
    y_all: np.ndarray, 
    feature_names: List[str] = None
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Inspect logistic regression weights on full dataset.
    
    Args:
        X_all: Full feature matrix
        y_all: Full target labels
        feature_names: Optional feature names for interpretation
        
    Returns:
        Dictionary with weights and intercept in original space
    """
    logreg_full = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty=None, solver='lbfgs', max_iter=3000, tol=1e-3)
    )
    logreg_full.fit(X_all, y_all)
    
    scaler = logreg_full.named_steps['standardscaler']
    clf = logreg_full.named_steps['logisticregression']
    
    # Convert coefficients back to original feature space
    w_scaled = clf.coef_.ravel()
    b_scaled = clf.intercept_.item()
    sigma = scaler.scale_
    mu = scaler.mean_
    
    w_unscaled = w_scaled / sigma
    b_unscaled = b_scaled - np.dot(w_scaled, mu / sigma)
    
    result = {
        'weights_original': w_unscaled,
        'intercept_original': b_unscaled,
        'weights_scaled': w_scaled,
        'intercept_scaled': b_scaled
    }
    
    if feature_names is not None:
        result['feature_names'] = feature_names
        print("LogReg weights (original space):")
        for name, weight in zip(feature_names, w_unscaled):
            print(f"  {name}: {weight:.4f}")
        print(f"LogReg intercept (original space): {b_unscaled:.4f}")
    else:
        print("LogReg weights (original space):", w_unscaled)
        print("LogReg intercept (original space):", b_unscaled)
    
    return result


def create_aggregate_predicate(
    sim_var_name_to_data: Dict[str, np.ndarray], 
    predicate_type: str,
    variables: List[str] = None
) -> np.ndarray:
    """
    Create aggregate predicates for different logical combinations.
    
    Args:
        sim_var_name_to_data: Dictionary mapping variable names to data
        predicate_type: Type of predicate ('all_stoves_on', 'at_least_two_stoves_on', 'egg_intact')
        variables: List of variable names (if None, uses default for predicate type)
        
    Returns:
        Binary array indicating when the predicate is satisfied
    """
    if predicate_type == "all_stoves_on":
        if variables is None:
            variables = list(get_stove_configuration().keys())
        stove_mat = np.stack([
            sim_var_name_to_data[v].astype(int) 
            for v in variables
        ], axis=1)
        # AND operation: all stoves must be ON
        return (stove_mat.min(axis=1) > 0).astype(int)
    
    elif predicate_type == "at_least_two_stoves_on":
        if variables is None:
            variables = list(get_stove_configuration().keys())
        stove_mat = np.stack([
            sim_var_name_to_data[v].astype(int) 
            for v in variables
        ], axis=1)
        # Count how many stoves are on, return 1 if >= 2
        return (stove_mat.sum(axis=1) >= 2).astype(int)
    
    elif predicate_type == "at_least_three_stoves_on":
        if variables is None:
            variables = list(get_stove_configuration().keys())
        stove_mat = np.stack([
            sim_var_name_to_data[v].astype(int) 
            for v in variables
        ], axis=1)
        # Count how many stoves are on, return 1 if >= 3
        return (stove_mat.sum(axis=1) >= 3).astype(int)
    
    elif predicate_type == "egg_intact":
        if variables is None:
            variables = ['Egg_afaaaca3_broken', 'Egg_afaaaca3_cooked']
        # Check that both variables exist
        missing_vars = [v for v in variables if v not in sim_var_name_to_data]
        if missing_vars:
            raise ValueError(f"Missing variables for egg_intact predicate: {missing_vars}")
        
        # Egg is intact if it's NOT (broken OR cooked)
        egg_broken = sim_var_name_to_data[variables[0]].astype(int)
        egg_cooked = sim_var_name_to_data[variables[1]].astype(int)
        # NOT (broken OR cooked) = NOT broken AND NOT cooked
        return ((1 - egg_broken) * (1 - egg_cooked)).astype(int)
    
    else:
        raise ValueError(f"Unknown predicate type: {predicate_type}. "
                        f"Supported types: 'all_stoves_on', 'at_least_two_stoves_on', 'at_least_three_stoves_on', 'egg_intact'")


def get_stove_configuration() -> Dict[str, int]:
    """
    Get the standard stove knob configuration.
    
    Returns:
        Dictionary mapping stove variable names to latent dimensions
    """
    return {
        'StoveKnob_38c1dbc2_on': 18,  # Front-Right
        'StoveKnob_690d0d5d_on': 11,  # Front-Left
        'StoveKnob_c8955f66_on': 12,  # Back-Left
        'StoveKnob_cf670576_on': 21,  # Back-Right
    }


def get_egg_configuration() -> Dict[str, int]:
    """
    Get the standard egg configuration.
    
    Returns:
        Dictionary mapping egg variable names to latent dimensions
    """
    return {
        'Egg_afaaaca3_broken': 3,
        'Egg_afaaaca3_cooked': 2,
    }


def get_predicate_configuration(predicate_type: str) -> Dict[str, Union[List[str], List[int]]]:
    """
    Get configuration for different predicate types.
    
    Args:
        predicate_type: Type of predicate
        
    Returns:
        Dictionary with 'variables' and 'latent_dims' keys
    """
    if predicate_type == "all_stoves_on":
        stove_config = get_stove_configuration()
        return {
            'variables': list(stove_config.keys()),
            'latent_dims': list(stove_config.values())
        }
    elif predicate_type == "at_least_two_stoves_on":
        stove_config = get_stove_configuration()
        return {
            'variables': list(stove_config.keys()),
            'latent_dims': list(stove_config.values())
        }
    elif predicate_type == "at_least_three_stoves_on":
        stove_config = get_stove_configuration()
        return {
            'variables': list(stove_config.keys()),
            'latent_dims': list(stove_config.values())
        }
    elif predicate_type == "egg_intact":
        # For egg predicate, we don't have specific latent dims since it's based on concepts
        return {
            'variables': ['Egg_afaaaca3_broken', 'Egg_afaaaca3_cooked'],
            'latent_dims': []  # No specific latent dimensions for this predicate
        }
    else:
        raise ValueError(f"Unknown predicate type: {predicate_type}")


def get_supported_predicates() -> List[str]:
    """
    Get list of supported predicate types.
    
    Returns:
        List of supported predicate type strings
    """
    return ["all_stoves_on", "at_least_two_stoves_on", "at_least_three_stoves_on", "egg_intact"]


def get_stove_predicates() -> List[str]:
    """
    Get list of stove-related predicates (which overlap in latent space).
    
    Returns:
        List of stove predicate type strings
    """
    return ["all_stoves_on", "at_least_two_stoves_on", "at_least_three_stoves_on"]


def validate_predicate_combination(predicates: List[str]) -> None:
    """
    Validate that predicate combination doesn't have overlapping latent dimensions.
    
    Args:
        predicates: List of predicate names to validate
        
    Raises:
        ValueError: If multiple stove predicates are specified
    """
    stove_predicates = get_stove_predicates()
    stove_count = sum(1 for p in predicates if p in stove_predicates)
    
    if stove_count > 1:
        stove_in_list = [p for p in predicates if p in stove_predicates]
        raise ValueError(
            f"Cannot use multiple stove predicates simultaneously as they have overlapping "
            f"latent dimensions. Found: {stove_in_list}. Use only one stove predicate at a time."
        )
