"""
Model evaluation utilities for permutation estimator experiments.

This module provides functions for evaluating different alignment methods:
- Baseline: Uses external standardization (StandardScaler)
- Linear/Kernel: Use raw data and rely on internal fit_transform in estimators
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from .math_utils import sigmoid, r2_safe, add_jitter_if_constant
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add external library to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(project_root, 'external', 'sample-efficient-learning-of-concepts'))

from permutation_estimator.estimator import (
    FeaturePermutationEstimator,
    KernelizedPermutationEstimator
)


def get_all_variable_info() -> Dict[str, str]:
    """
    Get comprehensive variable information for iTHOR dataset.
    
    Returns:
        Dict mapping variable names to their types
    """
    from collections import OrderedDict
    return OrderedDict([
        ('Cabinet_47fc321b_open', 'categ_2'),
        ('Egg_afaaaca3_broken', 'categ_2'),
        ('Egg_afaaaca3_center_x', 'continuous_1.0'),
        ('Egg_afaaaca3_center_y', 'continuous_1.0'),
        ('Egg_afaaaca3_center_z', 'continuous_1.0'),
        ('Egg_afaaaca3_cooked', 'categ_2'),
        ('Egg_afaaaca3_pickedup', 'categ_2'),
        ('Microwave_d8b935e4_on', 'categ_2'),
        ('Microwave_d8b935e4_open', 'categ_2'),
        ('Plate_49b95a7a_center_x', 'continuous_1.0'),
        ('Plate_49b95a7a_center_y', 'continuous_1.0'),
        ('Plate_49b95a7a_center_z', 'continuous_1.0'),
        ('Plate_49b95a7a_pickedup', 'categ_2'),
        ('StoveKnob_38c1dbc2_on', 'categ_2'),
        ('StoveKnob_690d0d5d_on', 'categ_2'),
        ('StoveKnob_c8955f66_on', 'categ_2'),
        ('StoveKnob_cf670576_on', 'categ_2'),
        ('Toaster_194647f5_on', 'categ_2')
    ])


def create_concept_order(variable_types: str = "binary") -> np.ndarray:
    """
    Create concept ordering for iTHOR experiments based on variable types.
    
    Args:
        variable_types: Type of variables to include:
            - "binary": Only binary/categorical variables (categ_2)
            - "continuous": Only continuous variables 
            - "all": All variables (binary + continuous)
    
    Returns:
        np.ndarray: Array of concept names in standard order
    """
    all_vars = get_all_variable_info()
    
    if variable_types == "binary":
        # Original binary variables in the specified order
        binary_order = [
            'Plate_49b95a7a_pickedup', 'Egg_afaaaca3_broken', 'Egg_afaaaca3_cooked',
            'Egg_afaaaca3_pickedup', 'StoveKnob_690d0d5d_on', 'StoveKnob_c8955f66_on',
            'StoveKnob_38c1dbc2_on', 'StoveKnob_cf670576_on', 'Microwave_d8b935e4_on',
            'Microwave_d8b935e4_open', 'Toaster_194647f5_on', 'Cabinet_47fc321b_open'
        ]
        return np.array(binary_order)
    
    elif variable_types == "continuous":
        # Continuous variables grouped by object
        continuous_order = [
            'Egg_afaaaca3_center_x', 'Egg_afaaaca3_center_y', 'Egg_afaaaca3_center_z',
            'Plate_49b95a7a_center_x', 'Plate_49b95a7a_center_y', 'Plate_49b95a7a_center_z'
        ]
        return np.array(continuous_order)
    
    elif variable_types == "all":
        # All variables: first binary, then continuous, in logical groups
        all_order = [
            # Binary variables (objects and states)
            'Cabinet_47fc321b_open',
            'Egg_afaaaca3_broken', 'Egg_afaaaca3_cooked', 'Egg_afaaaca3_pickedup',
            'Microwave_d8b935e4_on', 'Microwave_d8b935e4_open',
            'Plate_49b95a7a_pickedup',
            'StoveKnob_690d0d5d_on', 'StoveKnob_c8955f66_on', 
            'StoveKnob_38c1dbc2_on', 'StoveKnob_cf670576_on',
            'Toaster_194647f5_on',
            # Continuous variables (positions)
            'Egg_afaaaca3_center_x', 'Egg_afaaaca3_center_y', 'Egg_afaaaca3_center_z',
            'Plate_49b95a7a_center_x', 'Plate_49b95a7a_center_y', 'Plate_49b95a7a_center_z'
        ]
        return np.array(all_order)
    
    else:
        raise ValueError(f"Unknown variable_types: {variable_types}. Choose from 'binary', 'continuous', 'all'")


def create_data_split(total_samples: int, test_fraction: float = 0.2, 
                     random_seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Create train/test split indices for experiments.
    
    Args:
        total_samples: Total number of samples in dataset
        test_fraction: Fraction of data to use for testing
        random_seed: Random seed for reproducible splits
        
    Returns:
        Dict containing 'test_idx' and 'train_pool_perm' arrays
    """
    rng = np.random.default_rng(random_seed)
    all_indices = np.arange(total_samples)
    rng.shuffle(all_indices)
    
    n_test = int(total_samples * test_fraction)
    test_idx = all_indices[:n_test]
    train_pool_perm = all_indices[n_test:]
    
    return {
        'test_idx': test_idx,
        'train_pool_perm': train_pool_perm
    }


def evaluate_baseline_model_std(X_train_std: np.ndarray, Y_train_std: np.ndarray,
                               X_test_std: np.ndarray, Y_test_std: np.ndarray,
                               concept_names: List[str]) -> List[Dict]:
    """
    Evaluate baseline model using Spearman correlation + Hungarian assignment + Linear Regression.
    Uses standardized X and Y for fair comparison.
    
    Args:
        X_train_std: Standardized training features (d_vars, N_train)
        Y_train_std: Standardized training targets (n_concepts, N_train)
        X_test_std: Standardized test features (d_vars, N_test)
        Y_test_std: Standardized test targets (n_concepts, N_test)
        concept_names: List of concept names
        
    Returns:
        List of result dictionaries with concept, r2 values
    """
    num_concepts, num_dims = Y_train_std.shape[0], X_train_std.shape[0]
    corr_mat = np.zeros((num_concepts, num_dims))
    
    # Spearman uses ranks (Y scaling irrelevant); X scaled for consistency
    for i in range(num_concepts):
        y_tr = Y_train_std[i]
        if np.unique(y_tr).size < 2:
            continue
        for j in range(num_dims):
            rho, _ = spearmanr(y_tr, X_train_std[j])
            corr_mat[i, j] = 0.0 if np.isnan(rho) else rho

    # Hungarian assignment on absolute correlation
    _, col_ind = linear_sum_assignment(-np.abs(corr_mat))
    
    # Evaluate each concept
    results = []
    for i, concept in enumerate(concept_names):
        j = col_ind[i]
        y_tr = Y_train_std[i]
        y_te = Y_test_std[i]
        
        # Check if we can fit a meaningful model
        if np.unique(y_tr).size < 2 or np.std(y_te) == 0:
            r2v = np.nan
        else:
            lm = LinearRegression().fit(X_train_std[j][None, :].T, y_tr)
            y_hat = lm.predict(X_test_std[j][None, :].T)
            r2v = r2_safe(y_te, y_hat)
        
        results.append({
            "concept": concept,
            "r2": r2v,
            "assigned_dim": j
        })
    
    return results


def evaluate_linear_model_std(X_train_std: np.ndarray, Y_train_std: np.ndarray,
                             X_test_std: np.ndarray, Y_test_std: np.ndarray,
                             concept_names: List[str], alpha: float,
                             rng: np.random.Generator) -> List[Dict]:
    """
    Evaluate linear permutation estimator model with standardized inputs.
    
    Args:
        X_train_std: Standardized training features (d_vars, N_train)
        Y_train_std: Standardized training targets (n_concepts, N_train)
        X_test_std: Standardized test features (d_vars, N_test)
        Y_test_std: Standardized test targets (n_concepts, N_test)
        concept_names: List of concept names
        alpha: Regularization parameter
        rng: Random number generator
        
    Returns:
        List of result dictionaries with concept, r2 values
    """
    est = FeaturePermutationEstimator(
        regularizer='lasso',
        optim_kwargs={'alpha': alpha},
        feature_transform=None,
        d_variables=X_train_std.shape[0],
        n_features=1
    )
    
    est.fit(X_train_std, add_jitter_if_constant(Y_train_std, rng))
    Y_hat_std = est.predict_match(X_test_std)
    
    return [{"concept": c, "r2": r2_safe(Y_test_std[i], Y_hat_std[i])}
            for i, c in enumerate(concept_names)]


def evaluate_linear_model_raw(X_train_raw: np.ndarray, Y_train_raw: np.ndarray,
                             X_test_raw: np.ndarray, Y_test_raw: np.ndarray,
                             concept_names: List[str], alpha: float,
                             rng: np.random.Generator) -> List[Dict]:
    """
    Evaluate linear permutation estimator model with raw (unstandardized) inputs.
    Uses internal fit_transform in the estimator for standardization.
    
    Args:
        X_train_raw: Raw training features (d_vars, N_train)
        Y_train_raw: Raw training targets (n_concepts, N_train)
        X_test_raw: Raw test features (d_vars, N_test)
        Y_test_raw: Raw test targets (n_concepts, N_test)
        concept_names: List of concept names
        alpha: Regularization parameter
        rng: Random number generator
        
    Returns:
        List of result dictionaries with concept, r2 values
    """
    est = FeaturePermutationEstimator(
        regularizer='lasso',
        optim_kwargs={'alpha': alpha},
        feature_transform=None,
        d_variables=X_train_raw.shape[0],
        n_features=1
    )
    
    est.fit(X_train_raw, add_jitter_if_constant(Y_train_raw, rng))
    Y_hat_raw = est.predict_match(X_test_raw)
    
    return [{"concept": c, "r2": r2_safe(Y_test_raw[i], Y_hat_raw[i])}
            for i, c in enumerate(concept_names)]


def evaluate_logistic_model_raw(X_train_raw: np.ndarray, Y_train_raw: np.ndarray,
                                X_test_raw: np.ndarray, Y_test_raw: np.ndarray,
                                concept_names: List[str],
                                rng: np.random.Generator) -> List[Dict]:
    """
    Evaluate logistic permutation estimator model with raw (unstandardized) inputs.
    Uses internal fit_transform in the estimator for standardization.

    This assumes binary targets for all concepts; if any concept has a single
    class in training data, returns NaN for that concept to avoid invalid fits.

    Args:
        X_train_raw: Raw training features (d_vars, N_train)
        Y_train_raw: Raw training targets (n_concepts, N_train) expected binary
        X_test_raw: Raw test features (d_vars, N_test)
        Y_test_raw: Raw test targets (n_concepts, N_test)
        concept_names: List of concept names
        rng: Random number generator (for jitter decisions if needed)

    Returns:
        List of result dictionaries with concept, r2 values
    """
    # Ensure integer/binary labels for logistic
    Y_train_bin = Y_train_raw.astype(int, copy=True)

    # Check class presence per concept
    ok_classes = [np.unique(Y_train_bin[i]).size > 1 for i in range(len(concept_names))]

    est = FeaturePermutationEstimator(
        regularizer='logistic',
        optim_kwargs={
            'penalty': 'l1',
            'solver': 'liblinear',
            'C': 1.0,
            'class_weight': 'balanced',
            'max_iter': 20000,
        },
        feature_transform=None,
        d_variables=X_train_raw.shape[0],
        n_features=1
    )

    if all(ok_classes):
        est.fit(X_train_raw, Y_train_bin)
        Y_hat_raw = est.predict_match(X_test_raw)
        return [{"concept": c, "r2": r2_safe(Y_test_raw[i], sigmoid(Y_hat_raw[i]))}
                for i, c in enumerate(concept_names)]
    else:
        # If some concept cannot be trained (single class), still align shape
        # and return NaNs for those concepts
        try:
            est.fit(X_train_raw, Y_train_bin)
            Y_hat_raw = est.predict_match(X_test_raw)
        except Exception:
            # Fallback: predict zeros to keep dimensions; r2_safe will likely be NaN
            Y_hat_raw = np.zeros_like(Y_test_raw)
        results = []
        for i, c in enumerate(concept_names):
            if ok_classes[i]:
                results.append({"concept": c, "r2": r2_safe(Y_test_raw[i], Y_hat_raw[i])})
            else:
                results.append({"concept": c, "r2": np.nan})
        return results

def evaluate_kernel_model_std(X_train_std: np.ndarray, Y_train_std: np.ndarray,
                             X_test_std: np.ndarray, Y_test_std: np.ndarray,
                             concept_names: List[str], alpha: float,
                             kernel: str, parameter: float,
                             rng: np.random.Generator) -> List[Dict]:
    """
    Evaluate kernel permutation estimator model with standardized inputs.
    
    Args:
        X_train_std: Standardized training features (d_vars, N_train)
        Y_train_std: Standardized training targets (n_concepts, N_train)
        X_test_std: Standardized test features (d_vars, N_test)
        Y_test_std: Standardized test targets (n_concepts, N_test)
        concept_names: List of concept names
        alpha: Regularization parameter
        kernel: Kernel type ('rbf', 'laplacian')
        parameter: Kernel parameter (gamma)
        rng: Random number generator
        
    Returns:
        List of result dictionaries with concept, r2 values
    """
    # Cap Nystroem components to avoid small-N issues
    n_feat = min(64, X_train_std.shape[1])
    
    est = KernelizedPermutationEstimator(
        regularizer='lasso',
        optim_kwargs={'alpha': alpha},
        kernel=kernel,
        parameter=parameter,
        d_variables=X_train_std.shape[0],
        n_features=n_feat,
        groups=None,
        two_stage=None
    )
    
    est.fit(X_train_std, add_jitter_if_constant(Y_train_std, rng))
    Y_hat_std = est.predict_match(X_test_std)
    
    return [{"concept": c, "r2": r2_safe(Y_test_std[i], Y_hat_std[i])}
            for i, c in enumerate(concept_names)]


def evaluate_kernel_model_raw(X_train_raw: np.ndarray, Y_train_raw: np.ndarray,
                             X_test_raw: np.ndarray, Y_test_raw: np.ndarray,
                             concept_names: List[str], alpha: float,
                             kernel: str, parameter: float,
                             rng: np.random.Generator) -> List[Dict]:
    """
    Evaluate kernel permutation estimator model with raw (unstandardized) inputs.
    Uses internal fit_transform in the estimator for standardization.
    
    Args:
        X_train_raw: Raw training features (d_vars, N_train)
        Y_train_raw: Raw training targets (n_concepts, N_train)
        X_test_raw: Raw test features (d_vars, N_test)
        Y_test_raw: Raw test targets (n_concepts, N_test)
        concept_names: List of concept names
        alpha: Regularization parameter
        kernel: Kernel type ('rbf', 'laplacian')
        parameter: Kernel parameter (gamma)
        rng: Random number generator
        
    Returns:
        List of result dictionaries with concept, r2 values
    """
    # Cap Nystroem components to avoid small-N issues
    n_feat = min(64, X_train_raw.shape[1])
    
    est = KernelizedPermutationEstimator(
        regularizer='lasso',
        optim_kwargs={'alpha': alpha},
        kernel=kernel,
        parameter=parameter,
        d_variables=X_train_raw.shape[0],
        n_features=n_feat,
        groups=None,
        two_stage=None
    )
    
    est.fit(X_train_raw, add_jitter_if_constant(Y_train_raw, rng))
    Y_hat_raw = est.predict_match(X_test_raw)
    
    return [{"concept": c, "r2": r2_safe(Y_test_raw[i], Y_hat_raw[i])}
            for i, c in enumerate(concept_names)]


def run_model_comparison(X_all: np.ndarray, Y_all: np.ndarray,
                        concept_names: List[str], split_indices: Dict,
                        n_train_values: List[int], alpha: float,
                        seeds: List[int], models_config: Dict) -> pd.DataFrame:
    """
    Run comprehensive model comparison across different training sizes and seeds.
    Uses standardized inputs for baseline model, raw inputs for linear/kernel models.
    
    Args:
        X_all: All feature data (N_total, d_vars)
        Y_all: All target data (n_concepts, N_total)
        concept_names: List of concept names
        split_indices: Dict with 'test_idx' and 'train_pool_perm'
        n_train_values: List of training set sizes to evaluate
        alpha: Regularization parameter
        seeds: List of random seeds for multiple runs
        models_config: Dict mapping model names to (kind, kernel, parameter) tuples
        
    Returns:
        DataFrame with columns: model, seed, concept, n_train, r2
    """
    records = []
    test_idx = split_indices["test_idx"]
    train_pool_perm0 = split_indices["train_pool_perm"]
    
    # Filter training sizes based on available data after balancing
    max_feasible_n = len(train_pool_perm0)
    original_n_trains = n_train_values.copy()
    feasible_n_trains = [n for n in n_train_values if n <= max_feasible_n]
    
    if len(feasible_n_trains) < len(original_n_trains):
        filtered_out = [n for n in original_n_trains if n > max_feasible_n]
        print(f"  Filtered out training sizes (exceed available data): {filtered_out}")
        n_train_values = feasible_n_trains
        print(f"  Using feasible training sizes: {feasible_n_trains}")

    # Fixed test block (raw data)
    X_test = X_all[test_idx].T    # (d, N_test)
    Y_test = Y_all[:, test_idx]   # (m, N_test)
    
    for seed in seeds:
        print(f"\n*** Seed: {seed} ***")
        rng = np.random.default_rng(seed)
        
        # Same permutation for all models in this seed
        permuted_pool = rng.permutation(train_pool_perm0)
        
        # Filter training sizes that are feasible
        Ns_used = [N for N in n_train_values if N <= len(permuted_pool)]
        
        for N in Ns_used:
            print(f"--- N_train = {N} ---")
            subset_idx = permuted_pool[:N]
            X_train = X_all[subset_idx].T  # (d, N_train)
            Y_train = Y_all[:, subset_idx]  # (m, N_train)
            
            # --- row-wise z-scoring for X and Y (only for baseline model) ---
            sc_x = StandardScaler()
            X_train_std = sc_x.fit_transform(X_train.T).T
            X_test_std = sc_x.transform(X_test.T).T

            # sc_y = StandardScaler()
            # Y_train_std = sc_y.fit_transform(Y_train.T).T
            # Y_test_std = sc_y.transform(Y_test.T).T
            
            for model_name, (kind, kernel, parameter) in models_config.items():
                print(f"--- {model_name} ---")
                
                if kind == "baseline":
                    # Baseline uses standardized data
                    results = evaluate_baseline_model_std(
                        X_train_std, Y_train, X_test_std, Y_test, concept_names
                    )
                elif kind == "linear":
                    # Linear model uses raw data (relies on internal fit_transform)
                    results = evaluate_linear_model_raw(
                        X_train, Y_train, X_test, Y_test, concept_names, alpha, rng
                    )
                elif kind == "logistic":
                    # Logistic model uses raw data (internal fit_transform, binary targets)
                    results = evaluate_logistic_model_raw(
                        X_train, Y_train, X_test, Y_test, concept_names, rng
                    )
                elif kind == "kernel":
                    # Kernel model uses raw data (relies on internal fit_transform)
                    results = evaluate_kernel_model_raw(
                        X_train, Y_train, X_test, Y_test,
                        concept_names, alpha, kernel, parameter, rng
                    )
                else:
                    raise ValueError(f"Unknown model kind: {kind}")
                
                # Add metadata to results
                for result in results:
                    result.update({
                        "model": model_name,
                        "seed": seed,
                        "n_train": N
                    })
                    records.append(result)
    
    return pd.DataFrame(records)
