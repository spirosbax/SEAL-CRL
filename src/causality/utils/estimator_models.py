"""
Advanced estimator model utilities for causal grouping experiments.

This module provides functions for evaluating permutation estimators
(Linear, Logistic, Kernel RBF) with capacitated matching for N→1 alignment.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Optional, Any, Callable
import sys
import os

# Add external library to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(
    os.path.join(project_root, "external", "sample-efficient-learning-of-concepts")
)

from permutation_estimator.estimator import (
    FeaturePermutationEstimator,
    KernelizedPermutationEstimator,
)

from .baseline_models import stratified_subset_from_pool
from .math_utils import r2_safe, sigmoid, add_row_jitter


def standardize_xy(
    X_train: np.ndarray, X_test: np.ndarray, Y_train: np.ndarray, Y_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Row-wise standardization for X and Y matrices.

    Args:
        X_train: Training features (d_vars, N_train)
        X_test: Test features (d_vars, N_test)
        Y_train: Training targets (n_concepts, N_train)
        Y_test: Test targets (n_concepts, N_test)

    Returns:
        Tuple of standardized matrices
    """
    sc_x = StandardScaler()
    X_train_std = sc_x.fit_transform(X_train.T).T
    X_test_std = sc_x.transform(X_test.T).T

    sc_y = StandardScaler()
    Y_train_std = sc_y.fit_transform(Y_train.T).T
    Y_test_std = sc_y.transform(Y_test.T).T

    return X_train_std, X_test_std, Y_train_std, Y_test_std


## helpers imported from math_utils: add_row_jitter, r2_safe, sigmoid


def capacitated_match(
    S: np.ndarray, concepts: List[str], agg_name: str, K: int
) -> Dict[str, List[int]]:
    """
    Perform capacitated matching using Hungarian algorithm.

    For the aggregate concept, we duplicate its row K times to allow
    it to be assigned K latent variables.

    Args:
        S: Score matrix (n_concepts, d_vars) with larger = better
        concepts: List of concept names
        agg_name: Name of the aggregate concept (gets K assignments)
        K: Number of assignments for the aggregate concept

    Returns:
        Dictionary mapping concept names to lists of assigned latent indices
    """
    m, d = S.shape
    agg_idx = concepts.index(agg_name)

    # Create expanded matrix
    rows, owners = [], []
    for i in range(m):
        if i == agg_idx:
            # Duplicate aggregate concept K times
            for _ in range(K):
                rows.append(S[i])
                owners.append(i)
        else:
            rows.append(S[i])
            owners.append(i)

    S_exp = np.stack(rows, axis=0)

    # Hungarian algorithm minimizes, so negate to maximize
    S_exp = np.nan_to_num(S_exp, nan=0.0, posinf=0.0, neginf=0.0)
    r_idx, c_idx = linear_sum_assignment(-S_exp)

    # Map back to original concepts
    assign = {name: [] for name in concepts}
    for r, c in zip(r_idx, c_idx):
        assign[concepts[owners[r]]].append(int(c))

    # Sort assignments by descending score for readability
    for name in assign:
        idx = concepts.index(name)
        assign[name] = sorted(assign[name], key=lambda j: -S[idx, j])

    return assign


def capacitated_match_multiple(
    S: np.ndarray, concepts: List[str], K_assignments: Dict[str, int]
) -> Dict[str, List[int]]:
    """
    Perform capacitated matching using Hungarian algorithm with multiple concepts having different K values.

    For each concept in K_assignments, we duplicate its row K times to allow
    it to be assigned K latent variables.

    Args:
        S: Score matrix (n_concepts, d_vars) with larger = better
        concepts: List of concept names
        K_assignments: Dictionary mapping concept names to number of assignments

    Returns:
        Dictionary mapping concept names to lists of assigned latent indices
    """
    m, d = S.shape

    # Create expanded matrix
    rows, owners = [], []
    for i, concept in enumerate(concepts):
        K = K_assignments.get(concept, 1)  # Default to 1 assignment if not specified
        # Duplicate concept K times
        for _ in range(K):
            rows.append(S[i])
            owners.append(i)

    S_exp = np.stack(rows, axis=0)

    # Hungarian algorithm minimizes, so negate to maximize
    S_exp = np.nan_to_num(S_exp, nan=0.0, posinf=0.0, neginf=0.0)
    r_idx, c_idx = linear_sum_assignment(-S_exp)

    # Map back to original concepts
    assign = {name: [] for name in concepts}
    for r, c in zip(r_idx, c_idx):
        assign[concepts[owners[r]]].append(int(c))

    # Sort assignments by descending score for readability
    for name in assign:
        idx = concepts.index(name)
        assign[name] = sorted(assign[name], key=lambda j: -S[idx, j])

    return assign


def predict_linear_assigned(
    X_std: np.ndarray,
    coef_mat: np.ndarray,
    intercept_vec: Optional[np.ndarray],
    assigned_cols: Dict[str, List[int]],
    concept_names: List[str],
    i: int,
) -> np.ndarray:
    """
    Linear prediction using only assigned columns.

    Args:
        X_std: Standardized features (d_vars, N)
        coef_mat: Coefficient matrix (n_concepts, d_vars)
        intercept_vec: Intercept vector (n_concepts,)
        assigned_cols: Dictionary mapping concept names to assigned latent indices
        concept_names: List of concept names
        i: Index of concept to predict

    Returns:
        Predictions for concept i
    """
    cols = assigned_cols[concept_names[i]]
    if len(cols) == 0:
        return np.zeros(X_std.shape[1])

    w = coef_mat[i, cols]
    Xs = X_std[cols, :]
    b0 = intercept_vec[i] if intercept_vec is not None else 0.0

    return w @ Xs + b0


def predict_logistic_assigned(
    X_std: np.ndarray,
    coef_mat: np.ndarray,
    intercept_vec: Optional[np.ndarray],
    assigned_cols: Dict[str, List[int]],
    concept_names: List[str],
    i: int,
) -> np.ndarray:
    """
    Logistic prediction (probabilities) using only assigned columns.

    Args:
        X_std: Standardized features (d_vars, N)
        coef_mat: Coefficient matrix (n_concepts, d_vars)
        intercept_vec: Intercept vector (n_concepts,)
        assigned_cols: Dictionary mapping concept names to assigned latent indices
        concept_names: List of concept names
        i: Index of concept to predict

    Returns:
        Predicted probabilities for concept i
    """
    cols = assigned_cols[concept_names[i]]
    if len(cols) == 0:
        return np.full(X_std.shape[1], 0.5)

    w = coef_mat[i, cols]
    Xs = X_std[cols, :]
    b0 = intercept_vec[i] if intercept_vec is not None else 0.0

    return w @ Xs + b0


def kernel_group_scores(beta_all: np.ndarray, d_vars: int, n_feat: int) -> np.ndarray:
    """
    Compute group scores for kernel estimator.

    Args:
        beta_all: Flattened coefficients (n_concepts, d_vars * n_feat)
        d_vars: Number of variables
        n_feat: Number of features per variable

    Returns:
        Score matrix (n_concepts, d_vars) with L2 norms
    """
    m = beta_all.shape[0]
    B = beta_all.reshape(m, d_vars, n_feat)
    return np.linalg.norm(B, axis=2)


def predict_kernel_assigned(
    est: Any,
    X_std: np.ndarray,
    beta_all: np.ndarray,
    intercept_vec: Optional[np.ndarray],
    assigned_cols: Dict[str, List[int]],
    concept_names: List[str],
    i: int,
    d_vars: int,
    n_feat: int,
) -> np.ndarray:
    """
    Kernel prediction using assigned variable blocks.

    Args:
        est: Fitted kernel estimator
        X_std: Standardized features (d_vars, N)
        beta_all: Flattened coefficients (n_concepts, d_vars * n_feat)
        intercept_vec: Intercept vector (n_concepts,)
        assigned_cols: Dictionary mapping concept names to assigned latent indices
        concept_names: List of concept names
        i: Index of concept to predict
        d_vars: Number of variables
        n_feat: Number of features per variable

    Returns:
        Predictions for concept i
    """
    phi = _kernel_transform(est, X_std)  # (d_vars*n_feat, N)
    cols = assigned_cols[concept_names[i]]

    if len(cols) == 0:
        return np.zeros(phi.shape[1])

    # Reshape coefficients
    b_row = beta_all[i].reshape(d_vars, n_feat)
    out = np.zeros(phi.shape[1])

    # Sum contributions from assigned variable blocks
    for j in cols:
        block = slice(j * n_feat, (j + 1) * n_feat)
        out += b_row[j] @ phi[block, :]

    b0 = intercept_vec[i] if intercept_vec is not None else 0.0
    return out + b0


def _get_intercept(obj: Any, fallback_shape0: int) -> np.ndarray:
    """
    Safely extract intercept from estimator object.

    Args:
        obj: Estimator object
        fallback_shape0: Fallback shape for intercept array

    Returns:
        Intercept array
    """
    for name in ("intercept_hat_all_", "intercept_", "_intercept"):
        if hasattr(obj, name):
            arr = getattr(obj, name)
            return np.array(arr, dtype=float).reshape(-1)
    return np.zeros(fallback_shape0, dtype=float)


def _kernel_transform(est: Any, X_std: np.ndarray) -> np.ndarray:
    """
    Safely apply kernel transformation.

    Args:
        est: Kernel estimator
        X_std: Standardized features

    Returns:
        Transformed features
    """
    if hasattr(est, "transform"):
        return est.transform(X_std)
    if hasattr(est, "_optim") and hasattr(est._optim, "transform"):
        return est._optim.transform(X_std)
    raise RuntimeError("Kernelized estimator has no .transform method exposed.")


def evaluate_estimator_models(
    X_all: np.ndarray,
    Y_all: np.ndarray,
    concept_names: List[str],
    agg_name: str,
    K_agg: int,
    split_dict: Dict[str, np.ndarray],
    N_train_values: List[int],
    alpha_fixed: float,
    kernel_gamma: float = 0.5,
    max_nystrom: int = 64,
    max_n_value: Optional[int] = 1280,
    seeds: List[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate permutation estimator models with capacitated matching.

    Args:
        X_all: Full feature matrix (N, d_vars)
        Y_all: Full target matrix (n_concepts, N)
        concept_names: List of concept names
        agg_name: Name of aggregate concept (gets K assignments)
        K_agg: Number of assignments for aggregate concept
        split_dict: Dictionary with 'test_idx' and 'train_pool_perm'
        N_train_values: List of training set sizes
        alpha_fixed: Regularization parameter
        kernel_gamma: RBF kernel gamma parameter
        max_nystrom: Maximum Nyström components per latent
        max_n_value: Maximum N_train value to include (filters N_train_values)
        seeds: List of random seeds
        verbose: Whether to print progress

    Returns:
        Tuple of (results DataFrame, picks_by_seed dictionary)
    """
    if seeds is None:
        seeds = [0, 1, 2]

    # Filter N_train_values to only include values up to and including max_n_value
    if max_n_value is not None:
        N_train_values = [n for n in N_train_values if n <= max_n_value]
    
    test_idx = split_dict["test_idx"]
    train_pool_base = split_dict["train_pool_perm"]

    records = []
    picks_by_seed = {s: {agg_name: []} for s in seeds}

    for seed in seeds:
        if verbose:
            print(f"\n*** Seed: {seed} ***")

        rng = np.random.default_rng(seed)
        permuted_pool = rng.permutation(train_pool_base)

        # Fixed test split for all models
        X_test = X_all[test_idx].T  # (d_vars, N_test)
        Y_test = Y_all[:, test_idx]  # (n_concepts, N_test)

        for N in N_train_values:
            # Skip N values that are too large for the training pool, but still append empty picks
            if N > len(permuted_pool):
                picks_by_seed[seed][agg_name].append([])
                continue

            if verbose and seed == seeds[0]:  # Only print for first seed
                print(f"  N_train: {N}")

            subset_idx = permuted_pool[:N]
            X_train = X_all[subset_idx].T  # (d_vars, N_train)
            Y_train = Y_all[:, subset_idx]  # (n_concepts, N_train)

            # Apply jitter to raw data to avoid numerical issues
            Y_train_j = add_row_jitter(Y_train, rng)

            # 1) Linear (group-lasso with identity features)
            try:
                if verbose and seed == seeds[0]:  # Debug info for first seed only
                    print(f"    DEBUG: Linear model - Training data shapes: X_train_j={X_train.shape}, Y_train_j={Y_train_j.shape}")
                    print(f"    DEBUG: Linear model - Training data stats: X_train_j range=[{np.min(X_train):.3f}, {np.max(X_train):.3f}]")
                    print(f"    DEBUG: Linear model - Y_train_j stats: {[f'{concept}: min={np.min(Y_train_j[i]):.3f}, max={np.max(Y_train_j[i]):.3f}, unique={len(np.unique(Y_train_j[i]))}' for i, concept in enumerate(concept_names)]}")
                    print(f"    DEBUG: Linear model - alpha_fixed={alpha_fixed}")
                
                lin = FeaturePermutationEstimator(
                    regularizer="lasso",
                    optim_kwargs={"alpha": alpha_fixed},
                    feature_transform=None,
                    d_variables=X_train.shape[0],
                    n_features=1,
                )
                lin.fit(X_train, Y_train_j) # no transform here, we use the estimator's transform inside the fit method

                # Get coefficient info and assignment tracking
                coef_lin = lin.beta_hat_all_
                intercept_lin = _get_intercept(lin, coef_lin.shape[0])

                if verbose and seed == seeds[0]:  # Debug coefficient info
                    print(f"    DEBUG: Linear model - coef_lin shape: {coef_lin.shape}")
                    print(f"    DEBUG: Linear model - coef_lin stats: min={np.min(coef_lin):.6f}, max={np.max(coef_lin):.6f}, mean={np.mean(coef_lin):.6f}")
                    print(f"    DEBUG: Linear model - coef_lin non-zero count: {np.count_nonzero(coef_lin)}/{coef_lin.size}")
                    print(f"    DEBUG: Linear model - intercept_lin: {intercept_lin}")

                # Capacitated matching for N→1 assignment
                S_lin = np.abs(coef_lin)
                assign_lin = capacitated_match(S_lin, concept_names, agg_name, K=K_agg)
                current_picks = assign_lin[agg_name]
                
                if verbose and seed == seeds[0]:  # Debug assignment info
                    print(f"    DEBUG: Linear model - S_lin shape: {S_lin.shape}")
                    print(f"    DEBUG: Linear model - S_lin stats: min={np.min(S_lin):.6f}, max={np.max(S_lin):.6f}")
                    print(f"    DEBUG: Linear model - assignments: {assign_lin}")
                    print(f"    DEBUG: Linear model - current_picks for {agg_name}: {current_picks}")

                # Get transformed test data
                X_test_transformed = lin.transform(X_test)

                # Hybrid prediction: use capacitated for aggregate, standard for others
                for i, concept in enumerate(concept_names):
                    if concept == agg_name:
                        # Use capacitated matching prediction for aggregate concept (N→1)
                        yhat = predict_linear_assigned(
                            X_test_transformed,
                            coef_lin,
                            intercept_lin,
                            assign_lin,
                            concept_names,
                            i,
                        )
                    else:
                        # Use standard permutation prediction for non-aggregate concepts (1→1)
                        Y_hat_standard = lin.predict_match(X_test)
                        yhat = Y_hat_standard[i]
                    
                    r2 = r2_safe(Y_test[i], yhat)
                    records.append(
                        {
                            "model": "Linear",
                            "seed": seed,
                            "concept": concept,
                            "n_train": N,
                            "r2": r2,
                        }
                    )

            except Exception as e:
                if verbose:
                    print(f"    ERROR: Linear model failed: {type(e).__name__}: {e}")
                    import traceback
                    print(f"    ERROR: Linear model traceback:\n{traceback.format_exc()}")
                for concept in concept_names:
                    records.append(
                        {
                            "model": "Linear",
                            "seed": seed,
                            "concept": concept,
                            "n_train": N,
                            "r2": np.nan,
                        }
                    )
                current_picks = []  # Empty picks when Linear model fails

            # 2) Logistic (group-lasso)
            try:
                # Check if both classes exist in training data
                ytr_raw = Y_train.copy()
                ok_classes = [
                    np.unique(ytr_raw[i]).size > 1 for i in range(len(concept_names))
                ]

                if all(ok_classes):
                    logi = FeaturePermutationEstimator( # no transform here, we use the estimator's transform inside the fit method
                        regularizer="logistic",
                        optim_kwargs={
                            "penalty": "l1",
                            "solver": "liblinear",
                            "C": 1.0,
                            "class_weight": "balanced",
                        },
                        feature_transform=None,
                        d_variables=X_train.shape[0],
                        n_features=1,
                    )
                    logi.fit(X_train, Y_train)

                    # Get coefficient info for assignment tracking
                    coef_logi = logi.beta_hat_all_
                    intercept_logi = _get_intercept(logi, coef_logi.shape[0])

                    # Capacitated matching for N→1 assignment
                    S_logi = np.abs(coef_logi)
                    # returns dictionary mapping concept names to lists of assigned latent indices
                    assign_logi = capacitated_match(
                        S_logi, concept_names, agg_name, K=K_agg
                    )

                    # Get transformed test data
                    X_test_transformed = logi.transform(X_test)

                    # Hybrid prediction: use capacitated for aggregate, standard for others
                    for i, concept in enumerate(concept_names):
                        if concept == agg_name:
                            # Use capacitated matching prediction for aggregate concept (N→1)
                            yhat = predict_logistic_assigned(
                                X_test_transformed,
                                coef_logi,
                                intercept_logi,
                                assign_logi,
                                concept_names,
                                i,
                            )
                        else:
                            # Use standard permutation prediction for non-aggregate concepts (1→1)
                            Y_hat_standard = logi.predict_match(X_test) # output is a logit
                            yhat = Y_hat_standard[i]
                        
                        r2 = r2_safe(Y_test[i], sigmoid(yhat))
                        records.append(
                            {
                                "model": "Logistic",
                                "seed": seed,
                                "concept": concept,
                                "n_train": N,
                                "r2": r2,
                            }
                        )
                else:
                    # Not enough class diversity
                    for concept in concept_names:
                        records.append(
                            {
                                "model": "Logistic",
                                "seed": seed,
                                "concept": concept,
                                "n_train": N,
                                "r2": np.nan,
                            }
                        )

            except Exception as e:
                if verbose:
                    print(f"    ERROR: Logistic model failed: {type(e).__name__}: {e}")
                    import traceback
                    print(f"    ERROR: Logistic model traceback:\n{traceback.format_exc()}")
                for concept in concept_names:
                    records.append(
                        {
                            "model": "Logistic",
                            "seed": seed,
                            "concept": concept,
                            "n_train": N,
                            "r2": np.nan,
                        }
                    )

            # 3) Kernelized (RBF Nyström per latent) + lasso
            try:
                n_feat = min(max_nystrom, X_train.shape[1])
                kern = KernelizedPermutationEstimator(
                    regularizer="lasso",
                    optim_kwargs={"alpha": alpha_fixed},
                    kernel="rbf",
                    parameter=kernel_gamma,
                    d_variables=X_train.shape[0],
                    n_features=n_feat,
                    groups=None,
                    two_stage=None,
                )
                kern.fit(X_train, Y_train_j, is_fit_transformed=False)

                beta_k = kern.beta_hat_all_
                b0_k = _get_intercept(kern, beta_k.shape[0])
                S_k = kernel_group_scores(
                    beta_k, d_vars=X_train.shape[0], n_feat=n_feat
                )
                S_k = np.nan_to_num(S_k, nan=0.0, posinf=0.0, neginf=0.0)

                # Capacitated matching for N→1 assignment
                assign_k = capacitated_match(S_k, concept_names, agg_name, K=K_agg)
                current_picks = assign_k[agg_name]

                # Get intercept for kernel model
                b0_k = _get_intercept(kern, beta_k.shape[0])

                # Hybrid prediction: use capacitated for aggregate, standard for others
                for i, concept in enumerate(concept_names):
                    if concept == agg_name:
                        # Use capacitated matching prediction for aggregate concept (N→1)
                        yhat = predict_kernel_assigned(
                            kern,
                            X_test,  # Use raw X_test, function will transform internally
                            beta_k,
                            b0_k,
                            assign_k,
                            concept_names,
                            i,
                            X_train.shape[0],
                            n_feat,
                        )
                    else:
                        # Use standard permutation prediction for non-aggregate concepts (1→1)
                        Y_hat_standard = kern.predict_match(X_test)
                        yhat = Y_hat_standard[i]
                    
                    r2 = r2_safe(Y_test[i], yhat)
                    records.append(
                        {
                            "model": "Kernel RBF",
                            "seed": seed,
                            "concept": concept,
                            "n_train": N,
                            "r2": r2,
                        }
                    )

            except Exception as e:
                if verbose:
                    print(f"    ERROR: Kernel model failed: {type(e).__name__}: {e}")
                    import traceback
                    print(f"    ERROR: Kernel model traceback:\n{traceback.format_exc()}")
                for concept in concept_names:
                    records.append(
                        {
                            "model": "Kernel RBF",
                            "seed": seed,
                            "concept": concept,
                            "n_train": N,
                            "r2": np.nan,
                        }
                    )

            # At the end of each valid N iteration, store the picks (from Linear model)
            picks_by_seed[seed][agg_name].append(current_picks)

    return pd.DataFrame(records), picks_by_seed


def summarize_estimator_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize estimator model results across seeds.

    Args:
        df: DataFrame with individual results

    Returns:
        DataFrame with mean, min, max statistics
    """
    summary = (
        df.groupby(["model", "concept", "n_train"])["r2"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    return summary


def evaluate_estimator_models_multiple(
    X_all: np.ndarray,
    Y_all: np.ndarray,
    concept_names: List[str],
    K_assignments: Dict[str, int],
    split_dict: Dict[str, np.ndarray],
    N_train_values: List[int],
    alpha_fixed: float,
    kernel_gamma: float = 0.5,
    max_nystrom: int = 64,
    max_n_value: Optional[int] = 1280,
    seeds: List[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate permutation estimator models with multiple predicates in unified capacitated matching.

    Args:
        X_all: Full feature matrix (N, d_vars)
        Y_all: Full target matrix (n_concepts, N)
        concept_names: List of concept names
        K_assignments: Dictionary mapping concept names to number of assignments
        split_dict: Dictionary with 'test_idx' and 'train_pool_perm'
        N_train_values: List of training set sizes
        alpha_fixed: Regularization parameter
        kernel_gamma: RBF kernel gamma parameter
        max_nystrom: Maximum Nyström components per latent
        max_n_value: Maximum N_train value to include (filters N_train_values)
        seeds: List of random seeds
        verbose: Whether to print progress

    Returns:
        Tuple of (results DataFrame, picks_by_seed dictionary for each concept)
    """
    if seeds is None:
        seeds = [0, 1, 2]

    # Filter N_train_values to only include values up to and including max_n_value
    if max_n_value is not None:
        N_train_values = [n for n in N_train_values if n <= max_n_value]
    
    test_idx = split_dict["test_idx"]
    train_pool_base = split_dict["train_pool_perm"]

    records = []
    # Track picks for ALL concepts, organized by model
    model_names = ["Baseline (cap-LR)", "Linear", "Logistic", "Kernel RBF"]
    picks_by_seed = {
        s: {
            model: {name: [] for name in K_assignments.keys()} 
            for model in model_names
        } 
        for s in seeds
    }

    for seed in seeds:
        if verbose:
            print(f"\n*** Seed: {seed} ***")

        rng = np.random.default_rng(seed)
        permuted_pool = rng.permutation(train_pool_base)

        # Fixed test split for all models
        X_test = X_all[test_idx].T  # (d_vars, N_test)
        Y_test = Y_all[:, test_idx]  # (n_concepts, N_test)

        for N in N_train_values:
            # Skip N values that are too large for the training pool
            if N > len(permuted_pool):
                for concept_name, K in K_assignments.items():
                    if K > 1:
                        picks_by_seed[seed][concept_name].append([])
                continue

            if verbose and seed == seeds[0]:  # Only print for first seed
                print(f"  N_train: {N}")

            subset_idx = permuted_pool[:N]
            X_train = X_all[subset_idx].T  # (d_vars, N_train)
            Y_train = Y_all[:, subset_idx]  # (n_concepts, N_train)

            # Apply jitter to raw data to avoid numerical issues
            Y_train_j = add_row_jitter(Y_train, rng)

            # 0) Baseline (Spearman + capacitated + LR)
            try:
                # Standardize X for correlation and LR consistency
                sc_x = StandardScaler().fit(X_train.T)
                Xtr_std = sc_x.transform(X_train.T).T
                Xte_std = sc_x.transform(X_test.T).T

                m, d = Y_train.shape[0], Xtr_std.shape[0]
                C = np.zeros((m, d), dtype=float)
                for i in range(m):
                    y = Y_train[i]
                    if np.unique(y).size < 2:
                        continue
                    for j in range(d):
                        rho, _ = spearmanr(y, Xtr_std[j, :])
                        C[i, j] = 0.0 if np.isnan(rho) else abs(rho)

                assign_base = capacitated_match_multiple(C, concept_names, K_assignments)

                for i, concept in enumerate(concept_names):
                    cols = assign_base.get(concept, [])
                    if len(cols) == 0 or np.unique(Y_train[i]).size < 2 or np.std(Y_test[i]) == 0:
                        r2 = np.nan
                    else:
                        lm = LinearRegression().fit(Xtr_std[cols, :].T, Y_train[i])
                        yhat = lm.predict(Xte_std[cols, :].T)
                        r2 = r2_safe(Y_test[i], yhat)
                    records.append(
                        {
                            "model": "Baseline (cap-LR)",
                            "seed": seed,
                            "concept": concept,
                            "n_train": N,
                            "r2": r2,
                        }
                    )
            except Exception as e:
                if verbose:
                    print(f"    ERROR: Baseline (cap-LR) failed: {type(e).__name__}: {e}")
                    import traceback
                    print(f"    ERROR: Baseline traceback:\n{traceback.format_exc()}")
                for concept in concept_names:
                    records.append(
                        {
                            "model": "Baseline (cap-LR)",
                            "seed": seed,
                            "concept": concept,
                            "n_train": N,
                            "r2": np.nan,
                        }
                    )

            # 1) Linear (lasso with identity features)
            try:
                if verbose and seed == seeds[0]:
                    print(f"    DEBUG: Linear model - Training data shapes: X_train={X_train.shape}, Y_train_j={Y_train_j.shape}")
                    print(f"    DEBUG: Linear model - K_assignments={K_assignments}")
                
                lin = FeaturePermutationEstimator(
                    regularizer="lasso",
                    optim_kwargs={"alpha": alpha_fixed},
                    feature_transform=None,
                    d_variables=X_train.shape[0],
                    n_features=1,
                )
                lin.fit(X_train, Y_train_j)

                # Get coefficient info and assignment tracking
                coef_lin = lin.beta_hat_all_
                intercept_lin = _get_intercept(lin, coef_lin.shape[0])

                # Unified capacitated matching for ALL concepts
                S_lin = np.abs(coef_lin)
                assign_lin = capacitated_match_multiple(S_lin, concept_names, K_assignments)
                
                # Store picks for all concepts
                for concept_name in K_assignments.keys():
                    picks_by_seed[seed]["Linear"][concept_name].append(assign_lin[concept_name])
                
                if verbose and seed == seeds[0]:
                    print(f"    DEBUG: Linear model - assignments: {assign_lin}")

                # Get transformed test data
                X_test_transformed = lin.transform(X_test)

                # Predict for ALL concepts using capacitated assignments
                for i, concept in enumerate(concept_names):
                    yhat = predict_linear_assigned(
                        X_test_transformed,
                        coef_lin,
                        intercept_lin,
                        assign_lin,
                        concept_names,
                        i,
                    )
                    r2 = r2_safe(Y_test[i], yhat)
                    records.append(
                        {
                            "model": "Linear",
                            "seed": seed,
                            "concept": concept,
                            "n_train": N,
                            "r2": r2,
                        }
                    )

            except Exception as e:
                if verbose:
                    print(f"    ERROR: Linear model failed: {type(e).__name__}: {e}")
                    import traceback
                    print(f"    ERROR: Linear model traceback:\n{traceback.format_exc()}")
                for concept in concept_names:
                    records.append(
                        {
                            "model": "Linear",
                            "seed": seed,
                            "concept": concept,
                            "n_train": N,
                            "r2": np.nan,
                        }
                    )
                # Empty picks when Linear model fails
                for concept_name in K_assignments.keys():
                    picks_by_seed[seed]["Linear"][concept_name].append([])

            # 2) Logistic (logistic regression)
            try:
                # Check if both classes exist in training data
                ytr_raw = Y_train.copy()
                ok_classes = [
                    np.unique(ytr_raw[i]).size > 1 for i in range(len(concept_names))
                ]

                if all(ok_classes):
                    logi = FeaturePermutationEstimator(
                        regularizer="logistic",
                        optim_kwargs={
                            "penalty": "l1",
                            "solver": "liblinear",
                            "C": 1.0,
                            "class_weight": "balanced",
                        },
                        feature_transform=None,
                        d_variables=X_train.shape[0],
                        n_features=1,
                    )
                    logi.fit(X_train, Y_train)

                    # Get coefficient info for assignment tracking
                    coef_logi = logi.beta_hat_all_
                    intercept_logi = _get_intercept(logi, coef_logi.shape[0])

                    # Unified capacitated matching for ALL concepts
                    S_logi = np.abs(coef_logi)
                    assign_logi = capacitated_match_multiple(S_logi, concept_names, K_assignments)
                    
                    # Store picks for all concepts
                    for concept_name in K_assignments.keys():
                        picks_by_seed[seed]["Logistic"][concept_name].append(assign_logi[concept_name])

                    # Get transformed test data
                    X_test_transformed = logi.transform(X_test)

                    # Predict for ALL concepts using capacitated assignments
                    for i, concept in enumerate(concept_names):
                        logit = predict_logistic_assigned(
                            X_test_transformed,
                            coef_logi,
                            intercept_logi,
                            assign_logi,
                            concept_names,
                            i,
                        )
                        r2 = r2_safe(Y_test[i], sigmoid(logit))
                        records.append(
                            {
                                "model": "Logistic",
                                "seed": seed,
                                "concept": concept,
                                "n_train": N,
                                "r2": r2,
                            }
                        )
                else:
                    # Not enough class diversity
                    for concept in concept_names:
                        records.append(
                            {
                                "model": "Logistic",
                                "seed": seed,
                                "concept": concept,
                                "n_train": N,
                                "r2": np.nan,
                            }
                        )

            except Exception as e:
                if verbose:
                    print(f"    ERROR: Logistic model failed: {type(e).__name__}: {e}")
                    import traceback
                    print(f"    ERROR: Logistic model traceback:\n{traceback.format_exc()}")
                for concept in concept_names:
                    records.append(
                        {
                            "model": "Logistic",
                            "seed": seed,
                            "concept": concept,
                            "n_train": N,
                            "r2": np.nan,
                        }
                    )

            # 3) Kernelized (RBF Nyström per latent) + lasso
            try:
                n_feat = min(max_nystrom, X_train.shape[1])
                kern = KernelizedPermutationEstimator(
                    regularizer="lasso",
                    optim_kwargs={"alpha": alpha_fixed},
                    kernel="rbf",
                    parameter=kernel_gamma,
                    d_variables=X_train.shape[0],
                    n_features=n_feat,
                    groups=None,
                    two_stage=None,
                )
                kern.fit(X_train, Y_train_j, is_fit_transformed=False)

                beta_k = kern.beta_hat_all_
                b0_k = _get_intercept(kern, beta_k.shape[0])
                S_k = kernel_group_scores(
                    beta_k, d_vars=X_train.shape[0], n_feat=n_feat
                )
                S_k = np.nan_to_num(S_k, nan=0.0, posinf=0.0, neginf=0.0)

                # Unified capacitated matching for ALL concepts
                assign_k = capacitated_match_multiple(S_k, concept_names, K_assignments)
                
                # Store picks for all concepts
                for concept_name in K_assignments.keys():
                    picks_by_seed[seed]["Kernel RBF"][concept_name].append(assign_k[concept_name])

                # Predict for ALL concepts using capacitated assignments
                for i, concept in enumerate(concept_names):
                    yhat = predict_kernel_assigned(
                        kern,
                        X_test,  # Use raw X_test, function will transform internally
                        beta_k,
                        b0_k,
                        assign_k,
                        concept_names,
                        i,
                        X_train.shape[0],
                        n_feat,
                    )
                    r2 = r2_safe(Y_test[i], yhat)
                    records.append(
                        {
                            "model": "Kernel RBF",
                            "seed": seed,
                            "concept": concept,
                            "n_train": N,
                            "r2": r2,
                        }
                    )

            except Exception as e:
                if verbose:
                    print(f"    ERROR: Kernel model failed: {type(e).__name__}: {e}")
                    import traceback
                    print(f"    ERROR: Kernel model traceback:\n{traceback.format_exc()}")
                for concept in concept_names:
                    records.append(
                        {
                            "model": "Kernel RBF",
                            "seed": seed,
                            "concept": concept,
                            "n_train": N,
                            "r2": np.nan,
                        }
                    )

    return pd.DataFrame(records), picks_by_seed
