#!/usr/bin/env python3
"""
End-to-end robustness for multiple aggregate predicates with capacitated matching.

We perturb ALL latent dimensions in the train set, re-fit + re-match, then measure:
  - Global assignment stability:
      * jaccard_global (↑): Jaccard between the union of all matched latent indices
  - Per-concept assignment stability:
      * jaccard_concept (↑): Jaccard of matched indices for each concept
  - Per-concept prediction stability:
      * pred_corr_concept (↑): Pearson corr between base and noisy predictions

Models supported:
  - Baseline (Spearman + capacitated Hungarian + multi-variate LR per concept)
  - Linear (FeaturePermutationEstimator, lasso)
  - Logistic (FeaturePermutationEstimator, sklearn LogisticRegression)
  - Kernel RBF (KernelizedPermutationEstimator, Nyström RBF)
"""

import os, sys, argparse, numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Repo imports
from causality.data.ithor_loader import load_causality_data
from causality.utils.utils import setup_output_dir
from causality.utils.baseline_models import (
    create_train_test_split,
    create_aggregate_predicate,
    get_predicate_configuration,
)
from causality.utils.visualization import MODEL_COLOR_MAP as GLOBAL_MODEL_COLOR_MAP

# External permutation estimators
from permutation_estimator.estimator import (
    FeaturePermutationEstimator,
    KernelizedPermutationEstimator,
)


# ------------------------- Helpers -------------------------

def add_row_jitter(A: np.ndarray, rng: np.random.Generator, noise: float = 1e-6) -> np.ndarray:
    """Add tiny noise to rows with zero variance to avoid degenerate fits.

    This is used on Y when a concept is constant within a subset to prevent
    numerical issues in downstream estimators.
    """
    A2 = A.copy()
    row_std = A2.std(axis=1, ddof=0)
    idx = np.where(row_std == 0.0)[0]
    if idx.size:
        A2[idx] += rng.normal(0.0, noise, size=(idx.size, A2.shape[1]))
    return A2

def standardize_x(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Z-score standardize X per-latent dimension using training statistics.

    Returns X_train and X_test standardized consistently.
    """
    sc_x = StandardScaler()
    Xtr = sc_x.fit_transform(X_train.T).T
    Xte = sc_x.transform(X_test.T).T
    return Xtr, Xte

def jaccard_set(a: List[int], b: List[int]) -> float:
    """Compute Jaccard similarity between two sets of column indices.

    Ignores negative indices and returns 1.0 for two empty sets.
    """
    A = set(int(x) for x in a if x >= 0)
    B = set(int(x) for x in b if x >= 0)
    if not A and not B:
        return 1.0
    return len(A & B) / max(1, len(A | B))

def corr_safe(y0: np.ndarray, y1: np.ndarray) -> float:
    """Numerically safe Pearson correlation with edge-case handling."""
    if y0.size == 0 or y1.size == 0:
        return np.nan
    if np.std(y0) == 0 or np.std(y1) == 0:
        return 1.0 if np.allclose(y0, y1) else 0.0
    c = np.corrcoef(y0, y1)[0, 1]
    return 0.0 if np.isnan(c) else float(c)

def _get_intercept(obj, fallback_len: int) -> np.ndarray:
    """Extract per-concept intercept vector from estimator if available."""
    for name in ("intercept_hat_all_", "intercept_", "_intercept"):
        if hasattr(obj, name):
            arr = getattr(obj, name)
            return np.array(arr, dtype=float).reshape(-1)
    return np.zeros(fallback_len, dtype=float)

def capacitated_match_multiple(S: np.ndarray, concept_names: List[str], K_assignments: Dict[str, int]) -> Dict[str, List[int]]:
    """Capacitated matching across all concepts using Hungarian on an expanded score matrix.

    For each concept i, we duplicate its score row K_i times to enforce that
    exactly K_i latents are assigned to that concept. Rows are tracked back to
    their concept owners to reconstruct the per-concept assignments.
    """
    m, d = S.shape
    rows, owners = [], []
    for i, concept in enumerate(concept_names):
        K = int(K_assignments.get(concept, 1))
        for _ in range(K):
            rows.append(S[i])
            owners.append(i)
    S_exp = np.stack(rows, axis=0)
    S_exp = np.nan_to_num(S_exp, nan=0.0, posinf=0.0, neginf=0.0)
    r_idx, c_idx = linear_sum_assignment(-S_exp)
    assign = {name: [] for name in concept_names}
    for r, c in zip(r_idx, c_idx):
        assign[concept_names[owners[r]]].append(int(c))
    for name in assign:
        i = concept_names.index(name)
        assign[name] = sorted(assign[name], key=lambda j: -S[i, j])
    return assign

def build_predicates(predicates: List[str], sim_var_name_to_data: Dict[str, np.ndarray]) -> Tuple[List[str], Dict[str, int]]:
    """Create per-concept target rows and capacities from requested predicates.

    - Aggregate predicates (stoves, egg) are converted to new concept rows with
      user-friendly names (e.g., All_Stoves_On) and appropriate capacities.
    - Raw simulator variables (e.g., Microwave_d8b935e4_open) are passed through
      as single-capacity concepts.
    Returns concept_names (row labels) and K_assignments (capacities per concept).
    """
    concept_names: List[str] = []
    K_assignments: Dict[str, int] = {}
    for predicate in predicates:
        try:
            # Supported aggregate predicates (stoves, egg)
            cfg = get_predicate_configuration(predicate)
            vars_for_pred = cfg['variables']
            agg_name = predicate.replace('_', ' ').title().replace(' ', '_')
            concept_names.append(agg_name)
            if predicate in ["all_stoves_on", "at_least_two_stoves_on", "at_least_three_stoves_on"]:
                K_assignments[agg_name] = len(vars_for_pred)
            else:
                K_assignments[agg_name] = 2
            y_pred = create_aggregate_predicate(sim_var_name_to_data, predicate, vars_for_pred)
            sim_var_name_to_data[agg_name] = y_pred.astype(int)
        except Exception:
            # Fall back: treat as raw concept name if present in data (e.g., Microwave_d8b935e4_open)
            if predicate in sim_var_name_to_data:
                concept_names.append(predicate)
                K_assignments[predicate] = 1
            else:
                raise
    return concept_names, K_assignments

def perturb_train_all_dims(X_train: np.ndarray, rng: np.random.Generator, sigma: float, row_std: np.ndarray = None) -> np.ndarray:
    """Add Gaussian noise to every latent dimension of X_train, scaled per-dim std.

    Used to evaluate robustness by refitting and rematching under perturbed data.
    """
    if row_std is None:
        row_std = X_train.std(axis=1, ddof=0)
        row_std = row_std.copy()
        row_std[row_std == 0.0] = 1.0
    noise = rng.normal(0.0, sigma * row_std[:, None], size=X_train.shape)
    return X_train + noise


# ------------------------- Baseline: Spearman + capacitated + LR -------------------------

def baseline_match_capacitated(X_train: np.ndarray, Y_train: np.ndarray, concept_names: List[str], K_assign: Dict[str, int]) -> Dict[str, List[int]]:
    """Compute absolute Spearman correlation per (concept, latent), then assign with capacities."""
    # Standardize X rows for correlation stability
    Xtr, _ = standardize_x(X_train, X_train)
    m, d = Y_train.shape[0], Xtr.shape[0]
    from scipy.stats import spearmanr
    C = np.zeros((m, d))
    for i in range(m):
        y = Y_train[i]
        if np.unique(y).size < 2:
            continue
        for j in range(d):
            rho, _ = spearmanr(y, Xtr[j, :])
            C[i, j] = 0.0 if np.isnan(rho) else abs(rho)
    return capacitated_match_multiple(C, concept_names, K_assign)

def baseline_predict_assigned(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, assignments: Dict[str, List[int]], concept_names: List[str]) -> np.ndarray:
    """Train per-concept linear regressions on assigned latents and predict on test."""
    # Standardize X for LR consistency
    Xtr, Xte = standardize_x(X_train, X_test)
    m = len(concept_names)
    Yhat = np.full((m, Xte.shape[1]), np.nan, dtype=float)
    for i, name in enumerate(concept_names):
        cols = assignments.get(name, [])
        if len(cols) == 0 or np.unique(Y_train[i]).size < 2:
            continue
        lm = LinearRegression()
        lm.fit(Xtr[cols, :].T, Y_train[i])
        Yhat[i] = lm.predict(Xte[cols, :].T)
    return Yhat


# ------------------------- Permutation estimators (capacitied assign + assigned predict) -------------------------

def predict_linear_assigned(est: FeaturePermutationEstimator, X: np.ndarray, beta_all: np.ndarray, intercept_vec: np.ndarray, assignments: Dict[str, List[int]], concept_names: List[str]) -> np.ndarray:
    """Predict with linear permutation estimator restricted to assigned columns."""
    phi = est.transform(X)  # (d_vars, N) for n_features=1
    m, N = beta_all.shape[0], phi.shape[1]
    Yhat = np.zeros((m, N))
    for i, name in enumerate(concept_names):
        cols = assignments.get(name, [])
        if len(cols) == 0:
            continue
        Yhat[i] = beta_all[i, cols] @ phi[cols, :] + (intercept_vec[i] if intercept_vec is not None else 0.0)
    return Yhat

def predict_logistic_assigned(est: FeaturePermutationEstimator, X: np.ndarray, beta_all: np.ndarray, intercept_vec: np.ndarray, assignments: Dict[str, List[int]], concept_names: List[str]) -> np.ndarray:
    """Predict logits with logistic permutation estimator restricted to assigned columns.

    Returns raw logits (no sigmoid) for correlation stability analysis.
    """
    phi = est.transform(X)
    m, N = beta_all.shape[0], phi.shape[1]
    Yhat = np.zeros((m, N))
    for i, name in enumerate(concept_names):
        cols = assignments.get(name, [])
        if len(cols) == 0:
            continue
        Yhat[i] = beta_all[i, cols] @ phi[cols, :] + (intercept_vec[i] if intercept_vec is not None else 0.0)
    return Yhat

def predict_kernel_assigned(est: KernelizedPermutationEstimator, X: np.ndarray, beta_all: np.ndarray, intercept_vec: np.ndarray, assignments: Dict[str, List[int]], concept_names: List[str], d_vars: int, n_feat: int) -> np.ndarray:
    """Predict with kernelized estimator by summing blocks for assigned latent variables."""
    phi = est.transform(X)  # (d_vars*n_feat, N)
    m, N = beta_all.shape[0], phi.shape[1]
    Yhat = np.zeros((m, N))
    for i, name in enumerate(concept_names):
        cols = assignments.get(name, [])
        if len(cols) == 0:
            continue
        b_row = beta_all[i].reshape(d_vars, n_feat)
        out = np.zeros(N)
        for j in cols:
            block = slice(j*n_feat, (j+1)*n_feat)
            out += b_row[j] @ phi[block, :]
        Yhat[i] = out + (intercept_vec[i] if intercept_vec is not None else 0.0)
    return Yhat


# ------------------------- Main -------------------------

def parse_args():
    """CLI parameters for grouped robustness with capacitated matching.

    Note: When multiple stove predicates are provided, the script will run
    separate subruns for each stove predicate with a microwave control and
    merge results to avoid latent competition across stove concepts.
    """
    p = argparse.ArgumentParser("Grouped end-to-end robustness (capacitated)")
    p.add_argument("--data_dir", type=str, default="src/data/ithor")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--output_dir", type=str, default="results/robustness/groups")
    p.add_argument("--no_cache", action="store_true")

    p.add_argument("--predicates", type=str, nargs="+", default=["all_stoves_on", "at_least_three_stoves_on", "at_least_two_stoves_on", "egg_intact"], help="Aggregate predicates (e.g., all_stoves_on egg_intact microwave_on)")

    # Estimator params
    p.add_argument("--alpha", type=float, default=2.154e-3)
    p.add_argument("--kernel_gamma", type=float, default=0.5)
    p.add_argument("--nystrom_components", type=int, default=64)

    # Experiment params
    p.add_argument("--models", type=str, nargs="+", default=["baseline","linear","logistic","kernel_rbf"], choices=["baseline","linear","logistic","kernel_rbf"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    p.add_argument("--n_train_values", type=int, nargs="+", default=[5,10,20,40,80,160,320,640,1280])
    p.add_argument("--test_fraction", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--n_trials", type=int, default=3)
    p.add_argument("--sigma", type=float, default=0.5)

    # Plot & performance
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--skip_per_concept", action="store_true")
    p.add_argument("--balance_classes", action="store_true", help="Balance classes per subrun using the primary predicate")
    return p.parse_args()


def main():
    """Entry point: orchestrates data loading, subruns, aggregation, and plotting.

    High-level flow:
      1) Load data and create a single train/test split for consistency.
      2) If multiple stove predicates are requested, create independent subruns
         (each: one stove predicate + microwave control) to avoid latent
         competition between stove concepts; optionally add an egg run.
      3) For each subrun: fit, match with capacities, perturb/train trials,
         rematch, and record stability metrics.
      4) Merge all subruns' records and generate CSVs and plots, including a
         publication-ready 2x4 grid (Jaccard row; prediction corr row).
    """
    args = parse_args()
    outdir = f"{args.output_dir}"
    setup_output_dir(outdir)

    # Load data
    biscuit_vars, sim_vars, var_names, sim_var_name_to_data = load_causality_data(
        data_dir=args.data_dir, split=args.split, max_files=args.max_files, use_cache=not args.no_cache
    )
    # Prepare matrices (X only once)
    X_all = biscuit_vars

    # Create a single, consistent train/test split based on the first requested predicate
    first_pred = args.predicates[0]
    cfg0 = get_predicate_configuration(first_pred)
    y0 = create_aggregate_predicate(sim_var_name_to_data, first_pred, cfg0['variables'])
    split = create_train_test_split(y_all=y0, test_size=args.test_fraction, random_state=args.split_seed, balance_classes=False, target_ratio=0.5, verbose=False)

    # Helper to run the full robustness pipeline for a given list of predicates (no competition outside that set)
    def _run_for_predicates(pred_list: List[str]) -> Tuple[List[dict], List[dict]]:
        # Build concepts and capacities for this subrun
        concept_names, K_assign = build_predicates(pred_list, sim_var_name_to_data)
        Y_all_local = np.vstack([sim_var_name_to_data[c] for c in concept_names]).astype(int)
        test_idx = split["test_idx"]; train_pool = split["train_pool_perm"]
        X_test = X_all[test_idx].T
        rows_local: List[dict] = []
        per_concept_rows_local: List[dict] = []

        # Determine primary predicate for balancing (prefer stove/egg over control)
        stove_preds = {"all_stoves_on", "at_least_two_stoves_on", "at_least_three_stoves_on"}
        control_name = "Microwave_d8b935e4_open"
        def to_concept_name(p: str) -> str:
            return p.replace('_', ' ').title().replace(' ', '_') if (p in stove_preds or p == "egg_intact") else p
        primary_pred = None
        for p in pred_list:
            if p in stove_preds or p == "egg_intact":
                primary_pred = p; break
        if primary_pred is None and len(pred_list) > 0:
            primary_pred = pred_list[0]
        primary_concept_name = to_concept_name(primary_pred)
        if primary_concept_name in concept_names:
            primary_idx = concept_names.index(primary_concept_name)
        else:
            primary_idx = 0  # fallback

        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            permuted_pool = rng.permutation(train_pool)
            Ns = [N for N in args.n_train_values if N <= len(permuted_pool)]

            # If balancing, precompute class-specific orders within the train pool
            if args.balance_classes:
                cls0 = [idx for idx in train_pool if Y_all_local[primary_idx, idx] == 0]
                cls1 = [idx for idx in train_pool if Y_all_local[primary_idx, idx] == 1]
                cls0_perm = rng.permutation(cls0)
                cls1_perm = rng.permutation(cls1)

            for N in Ns:
                if args.balance_classes:
                    k = min(N // 2, len(cls0_perm), len(cls1_perm))
                    N_eff = 2 * k
                    if N_eff == 0:
                        continue  # cannot form a balanced subset
                    subset = np.concatenate([cls0_perm[:k], cls1_perm[:k]])
                else:
                    subset = permuted_pool[:N]
                    N_eff = len(subset)
                X_train = X_all[subset].T
                Y_train = Y_all_local[:, subset]

                row_std_cache = X_train.std(axis=1, ddof=0)
                row_std_cache[row_std_cache == 0.0] = 1.0

                for model in args.models:
                    if model == "baseline":
                        assign_base = baseline_match_capacitated(X_train, Y_train, concept_names, K_assign)
                        Yhat_base = baseline_predict_assigned(X_train, Y_train, X_test, assign_base, concept_names)
                    elif model == "linear":
                        lin = FeaturePermutationEstimator(
                            regularizer='lasso', optim_kwargs={'alpha': args.alpha}, feature_transform=None,
                            d_variables=X_train.shape[0], n_features=1
                        )
                        Y_tr = Y_train.astype(float).copy()
                        const_rows = np.where(np.std(Y_tr, axis=1) == 0.0)[0]
                        if const_rows.size:
                            Y_tr[const_rows] += rng.normal(0.0, 1e-6, size=(const_rows.size, Y_tr.shape[1]))
                        Y_fit = Y_tr if Y_tr.shape[0] > 1 else np.vstack([Y_tr, Y_tr])
                        lin.fit(X_train, Y_fit)
                        beta = lin.beta_hat_all_
                        if Y_tr.shape[0] == 1:
                            beta = beta.reshape(2, -1)[:1, :]
                        b0 = _get_intercept(lin, beta.shape[0])
                        S = np.abs(beta)
                        assign_base = capacitated_match_multiple(S, concept_names, K_assign)
                        Yhat_base = predict_linear_assigned(lin, X_test, beta, b0, assign_base, concept_names)
                    elif model == "logistic":
                        logi = FeaturePermutationEstimator(
                            regularizer='logistic',
                            optim_kwargs={'penalty':'l1','solver':'liblinear','C':1.0,'class_weight':'balanced','max_iter':20000},
                            feature_transform=None, d_variables=X_train.shape[0], n_features=1
                        )
                        logi.fit(X_train, Y_train.astype(int))
                        beta = logi.beta_hat_all_
                        b0 = _get_intercept(logi, beta.shape[0])
                        S = np.abs(beta)
                        assign_base = capacitated_match_multiple(S, concept_names, K_assign)
                        Yhat_base = predict_logistic_assigned(logi, X_test, beta, b0, assign_base, concept_names)
                    else:
                        n_feat = min(args.nystrom_components, X_train.shape[1])
                        kern = KernelizedPermutationEstimator(
                            regularizer='lasso', optim_kwargs={'alpha': args.alpha}, kernel='rbf', parameter=args.kernel_gamma,
                            d_variables=X_train.shape[0], n_features=n_feat, groups=None, two_stage=None
                        )
                        Y_tr = Y_train.astype(float).copy()
                        const_rows = np.where(np.std(Y_tr, axis=1) == 0.0)[0]
                        if const_rows.size:
                            Y_tr[const_rows] += rng.normal(0.0, 1e-6, size=(const_rows.size, Y_tr.shape[1]))
                        Y_fit = Y_tr if Y_tr.shape[0] > 1 else np.vstack([Y_tr, Y_tr])
                        kern.fit(X_train, Y_fit)
                        beta = kern.beta_hat_all_
                        if Y_tr.shape[0] == 1:
                            beta = beta.reshape(2, -1)[:1, :]
                        b0 = _get_intercept(kern, beta.shape[0])
                        d_vars = X_train.shape[0]
                        B = beta.reshape(beta.shape[0], d_vars, n_feat)
                        S = np.linalg.norm(B, axis=2)
                        S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
                        assign_base = capacitated_match_multiple(S, concept_names, K_assign)
                        Yhat_base = predict_kernel_assigned(kern, X_test, beta, b0, assign_base, concept_names, d_vars=d_vars, n_feat=n_feat)

                    jacc_global_list = []
                    per_concept_jacc_list = {name: [] for name in concept_names}
                    per_concept_corr_list = {name: [] for name in concept_names}

                    for _ in range(args.n_trials):
                        Xp = perturb_train_all_dims(X_train, rng, sigma=args.sigma, row_std=row_std_cache)
                        if model == "baseline":
                            assign_noisy = baseline_match_capacitated(Xp, Y_train, concept_names, K_assign)
                            Yhat_noisy = baseline_predict_assigned(Xp, Y_train, X_test, assign_noisy, concept_names)
                        elif model == "linear":
                            lin_n = FeaturePermutationEstimator(
                                regularizer='lasso', optim_kwargs={'alpha': args.alpha}, feature_transform=None,
                                d_variables=Xp.shape[0], n_features=1
                            )
                            Y_tr = Y_train.astype(float).copy()
                            const_rows = np.where(np.std(Y_tr, axis=1) == 0.0)[0]
                            if const_rows.size:
                                Y_tr[const_rows] += rng.normal(0.0, 1e-6, size=(const_rows.size, Y_tr.shape[1]))
                            Y_fit = Y_tr if Y_tr.shape[0] > 1 else np.vstack([Y_tr, Y_tr])
                            lin_n.fit(Xp, Y_fit)
                            beta_n = lin_n.beta_hat_all_
                            if Y_tr.shape[0] == 1:
                                beta_n = beta_n.reshape(2, -1)[:1, :]
                            b0_n = _get_intercept(lin_n, beta_n.shape[0])
                            S_n = np.abs(beta_n)
                            assign_noisy = capacitated_match_multiple(S_n, concept_names, K_assign)
                            Yhat_noisy = predict_linear_assigned(lin_n, X_test, beta_n, b0_n, assign_noisy, concept_names)
                        elif model == "logistic":
                            logi_n = FeaturePermutationEstimator(
                                regularizer='logistic',
                                optim_kwargs={'penalty':'l1','solver':'liblinear','C':1.0,'class_weight':'balanced','max_iter':20000},
                                feature_transform=None, d_variables=Xp.shape[0], n_features=1
                            )
                            logi_n.fit(Xp, Y_train.astype(int))
                            beta_n = logi_n.beta_hat_all_
                            b0_n = _get_intercept(logi_n, beta_n.shape[0])
                            S_n = np.abs(beta_n)
                            assign_noisy = capacitated_match_multiple(S_n, concept_names, K_assign)
                            Yhat_noisy = predict_logistic_assigned(logi_n, X_test, beta_n, b0_n, assign_noisy, concept_names)
                        else:
                            n_feat = min(args.nystrom_components, Xp.shape[1])
                            kern_n = KernelizedPermutationEstimator(
                                regularizer='lasso', optim_kwargs={'alpha': args.alpha}, kernel='rbf', parameter=args.kernel_gamma,
                                d_variables=Xp.shape[0], n_features=n_feat, groups=None, two_stage=None
                            )
                            Y_tr = Y_train.astype(float).copy()
                            const_rows = np.where(np.std(Y_tr, axis=1) == 0.0)[0]
                            if const_rows.size:
                                Y_tr[const_rows] += rng.normal(0.0, 1e-6, size=(const_rows.size, Y_tr.shape[1]))
                            Y_fit = Y_tr if Y_tr.shape[0] > 1 else np.vstack([Y_tr, Y_tr])
                            kern_n.fit(Xp, Y_fit)
                            beta_n = kern_n.beta_hat_all_
                            if Y_tr.shape[0] == 1:
                                beta_n = beta_n.reshape(2, -1)[:1, :]
                            b0_n = _get_intercept(kern_n, beta_n.shape[0])
                            d_vars = Xp.shape[0]
                            Bn = beta_n.reshape(beta_n.shape[0], d_vars, n_feat)
                            S_n = np.linalg.norm(Bn, axis=2)
                            S_n = np.nan_to_num(S_n, nan=0.0, posinf=0.0, neginf=0.0)
                            assign_noisy = capacitated_match_multiple(S_n, concept_names, K_assign)
                            Yhat_noisy = predict_kernel_assigned(kern_n, X_test, beta_n, b0_n, assign_noisy, concept_names, d_vars=d_vars, n_feat=n_feat)

                        base_union = []
                        noisy_union = []
                        for nm in concept_names:
                            base_union.extend(assign_base.get(nm, []))
                            noisy_union.extend(assign_noisy.get(nm, []))
                        jacc_global = jaccard_set(base_union, noisy_union)
                        jacc_global_list.append(jacc_global)

                        for i, nm in enumerate(concept_names):
                            ja = jaccard_set(assign_base.get(nm, []), assign_noisy.get(nm, []))
                            per_concept_jacc_list[nm].append(ja)
                            co = corr_safe(Yhat_base[i], Yhat_noisy[i])
                            per_concept_corr_list[nm].append(co)

                    rows_local.append(dict(
                        seed=seed, n_train=int(N_eff), model=model,
                        jacc_global_mean=np.nanmean(jacc_global_list),
                        jacc_global_min=np.nanmin(jacc_global_list),
                        jacc_global_max=np.nanmax(jacc_global_list),
                    ))

                    if not args.skip_per_concept:
                        for nm in concept_names:
                            per_concept_rows_local.append(dict(
                                seed=seed, n_train=int(N_eff), model=model, concept=nm,
                                jaccard=np.nanmean(per_concept_jacc_list[nm]) if per_concept_jacc_list[nm] else np.nan,
                                pred_corr=np.nanmean(per_concept_corr_list[nm]) if per_concept_corr_list[nm] else np.nan,
                            ))

        return rows_local, per_concept_rows_local

    # Determine if we need to run stove predicates separately
    stove_predicates = {"all_stoves_on", "at_least_three_stoves_on", "at_least_two_stoves_on"}
    selected_stoves = [p for p in args.predicates if p in stove_predicates]
    rows: List[dict] = []
    per_concept_rows: List[dict] = []

    if len(selected_stoves) > 1:
        # Build subruns: each stove predicate + microwave_on; also run egg_intact (if requested) + microwave_on
        # Use the exact microwave variable name present in the dataset
        control = "Microwave_d8b935e4_open"
        requested_set = set(args.predicates)
        subruns: List[List[str]] = []
        for sp in selected_stoves:
            sub = [sp, control] if control not in requested_set else [sp, control]
            subruns.append(sub)
        if "egg_intact" in requested_set:
            subruns.append(["egg_intact", control])
        # Deduplicate runs
        unique_subruns = []
        seen = set()
        for r in subruns:
            key = tuple(sorted(r))
            if key not in seen:
                unique_subruns.append(r)
                seen.add(key)
        # Execute all subruns and merge
        for pr in unique_subruns:
            r_local, pc_local = _run_for_predicates(pr)
            rows.extend(r_local)
            per_concept_rows.extend(pc_local)
    else:
        # Single stove or none: run as originally requested (all together)
        r_local, pc_local = _run_for_predicates(args.predicates)
        rows.extend(r_local)
        per_concept_rows.extend(pc_local)


    # Save results
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "robust_groups_summary.csv"), index=False)
    if per_concept_rows:
        df_pc = pd.DataFrame(per_concept_rows)
        df_pc.to_csv(os.path.join(outdir, "robust_groups_perconcept.csv"), index=False)
    else:
        df_pc = pd.DataFrame([])

    # ---------- Plotting ----------
    import matplotlib.pyplot as plt
    models_order = ["baseline","linear","logistic","kernel_rbf"]
    label_map = {"baseline":"Baseline (cap-LR)", "linear":"Linear", "logistic":"Logistic", "kernel_rbf":"Kernel RBF"}

    # Global Jaccard
    agg = (df.groupby(['model','n_train'])
             .agg({'jacc_global_mean':['mean','min','max']})
             .reset_index())

    # Use the shared model color map from visualization utilities for consistency
    MODEL_COLOR_MAP = GLOBAL_MODEL_COLOR_MAP

    def _plot(ax, metric, title):
        for m in models_order:
            sub = agg[agg['model']==m].sort_values('n_train')
            if sub.empty: continue
            x = sub['n_train']; y = sub[(metric,'mean')]
            lo = sub[(metric,'min')]; hi = sub[(metric,'max')]
            color = MODEL_COLOR_MAP.get(m)
            ax.plot(x, y, marker='o', label=label_map.get(m,m), color=color)
            ax.fill_between(x, lo, hi, alpha=0.15, color=color)
        ax.set_xscale('log')
        xticks = sorted(df['n_train'].unique())
        ax.set_xticks(xticks); ax.set_xticklabels(xticks, rotation=45)
        ax.grid(True, which='both', linestyle='--', linewidth=0.6)
        ax.set_title(title); ax.set_xlabel('N_train')

    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.6))
    _plot(ax, 'jacc_global_mean', 'Global Jaccard (↑)')
    ax.set_ylabel('Jaccard')
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "robust_groups_global_jaccard.png"), dpi=args.dpi, bbox_inches='tight')
    fig.savefig(os.path.join(outdir, "robust_groups_global_jaccard.pdf"), bbox_inches='tight')

    # Per-concept grids
    if not args.skip_per_concept and not df_pc.empty:
        # Per-concept Jaccard
        concepts_sorted = sorted(df_pc['concept'].unique().tolist())
        n_concepts = len(concepts_sorted)
        n_cols = min(5, n_concepts) if n_concepts > 0 else 1
        n_rows = int(np.ceil(n_concepts / n_cols)) if n_concepts > 0 else 1
        figJ, axesJ = plt.subplots(n_rows, n_cols, figsize=(3.4*n_cols, 2.8*n_rows), squeeze=False)
        for idx, concept in enumerate(concepts_sorted):
            r, c = divmod(idx, n_cols)
            axJ = axesJ[r][c]
            subc = (df_pc[df_pc['concept']==concept]
                      .groupby(['model','n_train'])
                      .agg(jacc=('jaccard','mean'))
                      .reset_index())
            for m in models_order:
                subm = subc[subc['model']==m].sort_values('n_train')
                if subm.empty: continue
                axJ.plot(subm['n_train'], subm['jacc'], marker='o', label=label_map.get(m,m), color=MODEL_COLOR_MAP.get(m))
            axJ.set_xscale('log')
            xticks = sorted(df['n_train'].unique())
            axJ.set_xticks(xticks); axJ.set_xticklabels(xticks, rotation=45, fontsize=7)
            axJ.grid(True, which='both', linestyle='--', linewidth=0.6)
            axJ.set_title(concept, fontsize=9)
        total_axes = n_rows * n_cols
        for j in range(n_concepts, total_axes):
            r, c = divmod(j, n_cols)
            axesJ[r][c].axis('off')
        axesJ[0][0].legend(fontsize=7)
        plt.tight_layout()
        figJ.savefig(os.path.join(outdir, "robust_groups_perconcept_jaccard.png"), dpi=args.dpi, bbox_inches='tight')
        figJ.savefig(os.path.join(outdir, "robust_groups_perconcept_jaccard.pdf"), bbox_inches='tight')

        # Per-concept prediction correlation
        figC, axesC = plt.subplots(n_rows, n_cols, figsize=(3.4*n_cols, 2.8*n_rows), squeeze=False)
        for idx, concept in enumerate(concepts_sorted):
            r, c = divmod(idx, n_cols)
            axC = axesC[r][c]
            subc = (df_pc[df_pc['concept']==concept]
                      .groupby(['model','n_train'])
                      .agg(corr=('pred_corr','mean'))
                      .reset_index())
            for m in models_order:
                subm = subc[subc['model']==m].sort_values('n_train')
                if subm.empty: continue
                axC.plot(subm['n_train'], subm['corr'], marker='o', label=label_map.get(m,m), color=MODEL_COLOR_MAP.get(m))
            axC.set_xscale('log')
            xticks = sorted(df['n_train'].unique())
            axC.set_xticks(xticks); axC.set_xticklabels(xticks, rotation=45, fontsize=7)
            axC.grid(True, which='both', linestyle='--', linewidth=0.6)
            axC.set_title(concept, fontsize=9)
        total_axes = n_rows * n_cols
        for j in range(n_concepts, total_axes):
            r, c = divmod(j, n_cols)
            axesC[r][c].axis('off')
        axesC[0][0].legend(fontsize=7)
        plt.tight_layout()
        figC.savefig(os.path.join(outdir, "robust_groups_perconcept_corr.png"), dpi=args.dpi, bbox_inches='tight')
        figC.savefig(os.path.join(outdir, "robust_groups_perconcept_corr.pdf"), bbox_inches='tight')

        # ---------------- Publication-ready 2x4 grid (Row 1: Jaccard, Row 2: Prediction corr) ----------------
        # Target columns (in order): All Stoves On, At Least 3, At Least 2, Egg Intact
        concept_keys = [
            "All_Stoves_On",
            "At_Least_Three_Stoves_On",
            "At_Least_Two_Stoves_On",
            "Egg_Intact",
        ]
        concept_titles = [
            "All Stoves On",
            "At Least 3",
            "At Least 2",
            "Egg Intact",
        ]

        # Filter to available concepts only, preserve requested order
        available = [ck for ck in concept_keys if ck in df_pc['concept'].unique().tolist()]
        available_titles = [t for ck, t in zip(concept_keys, concept_titles) if ck in available]
        if available:
            import matplotlib as mpl
            import matplotlib.pyplot as plt

            # Style
            mpl.rcParams.update({
                "font.size": 10,
                "axes.labelsize": 10,
                "axes.titlesize": 11,
                "legend.fontsize": 9,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "axes.linewidth": 0.8,
                "lines.linewidth": 1.8,
                "grid.linewidth": 0.5,
            })

            # Aggregate across seeds for bands (mean/min/max over seeds per model/N/concept)
            agg_j = (df_pc.groupby(["concept","model","n_train"])
                           .agg(mean_j=("jaccard","mean"),
                                min_j=("jaccard","min"),
                                max_j=("jaccard","max"))
                           .reset_index())
            agg_c = (df_pc.groupby(["concept","model","n_train"])
                           .agg(mean_c=("pred_corr","mean"),
                                min_c=("pred_corr","min"),
                                max_c=("pred_corr","max"))
                           .reset_index())

            xticks = sorted(df['n_train'].unique())

            figG, axesG = plt.subplots(2, 4, figsize=(3.2*4, 2.6*2), sharex=True)
            # Row 0: Jaccard, Row 1: Corr
            y_ticks_j = np.arange(0.0, 1.01, 0.2)
            y_ticks_c = np.arange(-0.0, 1.01, 0.2)

            handles_legend = []
            labels_legend = []

            for col, (ck, title) in enumerate(zip(available, available_titles)):
                # Top row: Jaccard
                axJ = axesG[0, col]
                axJ.set_xscale('log')
                axJ.set_ylim(0.0, 1.0)
                axJ.set_yticks(y_ticks_j)
                axJ.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.8)
                axJ.set_title(title)

                subJ = agg_j[agg_j['concept'] == ck]
                for m in models_order:
                    subm = subJ[subJ['model'] == m].sort_values('n_train')
                    if subm.empty:
                        continue
                    color = MODEL_COLOR_MAP.get(m, None)
                    (ln,) = axJ.plot(subm['n_train'], subm['mean_j'], marker='o', color=color, label=label_map.get(m, m))
                    axJ.fill_between(subm['n_train'], subm['min_j'], subm['max_j'], alpha=0.15, color=color)
                    if col == 0:
                        handles_legend.append(ln)
                        labels_legend.append(label_map.get(m, m))

                # Bottom row: Prediction correlation
                axC = axesG[1, col]
                axC.set_xscale('log')
                axC.set_ylim(-0.1, 1.0)
                axC.set_yticks(y_ticks_c)
                axC.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.8)

                subC = agg_c[agg_c['concept'] == ck]
                for m in models_order:
                    subm = subC[subC['model'] == m].sort_values('n_train')
                    if subm.empty:
                        continue
                    color = MODEL_COLOR_MAP.get(m, None)
                    axC.plot(subm['n_train'], subm['mean_c'], marker='o', color=color, label=label_map.get(m, m))
                    axC.fill_between(subm['n_train'], subm['min_c'], subm['max_c'], alpha=0.15, color=color)

            # Labels and ticks
            for r in range(2):
                for c in range(4):
                    ax = axesG[r, c]
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([str(v) for v in xticks], rotation=45)
                    if c != 0:
                        ax.tick_params(axis='y', labelleft=False)
            axesG[0,0].set_ylabel('Jaccard')
            axesG[1,0].set_ylabel('Prediction corr')
            for c in range(4):
                axesG[1,c].set_xlabel('N_train')

            # Legend
            figG.legend(
                handles_legend,
                labels_legend,
                loc='upper center',
                ncol=len(labels_legend),
                bbox_to_anchor=(0.5, 1.08),
                frameon=True,
                fontsize=13,
                markerscale=1.2,
                handlelength=2.4,
                handletextpad=0.7,
                columnspacing=1.2,
                borderpad=0.5,
            )
            plt.tight_layout(rect=[0, 0.0, 1, 0.98])

            figG.savefig(os.path.join(outdir, "robust_groups_grid_2x4.png"), dpi=args.dpi, bbox_inches='tight')
            figG.savefig(os.path.join(outdir, "robust_groups_grid_2x4.pdf"), bbox_inches='tight')

    print(f"Saved results and plots to {outdir}")


if __name__ == "__main__":
    main()


