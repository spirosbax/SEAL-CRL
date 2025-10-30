"""
End-to-end robustness for ALL binary concepts (no assigned/unassigned split).
We perturb ALL latent dimensions in the train set, re-fit + re-match, then
measure:
  - Assignment stability:
      * jaccard_global (↑): Jaccard between the set of all matched latents (global)
  - Prediction stability:
      * pred_corr_mean (↑): mean Pearson corr across concepts between base test
        predictions and noisy-refit test predictions (computed on clean test X)

Models supported:
  - Baseline (Spearman + Hungarian + LR)
  - Linear (FeaturePermutationEstimator, lasso)
  - Logistic (FeaturePermutationEstimator, sklearn LogisticRegression)
  - Kernel RBF (KernelizedPermutationEstimator, Nyström RBF)
"""

import os, sys, argparse, numpy as np, pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List

from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Repo imports
from causality.data.ithor_loader import load_causality_data
from causality.utils.model_evaluation import create_concept_order, create_data_split, r2_safe
from causality.utils.utils import setup_output_dir
from causality.utils.visualization import MODEL_COLOR_MAP

# External permutation estimators
from permutation_estimator.estimator import (
    FeaturePermutationEstimator,
    KernelizedPermutationEstimator,
)

# ------------------------- Helpers -------------------------

def add_row_jitter(A: np.ndarray, rng: np.random.Generator, noise: float = 1e-6) -> np.ndarray:
    """Add tiny noise to rows with zero variance to avoid degeneracies."""
    A2 = A.copy()
    row_std = A2.std(axis=1, ddof=0)
    idx = np.where(row_std == 0.0)[0]
    if idx.size:
        A2[idx] += rng.normal(0.0, noise, size=(idx.size, A2.shape[1]))
    return A2

def standardize_xy(X_train, X_test, Y_train, Y_test):
    """Row-wise standardization for X and Y (for linear/kernel pipelines)."""
    sc_x = StandardScaler()
    X_train_std = sc_x.fit_transform(X_train.T).T
    X_test_std  = sc_x.transform(X_test.T).T
    sc_y = StandardScaler()
    Y_train_std = sc_y.fit_transform(Y_train.T).T
    Y_test_std  = sc_y.transform(Y_test.T).T
    return X_train_std, X_test_std, Y_train_std, Y_test_std

def jaccard_set(a: np.ndarray, b: np.ndarray) -> float:
    A = set(int(x) for x in a if x >= 0)
    B = set(int(x) for x in b if x >= 0)
    if not A and not B: return 1.0
    return len(A & B) / max(1, len(A | B))

def corr_rowwise(Y0: np.ndarray, Y1: np.ndarray) -> np.ndarray:
    """
    Row-wise Pearson corr; returns (n_concepts,) with NaN if degenerate.
    Y0, Y1: (n_concepts, N_test)
    """
    out = np.full(Y0.shape[0], np.nan, dtype=float)
    for i in range(Y0.shape[0]):
        if np.std(Y0[i]) == 0 or np.std(Y1[i]) == 0:
            # Convention: if both flat and equal → corr 1; else 0
            out[i] = 1.0 if np.allclose(Y0[i], Y1[i]) else 0.0
        else:
            c = np.corrcoef(Y0[i], Y1[i])[0, 1]
            out[i] = 0.0 if np.isnan(c) else float(c)
    return out

def get_assign_from_baseline(X_train_std_NF, Y_train, n_dims) -> np.ndarray:
    """
    Build Spearman corr on standardized X, Hungarian on |rho|.
    Returns perm (n_concepts,) matched latent indices.
    """
    from scipy.stats import spearmanr
    m = Y_train.shape[0]
    C = np.zeros((m, n_dims))
    for i in range(m):
        y = Y_train[i]
        if np.unique(y).size < 2:
            continue
        for j in range(n_dims):
            rho, _ = spearmanr(y, X_train_std_NF[:, j])
            C[i, j] = 0.0 if np.isnan(rho) else rho
    row_ind, col_ind = linear_sum_assignment(-np.abs(C))
    perm = np.full(C.shape[0], -1, dtype=int)
    perm[row_ind] = col_ind
    return perm

def predict_baseline_lr(X_train_std_NF, X_test_std_NF, Y_train, perm) -> np.ndarray:
    """
    For each concept i, fit 1D LinearRegression on matched column perm[i]; predict test.
    Returns (n_concepts, N_test). If a concept has constant train labels → NaN row.
    """
    m = Y_train.shape[0]
    Yhat = np.full((m, X_test_std_NF.shape[0]), np.nan, dtype=float)
    for i in range(m):
        if np.unique(Y_train[i]).size < 2:
            continue
        j = int(perm[i])
        lm = LinearRegression().fit(X_train_std_NF[:, [j]], Y_train[i])
        Yhat[i] = lm.predict(X_test_std_NF[:, [j]])
    return Yhat

def fit_predict_baseline(X_train, Y_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    """
    Baseline pipeline: standardize X, Spearman+Hungarian → perm, then 1D LR per concept.
    Returns (Yhat, perm) with shapes ((m, N_test), (m,))
    """
    scaler = StandardScaler().fit(X_train.T)
    Xtr = scaler.transform(X_train.T)  # (N, d)
    Xte = scaler.transform(X_test.T)   # (N_test, d)

    perm = get_assign_from_baseline(Xtr, Y_train, n_dims=X_train.shape[0])
    Yhat = predict_baseline_lr(Xtr, Xte, Y_train, perm)
    return Yhat, perm

def fit_predict_linear(X_train, Y_train, X_test, alpha, rng) -> Tuple[np.ndarray, np.ndarray]:
    """
    FeaturePermutationEstimator (lasso). Uses internal z-scoring via .fit_transform.
    Returns (Yhat_match, perm_hat_match).
    """
    est = FeaturePermutationEstimator(
        regularizer='lasso',
        optim_kwargs={'alpha': alpha},
        feature_transform=None,
        d_variables=X_train.shape[0],
        n_features=1
    )
    # tiny jitter to Y helps small-N constants (regression-style target)
    Y_tr = Y_train.astype(float).copy()
    const_rows = np.where(np.std(Y_tr, axis=1) == 0.0)[0]
    if const_rows.size: # Add tiny jitter to Y to help small-N constants
        Y_tr[const_rows] += rng.normal(0.0, 1e-6, size=(const_rows.size, Y_tr.shape[1]))
    est.fit(X_train, Y_tr)
    return est.predict_match(X_test), est.perm_hat_match_.astype(int)

def fit_predict_logistic(X_train, Y_train, X_test, rng) -> Tuple[np.ndarray, np.ndarray]:
    """
    Logistic permutation estimator using sklearn LogisticRegression internally.
    Uses internal standardization via fit/transform; returns logits via predict_match.
    """
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
        d_variables=X_train.shape[0],
        n_features=1
    )
    # Keep labels binary for logistic; no jitter
    Y_tr = Y_train.astype(int, copy=True)
    est.fit(X_train, Y_tr)
    return est.predict_match(X_test), est.perm_hat_match_.astype(int)

def fit_predict_kernel(X_train, Y_train, X_test, alpha, gamma, rng, max_nystrom: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    KernelizedPermutationEstimator (lasso + Nyström RBF). Internally handles feature maps.
    """
    n_feat = min(max_nystrom, X_train.shape[1])  # cap by N and CLI
    est = KernelizedPermutationEstimator(
        regularizer='lasso',
        optim_kwargs={'alpha': alpha},
        kernel='rbf',
        parameter=gamma,
        d_variables=X_train.shape[0],
        n_features=n_feat,
        groups=None,
        two_stage=None
    )
    # tiny jitter to Y helps small-N constants (regression-style target)
    Y_tr = Y_train.astype(float).copy()
    const_rows = np.where(np.std(Y_tr, axis=1) == 0.0)[0]
    if const_rows.size: # Add tiny jitter to Y to help small-N constants
        Y_tr[const_rows] += rng.normal(0.0, 1e-6, size=(const_rows.size, Y_tr.shape[1]))
    est.fit(X_train, Y_tr)
    return est.predict_match(X_test), est.perm_hat_match_.astype(int)

def perturb_train_all_dims(
    X_train: np.ndarray,
    rng: np.random.Generator,
    sigma: float,
    row_std: np.ndarray = None,
) -> np.ndarray:
    """
    Add Gaussian noise to ALL latent rows, scaled by each row's std on the current train set.
    If row_std is provided, reuse it to avoid recomputing per trial.
    """
    if row_std is None:
        row_std = X_train.std(axis=1, ddof=0)
        row_std = row_std.copy()
        row_std[row_std == 0.0] = 1.0
    noise = rng.normal(0.0, sigma * row_std[:, None], size=X_train.shape)
    return X_train + noise


# ------------------------- Main experiment -------------------------

def parse_args():
    p = argparse.ArgumentParser("All-binary end-to-end robustness")
    p.add_argument("--data_dir", type=str, default="src/data/ithor")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--output_dir", type=str, default="results/robustness/binary")
    p.add_argument("--no_cache", action="store_true")

    p.add_argument("--alpha", type=float, default=2.154e-3)
    p.add_argument("--kernel_gamma", type=float, default=0.5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--n_train_values", type=int, nargs="+",
                   default=[5, 10, 20, 40, 80, 160, 320, 640, 1280])
    p.add_argument("--test_fraction", type=float, default=0.2)
    p.add_argument("--split_seed", type=int, default=42)

    p.add_argument("--models", type=str, nargs="+",
                   default=["baseline", "linear", "logistic", "kernel_rbf"],
                   choices=["baseline","linear","logistic","kernel_rbf"])

    # Robustness knobs
    p.add_argument("--n_trials", type=int, default=5)
    p.add_argument("--sigma", type=float, default=0.5, help="Noise scale (in train-space std units)")

    # Plot
    p.add_argument("--dpi", type=int, default=600)

    # Performance toggles
    p.add_argument("--skip_per_concept", action="store_true", help="Skip per-concept corr aggregation/plot to speed up")
    p.add_argument("--nystrom_components", type=int, default=64, help="Max Nyström components per latent for kernel model")

    return p.parse_args()


def main():
    args = parse_args()
    outdir = f"{args.output_dir}"
    setup_output_dir(outdir)

    # ---- Load data ----
    biscuit_vars, sim_vars, var_names, sim_var_name_to_data = load_causality_data(
        data_dir=args.data_dir, split=args.split, max_files=args.max_files,
        use_cache=not args.no_cache
    )

    # Concepts: ALL binary
    concepts = create_concept_order("binary")
    concepts = [c for c in concepts if c in sim_var_name_to_data]
    X_all = biscuit_vars  # (N_total, d_vars)
    Y_all = np.vstack([sim_var_name_to_data[c] for c in concepts]).astype(int)  # (m, N_total)

    # Split
    split = create_data_split(X_all.shape[0], test_fraction=args.test_fraction, random_seed=args.split_seed)
    test_idx = split["test_idx"]; train_pool = split["train_pool_perm"]

    # Fixed test
    X_test = X_all[test_idx].T
    Y_test = Y_all[:, test_idx]

    # Records
    rows = []
    per_concept_corr = []  # optionally keep per-concept correlations

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        permuted_pool = rng.permutation(train_pool)
        Ns = [N for N in args.n_train_values if N <= len(permuted_pool)]

        for N in Ns:
            subset = permuted_pool[:N]
            X_train = X_all[subset].T        # (d, N)
            Y_train = Y_all[:, subset]       # (m, N)

            for model in args.models:
                # 1) Base fit + match + predict
                if model == "baseline":
                    Yhat_base, assign_base = fit_predict_baseline(X_train, Y_train, X_test)
                elif model == "linear":
                    Yhat_base, assign_base = fit_predict_linear(X_train, Y_train, X_test, args.alpha, rng)
                elif model == "logistic":
                    Yhat_base, assign_base = fit_predict_logistic(X_train, Y_train, X_test, rng)
                else:  # kernel_rbf
                    Yhat_base, assign_base = fit_predict_kernel(X_train, Y_train, X_test, args.alpha, args.kernel_gamma, rng, args.nystrom_components)

                # Some concepts may have NaN predictions for baseline at tiny N; handle later in corr.

                # 2) Trials: perturb ALL train dims → refit+rematch → repredict
                jacc_global_list, corr_mean_list = [], []
                for t in range(args.n_trials):
                    # Precompute row std once per (seed,N) for speed
                    if t == 0:
                        row_std_cache = X_train.std(axis=1, ddof=0)
                        row_std_cache[row_std_cache == 0.0] = 1.0
                    Xp = perturb_train_all_dims(X_train, rng, sigma=args.sigma, row_std=row_std_cache)
                    if model == "baseline":
                        Yhat_noisy, assign_noisy = fit_predict_baseline(Xp, Y_train, X_test)
                    elif model == "linear":
                        Yhat_noisy, assign_noisy = fit_predict_linear(Xp, Y_train, X_test, args.alpha, rng)
                    elif model == "logistic":
                        Yhat_noisy, assign_noisy = fit_predict_logistic(Xp, Y_train, X_test, rng)
                    else:
                        Yhat_noisy, assign_noisy = fit_predict_kernel(Xp, Y_train, X_test, args.alpha, args.kernel_gamma, rng, args.nystrom_components)

                    # Assignment metrics
                    jac_global = jaccard_set(assign_base, assign_noisy)

                    # Prediction stability (per-concept corr → mean)
                    corr_vec = corr_rowwise(Yhat_base, Yhat_noisy)
                    corr_mean = float(np.nanmean(corr_vec))

                    jacc_global_list.append(jac_global)
                    corr_mean_list.append(corr_mean)

                    # optionally store per-concept correlations for later analysis
                    if not args.skip_per_concept:
                        for i, c in enumerate(concepts):
                            per_concept_corr.append(dict(
                                seed=seed, n_train=N, model=model, concept=c,
                                trial=t, corr=corr_vec[i]
                            ))

                # Aggregate over trials for this (seed, N, model)
                rows.append(dict(
                    seed=seed, n_train=N, model=model,
                    jacc_global_mean=np.nanmean(jacc_global_list),
                    jacc_global_min=np.nanmin(jacc_global_list),
                    jacc_global_max=np.nanmax(jacc_global_list),
                    pred_corr_mean=np.nanmean(corr_mean_list),
                    pred_corr_min=np.nanmin(corr_mean_list),
                    pred_corr_max=np.nanmax(corr_mean_list),
                ))

    # Save results
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "robust_all_binary_summary.csv"), index=False)
    if per_concept_corr:
        pd.DataFrame(per_concept_corr).to_csv(os.path.join(outdir, "robust_all_binary_perconcept_corr.csv"), index=False)

    # --------- Plotting (summary: 2 panels) ----------
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # Consistent model order and labels
    models_order = ["baseline","linear","logistic","kernel_rbf"]
    label_map = {"baseline":"Baseline (Hungarian LR)",
                 "linear":"Linear",
                 "logistic":"Logistic",
                 "kernel_rbf":"Kernel RBF"}

    agg = (df.groupby(['model','n_train'])
             .agg({'jacc_global_mean':['mean','min','max'],
                   'pred_corr_mean':['mean','min','max']})
             .reset_index())

    # Publication-ready matplotlib defaults
    mpl.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 11,
        'legend.fontsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.8,
        'grid.linewidth': 0.5,
    })

    def _plot(ax, metric, title):
        for m in models_order:
            sub = agg[agg['model']==m].sort_values('n_train')
            if sub.empty: continue
            x = sub['n_train']; y = sub[(metric,'mean')]
            lo = sub[(metric,'min')]; hi = sub[(metric,'max')]
            color = MODEL_COLOR_MAP.get(m)
            ax.plot(x, y, marker='o', markersize=4.5, label=label_map.get(m,m), color=color)
            ax.fill_between(x, lo, hi, alpha=0.18, color=color)
        ax.set_xscale('log')
        xticks = sorted(df['n_train'].unique())
        ax.set_xticks(xticks); ax.set_xticklabels([str(int(v)) for v in xticks])
        # y-limits: allow slight negative for correlations; keep 0..1 for Jaccard
        if 'corr' in metric:
            ax.set_ylim(-0.1, 1.05)
        else:
            ax.set_ylim(0.0, 1.05)
        ax.set_yticks(np.arange(0.0, 1.01, 0.2))
        ax.set_facecolor('white')
        ax.grid(True, which='both', linestyle='--', color='lightgray', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Training samples N')

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    _plot(axes[0], 'jacc_global_mean', 'Assignment stability: global Jaccard (↑)')
    axes[0].set_ylabel('Jaccard')
    _plot(axes[1], 'pred_corr_mean',   'Prediction stability: mean corr (↑)')
    axes[1].set_ylabel('Correlation')

    # Single, top-centered legend using consistent colors/labels
    legend_handles = [
        Line2D([0], [0], color=MODEL_COLOR_MAP[m], marker='o', linestyle='-', markersize=4.5, label=label_map[m])
        for m in models_order if m in agg['model'].unique()
    ]
    fig.legend(
        handles=legend_handles,
        labels=[h.get_label() for h in legend_handles],
        loc='upper center', ncol=len(legend_handles), bbox_to_anchor=(0.5, 1.02),
        frameon=True, fancybox=True, borderpad=0.4, handlelength=2.0, columnspacing=1.0, handletextpad=0.6
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(outdir, "robust_all_binary_curves.png"), dpi=args.dpi, bbox_inches='tight')
    fig.savefig(os.path.join(outdir, "robust_all_binary_curves.pdf"), bbox_inches='tight')
    print(f"Saved curves to {outdir}")

    # --------- Per-concept prediction correlation subplots ----------
    per_df = pd.DataFrame(per_concept_corr)
    if not args.skip_per_concept and not per_df.empty:
        aggr_per = (per_df.groupby(['concept','model','n_train'])
                          .agg(mean=('corr','mean'), min=('corr','min'), max=('corr','max'))
                          .reset_index())

        concept_list = sorted(per_df['concept'].unique().tolist())
        n_concepts = len(concept_list)
        n_cols = 4 if n_concepts > 0 else 1
        n_rows = int(np.ceil(n_concepts / n_cols)) if n_concepts > 0 else 1

        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(3.4*n_cols, 2.8*n_rows), squeeze=False)
        for idx, concept in enumerate(concept_list):
            r, c = divmod(idx, n_cols)
            ax = axes2[r][c]
            for m in models_order:
                sub = aggr_per[(aggr_per['concept']==concept) & (aggr_per['model']==m)].sort_values('n_train')
                if sub.empty:
                    continue
                x = sub['n_train']
                y = sub['mean']
                lo = sub['min']
                hi = sub['max']
                color = MODEL_COLOR_MAP.get(m)
                ax.plot(x, y, marker='o', markersize=4.0, label=label_map.get(m,m), color=color)
                ax.fill_between(x, lo, hi, alpha=0.18, color=color)
            ax.set_xscale('log')
            xticks = sorted(df['n_train'].unique())
            ax.set_xticks(xticks); ax.set_xticklabels([str(int(v)) for v in xticks], fontsize=7)
            # Allow slight negative for correlations per concept as well
            ax.set_ylim(-0.1, 1.05)
            ax.set_yticks(np.arange(0.0, 1.01, 0.2))
            ax.grid(True, which='both', linestyle='--', color='lightgray', alpha=0.7)
            ax.set_title(concept, fontsize=9)
        # Hide any unused subplots
        total_axes = n_rows * n_cols
        for j in range(n_concepts, total_axes):
            r, c = divmod(j, n_cols)
            axes2[r][c].axis('off')
        # Put a single legend (top center)
        legend_handles2 = [
            Line2D([0], [0], color=MODEL_COLOR_MAP[m], marker='o', linestyle='-', markersize=4.0, label=label_map[m])
            for m in models_order if m in aggr_per['model'].unique()
        ]
        fig2.legend(
            handles=legend_handles2,
            labels=[h.get_label() for h in legend_handles2],
            loc='upper center', ncol=len(legend_handles2), bbox_to_anchor=(0.5, 1.02),
            frameon=True, fancybox=True, borderpad=0.4, handlelength=2.0, columnspacing=1.0, handletextpad=0.6
        )
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        fig2.savefig(os.path.join(outdir, "robust_all_binary_perconcept_corr_grid.png"), dpi=args.dpi, bbox_inches='tight')
        fig2.savefig(os.path.join(outdir, "robust_all_binary_perconcept_corr_grid.pdf"), bbox_inches='tight')

if __name__ == "__main__":
    main()