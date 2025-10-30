"""
Plotting utilities for R² analysis and other visualizations.
"""

import os
import re
from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from causality.utils.utils import pretty_concept_name


def plot_r2_matrix(r2_matrix: np.ndarray, 
                   var_names: List[str], 
                   output_path: str,
                   fig_width: int = 12,
                   max_height: int = 14) -> None:
    """
    Plot R² correlation matrix as heatmap.
    
    Args:
        r2_matrix: R² matrix with shape (n_vars, n_dims)
        var_names: List of variable names
        output_path: Path to save the plot
        fig_width: Figure width
        max_height: Maximum figure height
    """
    n_vars, n_dims = r2_matrix.shape
    
    # Create pretty variable names
    display_vars = [pretty_concept_name(v) for v in var_names]
    
    # Scale height to number of concepts
    fig_h = max(5, min(0.35 * n_vars + 2, max_height))
    
    # Set theme and create figure
    sns.set_theme(style="white")
    plt.figure(figsize=(fig_width, fig_h))
    
    ax = sns.heatmap(
        r2_matrix,
        yticklabels=display_vars,           # concepts (prettified) on Y
        xticklabels=list(range(n_dims)),    # latent dims on X
        cmap="rocket",
        vmin=0, vmax=1,
        annot=False,                        # hide per-cell text
        cbar=True,
        linewidths=0                        # remove grid lines
    )
    
    # Titles & labels (bold / larger)
    plt.ylabel('Causal Variable', fontsize=14, fontweight='bold')
    plt.xlabel('Latent Dimension', fontsize=14, fontweight='bold')
    
    # Tick formatting
    plt.xticks(rotation=0, ha='center', fontsize=12, fontweight='bold')
    plt.yticks(rotation=0, fontsize=12, fontweight='bold')
    
    # Colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("R² value", fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()  # Close the figure to free memory


def plot_latents_vs_binary_vertical(
    Z: np.ndarray,
    vars_df: Union[pd.DataFrame, dict],
    var_name: str,
    latent_dims: Optional[List[int]] = None,
    jitter: float = 0.06,
    point_size: float = 10,
    height_per_plot: float = 2.6,
    subsample: Optional[int] = None,
    suptitle: bool = True,
    same_xlim: bool = True,
    xlim_percentiles: Tuple[float, float] = (1, 99),
    output_path: Optional[str] = None,
    dpi: int = 600,
    random_seed: int = 0,
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create vertical stack of clean scatter plots showing latent dimensions vs binary variables.
    
    This function creates publication-ready scatter plots with:
    - x-axis: latent dimension values
    - y-axis: binary variable (0/1) with jitter for visibility
    - Reference lines at y=0 and y=1
    - Consistent styling and optional x-axis limits
    
    Args:
        Z: Latent variables array of shape (N, D) where N is samples, D is dimensions
        vars_df: DataFrame or dict containing binary variables data
        var_name: Name of the binary variable to plot
        latent_dims: List of latent dimensions to plot (default: all dimensions)
        jitter: Amount of vertical jitter to add for visualization (default: 0.06)
        point_size: Size of scatter plot points (default: 10)
        height_per_plot: Height of each subplot in inches (default: 2.6)
        subsample: Optional number of samples to randomly subsample for faster plotting
        suptitle: Whether to add a figure title (default: True)
        same_xlim: Whether to use consistent x-limits across all subplots (default: True)
        xlim_percentiles: Percentile range for robust x-limits when same_xlim=True
        output_path: Optional path to save the figure
        dpi: DPI for saved figure (default: 600)
        random_seed: Random seed for reproducible jitter and subsampling (default: 0)
        
    Returns:
        Tuple of (figure, list of axes) for further customization
        
    Raises:
        AssertionError: If data shapes don't match or variable is not binary
        KeyError: If var_name is not found in vars_df
    """
    # ---- Data preparation ----
    if latent_dims is None:
        latent_dims = list(range(Z.shape[1]))
    
    X = np.asarray(Z)
    
    # Handle both DataFrame and dict inputs
    if isinstance(vars_df, dict):
        if var_name not in vars_df:
            raise KeyError(f"Variable '{var_name}' not found in vars_df")
        y = np.asarray(vars_df[var_name]).astype(int).ravel()
    else:  # pandas DataFrame
        if var_name not in vars_df.columns:
            raise KeyError(f"Variable '{var_name}' not found in vars_df columns")
        y = np.asarray(vars_df[var_name]).astype(int).ravel()
    
    assert X.shape[0] == y.shape[0], f"Z and vars_df lengths must match: {X.shape[0]} vs {y.shape[0]}"
    assert set(np.unique(y)).issubset({0, 1}), f"{var_name} must be binary (0/1), found: {np.unique(y)}"

    # Optional subsampling (consistent across latents)
    if subsample is not None and subsample < len(y):
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(y), size=subsample, replace=False)
        X = X[idx]
        y = y[idx]

    # Add jitter once so it's consistent across subplots
    rng = np.random.default_rng(random_seed)
    y_jitter = y + rng.uniform(-jitter, +jitter, size=len(y))

    # Compute consistent x-limits across panels (robust to outliers)
    if same_xlim:
        xs = np.concatenate([X[:, d] for d in latent_dims], axis=0)
        x_lo, x_hi = np.percentile(xs, xlim_percentiles)

    # ---- Styling configuration ----
    rc_params = {
        "font.size": 10,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 0.8,
    }

    with mpl.rc_context(rc_params):
        n = len(latent_dims)
        fig, axes = plt.subplots(n, 1, figsize=(7.0, height_per_plot * n), sharey=True)
        if n == 1:
            axes = [axes]

        # ---- Create plots ----
        for i, (ax, d) in enumerate(zip(axes, latent_dims)):
            x = X[:, d].ravel()
            
            # Main scatter plot
            ax.scatter(x, y_jitter, s=point_size, alpha=0.7, edgecolors='none')

            # Reference lines at y=0 and y=1
            ax.axhline(0, color="lightgray", lw=0.8, linestyle='--', zorder=0)
            ax.axhline(1, color="lightgray", lw=0.8, linestyle='--', zorder=0)

            # Configure ticks and labels
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["0", "1"])
            
            # Only show x-labels on bottom plot
            if i < n - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel("Latent value", fontweight='bold')

            # Set x-limits if requested
            if same_xlim:
                ax.set_xlim(x_lo, x_hi)

            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Panel title
            ax.set_title(f"latent[{d}]", pad=4)

        # Add y-label to first (top) axis
        axes[0].set_ylabel(pretty_concept_name(var_name), fontweight='bold')

        # Optional figure title
        if suptitle:
            fig.suptitle(
                f"Latents vs {pretty_concept_name(var_name)}", 
                fontsize=14, 
                fontweight='bold', 
                y=0.995
            )

        fig.tight_layout(rect=[0, 0, 1, 0.98])

        # Save figure if path provided
        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"Latent vs binary plot saved to: {output_path}")

        return fig, axes


def plot_baseline_comparison(
    summary: pd.DataFrame,
    target_n_trains: List[int],
    predicate_name: str = "Aggregate Predicate",
    figsize: Tuple[float, float] = None,
    output_path: Optional[str] = None,
    dpi: int = 600
) -> plt.Figure:
    """
    Plot baseline model comparison with R² curves only.
    Creates one row per concept with two plots: full range and R² >= 0.40.
    
    Args:
        summary: Summarized results DataFrame with mean/min/max columns
        target_n_trains: List of training set sizes
        predicate_name: Name of the predicate(s) being evaluated
        figsize: Figure size (width, height). If None, auto-calculated based on number of concepts.
        output_path: Optional path to save figure
        dpi: DPI for saved figure
        
    Returns:
        Figure object
    """
    # Check if we have multiple concepts (predicate column exists)
    if 'predicate' in summary.columns:
        concepts = summary['predicate'].unique()
        n_concepts = len(concepts)
    else:
        concepts = [predicate_name]
        n_concepts = 1
    
    # Calculate figure size if not provided
    if figsize is None:
        figsize = (10, 4 * n_concepts)  # 2 plots per row, 4 units height per concept
    
    fig, axes = plt.subplots(n_concepts, 2, figsize=figsize)
    
    # Handle single concept case (axes won't be 2D)
    if n_concepts == 1:
        axes = axes.reshape(1, -1)

    # Get unique models for consistent ordering
    model_order = summary['model'].unique()

    for concept_idx, concept in enumerate(concepts):
        # Filter data for this concept
        if 'predicate' in summary.columns:
            concept_data = summary[summary['predicate'] == concept]
            concept_title = pretty_concept_name(concept)
        else:
            concept_data = summary
            concept_title = predicate_name

        for model_name in model_order:
            sub = concept_data[concept_data['model'] == model_name].sort_values('n_train')
            
            if sub.empty:
                continue
            
            # Plot R² (full range)
            axes[concept_idx, 0].plot(sub['n_train'], sub['r2_mean'], marker='s', label=model_name, linestyle='-')
            axes[concept_idx, 0].fill_between(sub['n_train'], sub['r2_min'], sub['r2_max'], alpha=0.15)
            
            # Plot R² (starting from N >= 40)
            sub_40 = sub[sub['n_train'] >= 40]
            if not sub_40.empty:
                axes[concept_idx, 1].plot(sub_40['n_train'], sub_40['r2_mean'], marker='s', label=model_name, linestyle='-')
                axes[concept_idx, 1].fill_between(sub_40['n_train'], sub_40['r2_min'], sub_40['r2_max'], alpha=0.15)

        # Configure axes for this concept
        for plot_idx, (ylabel, title_suffix) in enumerate([
            ('R²', f'{concept_title} — $R^2$ (Full Range)'),
            ('R²', f'{concept_title} — $R^2$ (N≥40)')
        ]):
            ax = axes[concept_idx, plot_idx]
            ax.set_xlabel('N training images')
            ax.set_xscale('log')
            
            if plot_idx == 1:  # Second plot with N >= 40
                ax.set_ylim(bottom=0.0, top=1.0)
                # Only show xticks for N >= 40
                concept_data_40 = concept_data[concept_data['n_train'] >= 40]
                if not concept_data_40.empty:
                    valid_n_trains = sorted(concept_data_40['n_train'].unique())
                    if valid_n_trains:
                        ax.set_xticks(valid_n_trains)
                        ax.set_xticklabels(valid_n_trains, rotation=45)
            else:  # First plot with full range
                ax.set_ylim(bottom=-1.0, top=1.0)
                ax.set_xticks(target_n_trains)
                ax.set_xticklabels(target_n_trains, rotation=45)
            
            ax.set_ylabel(ylabel)
            ax.set_title(title_suffix, fontsize=10)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Add legend to the first plot of each row
        axes[concept_idx, 0].legend(ncol=2, fontsize=8, frameon=True, loc='lower right')

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Baseline comparison plot saved to: {output_path}")
    
    return fig


def plot_estimator_comparison(
    summary: pd.DataFrame,
    concepts_plot: List[str],
    N_train_values: List[int],
    model_order: List[str] = None,
    figsize: Tuple[float, float] = (11, 4.2),
    output_path: Optional[str] = None,
    dpi: int = 600
) -> plt.Figure:
    """
    Plot permutation estimator comparison across concepts.
    
    Args:
        summary: Summarized results DataFrame
        concepts_plot: List of concepts to plot
        N_train_values: List of training set sizes (used as fallback if no data available)
        model_order: Order of models in legend
        figsize: Figure size (width, height)
        output_path: Optional path to save figure
        dpi: DPI for saved figure
        
    Returns:
        Figure object
    """
    if model_order is None:
        model_order = ["Linear", "Logistic", "Kernel RBF"]
    
    fig, axes = plt.subplots(1, len(concepts_plot), figsize=figsize, sharey=False)
    if len(concepts_plot) == 1:
        axes = [axes]
    
    fig.suptitle("N→1 alignment: R² vs N (mean ± range over seeds)", y=0.98, fontsize=13)

    # Get actual N values that exist in the data
    actual_n_values = sorted(summary['n_train'].unique())
    if len(actual_n_values) == 0:
        # Fallback to provided N_train_values if no data
        actual_n_values = N_train_values

    for ax, concept in zip(axes, concepts_plot):
        sub_all = summary[summary['concept'] == concept]
        
        for m in model_order:
            sub = sub_all[sub_all['model'] == m].sort_values('n_train')
            if sub.empty:
                continue
            ax.plot(sub['n_train'], sub['mean'], marker='o', linestyle='-', label=m)
            ax.fill_between(sub['n_train'], sub['min'], sub['max'], alpha=0.18)
        
        ax.set_xscale('log')
        ax.set_xticks(actual_n_values)
        ax.set_xticklabels(actual_n_values, rotation=45)
        ax.set_ylim(bottom=max(-0.1, summary['min'].min() - 0.1), top=1.05)
        ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.8)
        ax.set_title(pretty_concept_name(concept))
        ax.set_xlabel('N_train')
        ax.set_ylabel('R²')
        ax.legend(fontsize=8)

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Estimator comparison plot saved to: {output_path}")
    
    return fig


def plot_latent_heatmap_grouped(
    picks_by_seed: Dict,
    concept: str,
    N_vals: List[int],
    seeds: List[int],
    K: int = 4,
    sort_each_col: bool = True,
    figsize_base: Tuple[float, float] = (1.0, 0.8),
    figsize_scale: Tuple[float, float] = (0.5, 0.6),
    output_path: Optional[str] = None,
    dpi: int = 600
) -> plt.Figure:
    """
    Visualize evolution of K assigned latent indices across N and seeds as heatmaps.
    
    This creates 3 subplots (one for each model: Linear, Logistic, Kernel RBF) showing
    which latent dimensions are assigned to a specific concept across different training 
    set sizes and random seeds.
    
    Args:
        picks_by_seed: Dictionary with structure [seed][model][concept] -> list over N of assigned latent ids
        concept: Name of concept to visualize
        N_vals: List of N values in the same order as picks were appended
        seeds: List of seeds in plotting order
        K: Number of assigned latents per N for this concept
        sort_each_col: Whether to sort K indices within each column for stable visual ordering
        figsize_base: Base figure size (width, height)
        figsize_scale: Scaling factors for figure size based on data dimensions
        output_path: Optional path to save figure
        dpi: DPI for saved figure
        
    Returns:
        Figure object
    """
    import matplotlib.colors as mcolors
    from matplotlib.cm import get_cmap
    
    models = ["Linear", "Logistic", "Kernel RBF"]
    S = len(seeds)
    
    # Check if we have the new structure with models
    sample_seed = seeds[0]
    if isinstance(picks_by_seed[sample_seed], dict) and any(model in picks_by_seed[sample_seed] for model in models):
        # New structure: [seed][model][concept]
        has_models = True
    else:
        # Old structure: [seed][concept] - fallback for compatibility
        has_models = False
        models = ["Single Model"]
    
    # Detect which N values actually have data by examining the picks structure
    if has_models:
        # Check the first available model and seed to determine actual data length
        sample_model = None
        for model in models:
            if model in picks_by_seed[sample_seed] and concept in picks_by_seed[sample_seed][model]:
                sample_model = model
                break
        
        if sample_model:
            actual_data_length = len(picks_by_seed[sample_seed][sample_model][concept])
        else:
            actual_data_length = len(N_vals)  # Fallback
    else:
        # Old structure
        if concept in picks_by_seed[sample_seed]:
            actual_data_length = len(picks_by_seed[sample_seed][concept])
        else:
            actual_data_length = len(N_vals)  # Fallback
    
    # Use only the N values that correspond to actual data
    if actual_data_length < len(N_vals):
        print(f"Warning: Only {actual_data_length} N values have data, truncating from {len(N_vals)} to {actual_data_length}")
        N_vals_with_data = N_vals[:actual_data_length]
    else:
        N_vals_with_data = N_vals
    
    C = len(N_vals_with_data)
    
    # Create subplots: 3 rows, 1 column (stacked vertically)
    n_models = len(models)
    # Improved figure size calculation that accounts for K and number of models
    fig_w = figsize_base[0] + figsize_scale[0] * C * 1.2  # Extra width for labels
    fig_h = figsize_base[1] + figsize_scale[1] * S * K * n_models + 1.5 * (n_models - 1)  # Better spacing
    
    # Ensure minimum reasonable size for readability
    fig_w = max(fig_w, 8 + C * 0.8)  # Minimum width based on columns
    fig_h = max(fig_h, 4 * n_models + S * K * 0.4)  # Minimum height based on rows
    
    fig, axes = plt.subplots(n_models, 1, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes.flatten()  # Convert to 1D array for easier indexing
    
    # Collect all latent IDs across all models for consistent color mapping
    all_latent_ids = set()
    
    for model_idx, model in enumerate(models):
        # Build matrix M of shape (S*K, len(N_vals)) for this model
        M = -np.ones((S*K, C), dtype=int)  # fill with -1 (unused) for safety

        # Fill M for this model
        for si, seed in enumerate(seeds):
            if has_models:
                if model in picks_by_seed[seed] and concept in picks_by_seed[seed][model]:
                    picks_over_N = picks_by_seed[seed][model][concept]  # length C, each a list of length K
                else:
                    picks_over_N = [[] for _ in range(C)]  # Empty if model/concept not found
            else:
                picks_over_N = picks_by_seed[seed][concept]  # Old structure
            
            # Ensure we have the right number of N values
            if len(picks_over_N) != C:
                # Pad with empty lists if needed
                picks_over_N = picks_over_N[:C] + [[] for _ in range(max(0, C - len(picks_over_N)))]
            
            for cj in range(C):
                if cj < len(picks_over_N):
                    latents = list(picks_over_N[cj]) if picks_over_N[cj] else []
                else:
                    latents = []
                    
                if sort_each_col:
                    latents = sorted(latents)
                
                # Place them in K consecutive rows for this seed
                for ki in range(min(K, len(latents))):
                    M[si*K + ki, cj] = latents[ki]
                    all_latent_ids.add(latents[ki])

        # Store the matrix for this model
        if model_idx == 0:
            all_matrices = []
        all_matrices.append(M)

    # ---------------------------
    # Build a stable color mapping across all models
    # ---------------------------
    uniq = np.sort(list(all_latent_ids))  # sorted latent ids
    if len(uniq) == 0:
        raise ValueError("No assigned latents found to plot.")

    # map latent id -> 0..(n-1)
    id2idx = {int(val): i for i, val in enumerate(uniq)}
    n_colors = len(uniq)

    # choose a colormap:
    if n_colors <= 20:
        cmap_name = 'tab20'
    else:
        cmap_name = 'hsv'

    base_cmap = get_cmap(cmap_name)
    palette = base_cmap(np.linspace(0, 1, n_colors))  # RGBA array shape (n_colors, 4)

    # Create ListedColormap from palette and tell it to draw masked values as white
    cmap_listed = mcolors.ListedColormap(palette)
    cmap_listed.set_bad(color='white')

    # Helper function for text color
    def _text_color_for_rgb(rgb):
        r, g, b = rgb[:3]
        lum = 0.299*r + 0.587*g + 0.114*b
        return 'white' if lum < 0.5 else 'black'

    # ---------------------------
    # Plot heatmaps for each model
    # ---------------------------
    for model_idx, (model, M) in enumerate(zip(models, all_matrices)):
        ax = axes[model_idx]
        
        # Remap M into index space (0..n_colors-1), keep sentinel -1 for unused
        M_idx = np.full_like(M, fill_value=-1, dtype=int)
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                val = M[r, c]
                if val >= 0:
                    M_idx[r, c] = id2idx[int(val)]

        # Mask the unused cells
        M_masked = np.ma.masked_where(M_idx < 0, M_idx)

        # Plot heatmap
        im = ax.imshow(M_masked, aspect='auto', cmap=cmap_listed, vmin=0, vmax=n_colors-1)

        # Configure axes and ticks
        ax.set_xticks(range(C))
        ax.set_xticklabels(N_vals_with_data, rotation=45)

        yticks, ylabels = [], []
        for si, seed in enumerate(seeds):
            for ki in range(K):
                yticks.append(si*K + ki)
                ylabels.append(f'seed {seed} · slot {ki+1}')

        ax.set_yticks(yticks)
        # Adaptive y-tick label font size
        ytick_fontsize = min(9, max(6, 80 / (S * K)))  # Smaller font for more rows
        ax.set_yticklabels(ylabels, fontsize=ytick_fontsize)
        if model_idx == n_models - 1:  # Only label x-axis on bottom plot
            ax.set_xlabel('N_train')
        else:
            ax.set_xlabel('')  # Remove x-axis label for upper plots
        ax.set_ylabel('seed · slot')  # Label y-axis on all plots
        ax.set_title(f'{model}')

        # Annotate each cell with the latent id (adaptive font size)
        # Calculate adaptive font size based on cell density - increased for better readability
        cell_height = fig_h / (S * K * n_models)
        cell_width = fig_w / C
        base_font_size = min(12, max(8, min(cell_height * 8, cell_width * 4)))
        
        for r in range(M_idx.shape[0]):
            for c in range(M_idx.shape[1]):
                idx = M_idx[r, c]
                if idx >= 0:
                    latent_val = uniq[idx]
                    color_rgb = palette[idx]
                    txt_color = _text_color_for_rgb(color_rgb)
                    ax.text(c, r, str(int(latent_val)), ha='center', va='center', 
                           fontsize=base_font_size, color=txt_color, weight='bold')

        # Add light separators between seed blocks
        for si in range(1, S):
            ax.axhline(si*K - 0.5, color='k', linewidth=0.6, alpha=0.4)

    # Add main title
    fig.suptitle(f'{pretty_concept_name(concept)}: assigned latent indices (K={K})', fontsize=12, fontweight='bold')

    # Optional: add a compact legend mapping latent id -> color (only if not too many)
    if n_colors <= 40 and n_colors > 0:
        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor=palette[i], edgecolor='k', label=str(int(uniq[i])))
                          for i in range(n_colors)]
        # Place legend horizontally under the bottom subplot
        axes[-1].legend(handles=legend_handles, title='latent id', bbox_to_anchor=(0.5, -0.15), loc='upper center',
                  frameon=False, fontsize=8, title_fontsize=9, ncol=min(n_colors, 10))

    # Improved spacing with more room between subplots
    plt.tight_layout(rect=[0, 0.1, 1, 0.95], h_pad=3.0)  # Leave space for suptitle and bottom legend, more vertical padding
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Latent assignment heatmap saved to: {output_path}")
    
    return fig
