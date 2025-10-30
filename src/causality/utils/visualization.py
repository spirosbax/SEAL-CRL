"""
Visualization utilities for model comparison experiments.

This module provides functions for creating publication-ready plots
comparing different alignment methods across concepts and training sizes.
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import LogLocator, NullFormatter
from typing import List, Tuple, Optional
from .utils import pretty_concept_name


# Consistent model colors across all scripts
# Baseline: black, Linear: blue, Logistic: orange, Kernel: green
MODEL_COLOR_MAP = {
    # canonical keys
    "baseline": "#000000",
    "linear": "#1f77b4",
    "logistic": "#ff7f0e",
    "kernel_rbf": "#2ca02c",
    "kernel_laplacian": "#2ca02c",
    # friendly labels often used in plots
    "Baseline (Hungarian LR)": "#000000",
    "Baseline (cap-LR)": "#000000",
    "Linear": "#1f77b4",
    "Logistic": "#ff7f0e",
    "Kernel RBF": "#2ca02c",
    "Kernel Laplacian": "#2ca02c",
}


def summarize_records(records_df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by model/concept/N_train and compute mean/min/max of R².

    Args:
        records_df: DataFrame with columns [model, concept, n_train, r2]

    Returns:
        DataFrame with mean, min, max R² values grouped by model/concept/n_train
    """
    return (
        records_df.groupby(["model", "concept", "n_train"])["r2"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )


def plot_r2_grid(
    records_df: pd.DataFrame,
    concepts_order: List[str],
    N_values: List[int],
    model_order: List[str],
    n_cols: Optional[int] = None,
    y_lim: Tuple[float, float] = (-0.1, 1.05),
    figsize_per_ax: Tuple[float, float] = (3.4, 2.8),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compact, journal-ready grid:
      - shared x/y limits & scales across all panels;
      - only leftmost y-labels and bottom-row x-labels are labeled;
      - single legend outside the grid (top).
    """
    summary = summarize_records(records_df)

    # --- Matplotlib defaults for a clean look ---
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "grid.linewidth": 0.5,
        }
    )

    n_concepts = len(concepts_order)
    
    # Determine optimal grid layout based on number of concepts
    if n_cols is None:
        if n_concepts == 6:  # continuous case
            n_cols, n_rows = 3, 2
        elif n_concepts == 18:  # all case
            n_cols, n_rows = 6, 3
        elif n_concepts == 12:  # binary case
            n_cols, n_rows = 4, 3
        else:
            # Default: 4 columns for other cases
            n_cols = 4
            n_rows = math.ceil(n_concepts / n_cols)
    else:
        n_rows = math.ceil(n_concepts / n_cols)
    
    figsize = (figsize_per_ax[0] * n_cols, figsize_per_ax[1] * n_rows)

    # CHANGE: Remove sharey=True to avoid shared y-axis formatting issues
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    # set common style and MANUALLY set same y-limits for all
    y_ticks = np.arange(0.0, 1.01, 0.2)
    for ax in axes_flat:
        ax.set_xscale("log")
        ax.set_ylim(*y_lim)  # manually set same y-limits
        ax.set_yticks(y_ticks)  # manually set same y-ticks
        ax.set_facecolor("white")
        ax.grid(True, which="both", linestyle="--", color="lightgray", alpha=0.7)

    # log tick locator across panels
    locator_x = LogLocator(base=10, subs=(1.0,))
    for ax in axes_flat:
        ax.xaxis.set_major_locator(locator_x)
        ax.set_xticks(N_values)
        # Force regular number format instead of scientific notation
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x)}"))

    # plot each concept
    handles_for_legend, labels_for_legend = [], []
    for idx, concept in enumerate(concepts_order):
        ax = axes_flat[idx]
        sub_all = summary[summary["concept"] == concept]
        for m in model_order:
            sub = sub_all[sub_all["model"] == m].sort_values("n_train")
            if sub.empty:
                continue
            color = MODEL_COLOR_MAP.get(m, None)
            (ln,) = ax.plot(sub["n_train"], sub["mean"], marker="o", label=m, color=color)
            ax.fill_between(sub["n_train"], sub["min"], sub["max"], alpha=0.18, color=color)
            if idx == 0:
                handles_for_legend.append(ln)
                labels_for_legend.append(m)
        ax.set_title(pretty_concept_name(concept), fontsize=12, fontweight="bold")

    # hide unused axes
    for k in range(n_concepts, len(axes_flat)):
        axes_flat[k].set_visible(False)

    # Handle labels and tick visibility
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if not ax.get_visible():
                continue

            # Y-axis: only leftmost column gets labels
            if c == 0:
                ax.set_ylabel(r"$R^2$", fontsize=12, fontweight="bold")
                # Keep default y-tick labels (they'll show 0.0, 0.2, etc.)
            else:
                # Hide y-tick labels for non-leftmost columns
                ax.tick_params(axis="y", labelleft=False)

            # X-axis: only bottom row gets labels
            if r == n_rows - 1:
                ax.set_xlabel("Training samples $N$", fontsize=12, fontweight="bold")
                ax.set_xticklabels([str(v) for v in N_values], rotation=0)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    # global legend (top center) - single line legend with smaller font
    fig.legend(
        handles_for_legend,
        labels_for_legend,
        loc="upper center",
        ncol=len(labels_for_legend),  # force single line
        bbox_to_anchor=(0.5, 1.06),  # slightly above the grid
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=12,  # smaller font to keep one line
        markerscale=1.0,
        handlelength=2.0,
        handletextpad=0.6,
        columnspacing=1.0,
        borderpad=0.4,
    )

    plt.tight_layout(rect=[0, 0.0, 1, 0.965])

    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig


def plot_summary_comparison(
    records_df: pd.DataFrame,
    model_order: List[str],
    metric: str = "mean",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a summary plot showing overall model performance.

    Args:
        records_df: DataFrame with model comparison results
        model_order: List of model names in desired order
        metric: Which metric to plot ('mean', 'median', 'max')
        output_path: Optional path to save the figure

    Returns:
        The matplotlib Figure object
    """
    # Group by model and compute summary statistics
    if metric == "mean":
        summary = records_df.groupby("model")["r2"].mean()
    elif metric == "median":
        summary = records_df.groupby("model")["r2"].median()
    elif metric == "max":
        summary = records_df.groupby("model")["r2"].max()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Reorder according to model_order
    summary = summary.reindex(model_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(len(summary)), summary.values)

    # Customize the plot
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{metric.capitalize()} R²", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Overall Model Performance ({metric.capitalize()} R²)",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary.index, rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, summary.values)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"Saved summary plot to {output_path}")

    return fig


def plot_class_counts_grid(
    concepts_order: List[str],
    counts_zero: List[int],
    counts_one: List[int],
    n_cols: Optional[int] = None,
    figsize_per_ax: Tuple[float, float] = (3.4, 2.8),
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a compact, publication-ready grid of binary class counts per concept.

    Args:
        concepts_order: Ordered list of concept names (length = M)
        counts_zero: List of counts for class 0 per concept (length = M)
        counts_one: List of counts for class 1 per concept (length = M)
        n_cols: Optional number of columns in the grid (auto if None)
        figsize_per_ax: Figure size per subplot (width, height)
        output_path: Optional path to save the figure

    Returns:
        Matplotlib Figure
    """
    import math

    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
        }
    )

    n_concepts = len(concepts_order)
    if n_cols is None:
        if n_concepts == 12:
            n_cols, n_rows = 4, 3
        elif n_concepts == 6:
            n_cols, n_rows = 3, 2
        elif n_concepts == 18:
            n_cols, n_rows = 6, 3
        else:
            n_cols = 4
            n_rows = math.ceil(n_concepts / n_cols)
    else:
        n_rows = math.ceil(n_concepts / n_cols)

    figsize = (figsize_per_ax[0] * n_cols, figsize_per_ax[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    axes_flat = axes.flatten()

    max_count = max([counts_zero[i] + counts_one[i] for i in range(n_concepts)])
    y_lim = (0, max(1, int(math.ceil(max_count * 1.05))))

    colors = ["#8da0cb", "#fc8d62"]

    for idx, concept in enumerate(concepts_order):
        ax = axes_flat[idx]
        c0 = counts_zero[idx]
        c1 = counts_one[idx]
        bars = ax.bar(["0", "1"], [c0, c1], color=colors)
        ax.set_ylim(*y_lim)
        ax.set_title(pretty_concept_name(concept), fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", color="lightgray", alpha=0.7)

        for bar, value in zip(bars, [c0, c1]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(1, 0.01 * y_lim[1]),
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Hide unused axes
    for k in range(n_concepts, len(axes_flat)):
        axes_flat[k].set_visible(False)

    # Label visibility: only leftmost y and bottom x labels
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c]
            if not ax.get_visible():
                continue
            if c != 0:
                ax.tick_params(axis="y", labelleft=False)
            if r != n_rows - 1:
                ax.tick_params(axis="x", labelbottom=False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    return fig


def plot_learning_curves(
    records_df: pd.DataFrame,
    concepts_to_show: List[str],
    model_order: List[str],
    show_individual_seeds: bool = False,
    output_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create learning curve plots for selected concepts.

    Args:
        records_df: DataFrame with model comparison results
        concepts_to_show: List of concept names to plot
        model_order: List of model names in desired order
        show_individual_seeds: Whether to show individual seed results
        output_path: Optional path to save the figure

    Returns:
        The matplotlib Figure object
    """
    n_concepts = len(concepts_to_show)
    n_cols = min(3, n_concepts)
    n_rows = math.ceil(n_concepts / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_concepts == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, concept in enumerate(concepts_to_show):
        ax = axes[idx]
        concept_data = records_df[records_df["concept"] == concept]

        for model in model_order:
            model_data = concept_data[concept_data["model"] == model]

            if show_individual_seeds:
                # Plot individual seeds
                for seed in model_data["seed"].unique():
                    seed_data = model_data[model_data["seed"] == seed].sort_values(
                        "n_train"
                    )
                    ax.plot(
                        seed_data["n_train"], seed_data["r2"], alpha=0.3, linewidth=1
                    )

            # Plot mean across seeds
            mean_data = (
                model_data.groupby("n_train")["r2"].agg(["mean", "std"]).reset_index()
            )

            ax.plot(
                mean_data["n_train"],
                mean_data["mean"],
                marker="o",
                linewidth=2,
                label=model,
            )

            # Add error bars if multiple seeds
            if len(model_data["seed"].unique()) > 1:
                ax.fill_between(
                    mean_data["n_train"],
                    mean_data["mean"] - mean_data["std"],
                    mean_data["mean"] + mean_data["std"],
                    alpha=0.2,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Training samples")
        ax.set_ylabel("R²")
        ax.set_title(pretty_concept_name(concept))
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Hide unused subplots
    for idx in range(n_concepts, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"Saved learning curves to {output_path}")

    return fig
