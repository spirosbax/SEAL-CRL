#!/usr/bin/env python3
"""
Binary variable class counts (publication-ready grid)

This script loads the same aligned data used by other experiments and produces:
- A CSV with class counts for all binary simulator variables
- A compact grid plot (PDF and PNG) with per-variable class counts

Usage example:
    python experiments/binary_class_counts.py \
        --data_dir src/data/ithor --split test --output_dir results/class_counts_binary
"""

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

# Prefer package-style imports; fallback to adding src for local runs
# Repo imports
from causality.data.ithor_loader import load_causality_data
from causality.utils.utils import setup_output_dir
from causality.utils.model_evaluation import create_concept_order
from causality.utils.visualization import plot_class_counts_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary variable class counts grid")
    parser.add_argument("--data_dir", type=str, default="src/data/ithor", help="Directory containing the iTHOR data")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (test, train, val)")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to load (None for all)")
    parser.add_argument("--output_dir", type=str, default="results/class_counts_binary", help="Directory to save results")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching of loaded data")
    parser.add_argument("--n_cols", type=int, default=None, help="Number of columns in grid plot (auto if None)")
    parser.add_argument("--figsize_per_ax", type=float, nargs=2, default=[3.4, 2.8], help="Figure size per subplot (width height)")
    parser.add_argument("--dpi", type=int, default=600, help="DPI for saved figures")
    return parser.parse_args()


def compute_binary_counts(
    concept_names: List[str],
    sim_var_name_to_data,
) -> Tuple[List[int], List[int]]:
    counts_zero: List[int] = []
    counts_one: List[int] = []
    for concept in concept_names:
        if concept not in sim_var_name_to_data:
            continue
        y = sim_var_name_to_data[concept].astype(int)
        classes, counts = np.unique(y, return_counts=True)
        count_map = {int(c): int(n) for c, n in zip(classes.tolist(), counts.tolist())}
        counts_zero.append(count_map.get(0, 0))
        counts_one.append(count_map.get(1, 0))
    return counts_zero, counts_one


def main() -> None:
    args = parse_args()

    print("Binary variable class counts (grid)")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Max files: {args.max_files}")
    print(f"Output directory: {args.output_dir}")
    print()

    setup_output_dir(args.output_dir)

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

    # Standard binary order used across experiments
    concept_order = list(create_concept_order(variable_types="binary"))
    available_concepts = [c for c in concept_order if c in sim_var_name_to_data]
    if len(available_concepts) < len(concept_order):
        missing = [c for c in concept_order if c not in available_concepts]
        print(f"Warning: Missing binary concepts in dataset: {missing}")
        concept_order = available_concepts

    print(f"Using {len(concept_order)} binary concepts:")
    for c in concept_order:
        print(f"  - {c}")

    counts_zero, counts_one = compute_binary_counts(concept_order, sim_var_name_to_data)

    df = pd.DataFrame(
        {
            "concept": concept_order,
            "count_0": counts_zero,
            "count_1": counts_one,
            "total": [counts_zero[i] + counts_one[i] for i in range(len(concept_order))],
        }
    )
    df["positive_rate"] = df["count_1"] / df["total"].replace(0, np.nan)

    csv_path = os.path.join(args.output_dir, "binary_class_counts.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved counts CSV to {csv_path}")

    grid_pdf = os.path.join(args.output_dir, "binary_class_counts_grid.pdf")
    fig = plot_class_counts_grid(
        concepts_order=concept_order,
        counts_zero=counts_zero,
        counts_one=counts_one,
        n_cols=args.n_cols,
        figsize_per_ax=tuple(args.figsize_per_ax),
        output_path=grid_pdf,
    )

    grid_png = grid_pdf.replace(".pdf", ".png")
    fig.savefig(grid_png, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved grid plot to {grid_pdf} and {grid_png}")

    print("Done.")


if __name__ == "__main__":
    main()


