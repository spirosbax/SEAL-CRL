#!/usr/bin/env python3
"""
Grouped predicate class counts (1x4 grid)

This script loads the aligned simulator data and produces:
- A CSV with class counts for four grouped predicates
- A compact 1x4 grid plot (PDF and PNG) with per-predicate class counts

Predicates:
- all_stoves_on
- at_least_three_stoves_on
- at_least_two_stoves_on
- egg_intact

Usage example:
    python experiments/predicate_class_counts_grid.py \
        --data_dir src/data/ithor --split test --output_dir results/class_counts_grouped
"""

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd

# Repo imports
from causality.data.ithor_loader import load_causality_data
from causality.utils.utils import setup_output_dir
from causality.utils.visualization import plot_class_counts_grid
from causality.utils.baseline_models import (
    create_aggregate_predicate,
    get_predicate_configuration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grouped predicate class counts (1x4 grid)")
    parser.add_argument("--data_dir", type=str, default="src/data/ithor", help="Directory containing the iTHOR data")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (test, train, val)")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to load (None for all)")
    parser.add_argument("--output_dir", type=str, default="results/class_counts_grouped", help="Directory to save results")
    parser.add_argument("--no_cache", action="store_true", help="Disable caching of loaded data")
    parser.add_argument("--dpi", type=int, default=600, help="DPI for saved figures")
    return parser.parse_args()


def compute_counts(y: np.ndarray) -> Tuple[int, int]:
    classes, counts = np.unique(y.astype(int), return_counts=True)
    m = {int(c): int(n) for c, n in zip(classes.tolist(), counts.tolist())}
    return m.get(0, 0), m.get(1, 0)


def main() -> None:
    args = parse_args()

    print("Grouped predicate class counts (1x4 grid)")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Split: {args.split}")
    print(f"Max files: {args.max_files}")
    print(f"Output directory: {args.output_dir}")
    print()

    setup_output_dir(args.output_dir)

    print("Loading causality data...")
    _, sim_vars, _, sim_var_name_to_data = load_causality_data(
        data_dir=args.data_dir,
        split=args.split,
        max_files=args.max_files,
        use_cache=not args.no_cache,
    )

    print(f"Loaded simulator variables: {sim_vars.shape}")

    # Fixed order and pretty names
    predicates: List[str] = [
        "all_stoves_on",
        "at_least_three_stoves_on",
        "at_least_two_stoves_on",
        "egg_intact",
    ]

    pretty_name = {
        "all_stoves_on": "All Stoves On",
        "at_least_three_stoves_on": "At Least Three Stoves On",
        "at_least_two_stoves_on": "At Least Two Stoves On",
        "egg_intact": "Egg Intact",
    }

    counts_zero: List[int] = []
    counts_one: List[int] = []
    available: List[str] = []

    for p in predicates:
        try:
            cfg = get_predicate_configuration(p)
            y = create_aggregate_predicate(sim_var_name_to_data, p, cfg["variables"]).astype(int)
            c0, c1 = compute_counts(y)
            counts_zero.append(c0)
            counts_one.append(c1)
            available.append(p)
        except Exception as e:
            print(f"Skipping {p}: {e}")

    if not available:
        raise RuntimeError("No predicates available to plot after checks.")

    # CSV
    df = pd.DataFrame(
        {
            "predicate": available,
            "pretty_name": [pretty_name.get(p, p) for p in available],
            "count_0": counts_zero,
            "count_1": counts_one,
            "total": [counts_zero[i] + counts_one[i] for i in range(len(available))],
        }
    )
    df["positive_rate"] = df["count_1"] / df["total"].replace(0, np.nan)

    csv_path = os.path.join(args.output_dir, "predicate_class_counts.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved counts CSV to {csv_path}")

    # 1x4 grid (force n_cols=4)
    grid_pdf = os.path.join(args.output_dir, "predicate_class_counts_grid.pdf")
    fig = plot_class_counts_grid(
        concepts_order=[pretty_name.get(p, p) for p in available],
        counts_zero=counts_zero,
        counts_one=counts_one,
        n_cols=4,
        figsize_per_ax=(3.4, 2.8),
        output_path=grid_pdf,
    )

    grid_png = grid_pdf.replace(".pdf", ".png")
    fig.savefig(grid_png, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved grid plot to {grid_pdf} and {grid_png}")

    print("Done.")


if __name__ == "__main__":
    main()


