#!/usr/bin/env python3
"""
Quick class counts utility for key predicates and a control concept.

Generates per-predicate bar plots (counts of class 0 vs 1) for:
- all_stoves_on
- at_least_two_stoves_on
- at_least_three_stoves_on
- egg_intact
- Microwave_d8b935e4_open (control)

Usage:
    python experiments/predicate_class_counts.py \
        --data_dir src/data/ithor --split test --output_dir results/class_counts
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# Repo imports
from causality.data.ithor_loader import load_causality_data
from causality.utils.utils import setup_output_dir
from causality.utils.baseline_models import (
    create_aggregate_predicate,
    get_predicate_configuration,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predicate class counts plots")
    parser.add_argument("--data_dir", type=str, default="src/data/ithor", help="Directory with iTHOR data")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (test/train/val)")
    parser.add_argument("--max_files", type=int, default=None, help="Max files to load (None = all)")
    parser.add_argument("--output_dir", type=str, default="results/class_counts", help="Directory to save plots")
    parser.add_argument("--no_cache", action="store_true", help="Disable data caching")
    parser.add_argument("--dpi", type=int, default=400, help="Figure DPI")
    return parser.parse_args()


def compute_predicate_labels(
    predicate: str,
    sim_var_name_to_data: Dict[str, np.ndarray]
) -> np.ndarray:
    """Return binary labels for a supported predicate or known control variable."""
    if predicate in {"all_stoves_on", "at_least_two_stoves_on", "at_least_three_stoves_on", "egg_intact"}:
        cfg = get_predicate_configuration(predicate)
        return create_aggregate_predicate(sim_var_name_to_data, predicate, cfg["variables"]).astype(int)

    # Control concept (microwave) is a direct simulator variable
    if predicate == "Microwave_d8b935e4_open":
        if predicate not in sim_var_name_to_data:
            raise KeyError(f"Simulator variable '{predicate}' not found in dataset.")
        return sim_var_name_to_data[predicate].astype(int)

    raise ValueError(f"Unsupported predicate or concept: {predicate}")


def plot_class_counts(y: np.ndarray, title: str, out_path: str, dpi: int = 400) -> None:
    classes, counts = np.unique(y.astype(int), return_counts=True)
    # Ensure bars for both 0 and 1 are present
    counts_map = {int(c): int(n) for c, n in zip(classes.tolist(), counts.tolist())}
    count_0 = counts_map.get(0, 0)
    count_1 = counts_map.get(1, 0)

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.bar(["0", "1"], [count_0, count_1], color=["#8da0cb", "#fc8d62"])
    ax.set_title(title)
    ax.set_ylabel("Count")
    for idx, val in enumerate([count_0, count_1]):
        ax.text(idx, val + max(1, 0.01 * max(count_0, count_1, 1)), str(val), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    setup_output_dir(args.output_dir)

    # Load dataset
    _, sim_vars, _, sim_var_name_to_data = load_causality_data(
        data_dir=args.data_dir,
        split=args.split,
        max_files=args.max_files,
        use_cache=not args.no_cache,
    )
    print(f"Loaded simulator variables: {sim_vars.shape}")

    # Targets to evaluate
    targets: List[str] = [
        "all_stoves_on",
        "at_least_three_stoves_on",
        "at_least_two_stoves_on",
        "egg_intact",
        "Microwave_d8b935e4_open",  # control
    ]

    # Friendly titles
    pretty_name = {
        "all_stoves_on": "All Stoves On",
        "at_least_three_stoves_on": "At Least Three Stoves On",
        "at_least_two_stoves_on": "At Least Two Stoves On",
        "egg_intact": "Egg Intact",
        "Microwave_d8b935e4_open": "Microwave Open (control)",
    }

    # Compute and plot counts
    for t in targets:
        try:
            y = compute_predicate_labels(t, sim_var_name_to_data)
        except Exception as e:
            print(f"Skipping {t}: {e}")
            continue

        title = f"Class Counts - {pretty_name.get(t, t)}"
        fname = t + ".png"
        out_path = os.path.join(args.output_dir, fname)
        plot_class_counts(y, title, out_path, dpi=args.dpi)
        classes, counts = np.unique(y, return_counts=True)
        print(f"{t}: " + ", ".join([f"{int(c)}={int(n)}" for c, n in zip(classes.tolist(), counts.tolist())]))
        print(f"  -> saved: {out_path}")


if __name__ == "__main__":
    main()


