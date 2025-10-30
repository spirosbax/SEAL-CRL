"""
iTHOR dataset loading utilities for causality analysis.
"""

import os
import sys
from glob import glob
from typing import Optional, Dict, List, Tuple
import numpy as np
from torch.utils.data import DataLoader

# Add BISCUIT to path for imports
sys.path.append("external/BISCUIT")
from experiments.datasets import iTHORDataset


def load_ithor_dataset(
    data_dir: str,
    split: str = "test",
    max_files: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    try_vae_encodings: bool = False,
) -> iTHORDataset:
    """
    Load the iTHOR dataset using the existing dataset class.

    Args:
        data_dir: Path to iTHOR dataset directory
        split: Dataset split to load ('test', 'train', 'val')
        max_files: Maximum number of files to load (None for all)
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        try_vae_encodings: Whether to try loading pre-computed VAE encodings instead of raw images

    Returns:
        iTHORDataset instance
    """
    print(f"Loading iTHOR {split} dataset from {data_dir}")

    # Create dataset with single_image=True for reconstruction checks
    dataset = iTHORDataset(
        data_folder=data_dir,
        split=split,
        single_image=True,
        return_robot_state=True,
        return_targets=True,
        return_latents=True,
        triplet=False,
        seq_len=1,
        cluster=False,
        try_encodings=try_vae_encodings,  # This will try to load pre-computed VAE encodings
    )

    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Image shape: {dataset.get_img_width()}x{dataset.get_img_width()}")
    print(f"Input channels: {dataset.get_inp_channels()}")
    print(f"Action size: {dataset.action_size()}")
    print(f"Number of causal variables: {dataset.num_vars()}")

    return dataset


def get_aligned_seq_files(data_dir: str, split: str) -> List[str]:
    """
    Get sequence files in the exact same order as iTHORDataset loads them.

    This ensures alignment between regular dataset loading and NF encoding loading.
    """
    split_dir = os.path.join(data_dir, split)
    # Use the same logic as iTHORDataset.load_data_from_folder
    seq_files = sorted(glob(os.path.join(split_dir, "*seq_*.npz")))
    seq_files = [f for f in seq_files if not f.endswith("_encodings.npz")]
    return seq_files


def load_nf_encodings_aligned(
    data_dir: str,
    split: str,
    max_files: Optional[int] = None,
    seq_files: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Load NF encodings in the exact same order as iTHORDataset loads sequence files.

    Args:
        data_dir: Path to iTHOR dataset directory
        split: Dataset split ('test', 'train', 'val')
        max_files: Maximum number of files to load (None for all)
        seq_files: Optional precomputed list of aligned sequence files to avoid re-scanning

    Returns:
        NF encodings array with shape (n_samples, n_dims)
    """
    # Use provided aligned sequence files if available, otherwise compute them
    if seq_files is None:
        seq_files = get_aligned_seq_files(data_dir, split)

    if max_files is not None:
        seq_files = seq_files[:max_files]

    print(f"Loading NF encodings from {len(seq_files)} files in {split} split...")

    all_encodings = []
    for i, seq_file in enumerate(seq_files):
        if i % 50 == 0:
            print(f"  Loading file {i+1}/{len(seq_files)}")

        # Convert sequence file path to NF encoding file path
        nf_file = seq_file.replace(".npz", "_nf_encodings.npz")

        if not os.path.exists(nf_file):
            raise FileNotFoundError(f"NF encoding file not found: {nf_file}")

        data = np.load(nf_file)
        encodings = data["encodings"]  # Shape: (seq_len, 40)
        all_encodings.append(encodings)

    # Concatenate all encodings
    biscuit_vars = np.concatenate(all_encodings, axis=0)
    print(f"Loaded NF encodings shape: {biscuit_vars.shape}")

    return biscuit_vars


def check_nf_encodings_exist(data_dir: str, split: str) -> Tuple[bool, List[str]]:
    """
    Check if NF encodings exist for the given split.

    Returns:
        Tuple of (exists, list_of_nf_files)
    """
    split_dir = os.path.join(data_dir, split)
    nf_files = []
    if os.path.exists(split_dir):
        nf_files = sorted(glob(os.path.join(split_dir, "*_nf_encodings.npz")))

    has_nf_encodings = len(nf_files) > 0
    return has_nf_encodings, nf_files


def load_simulator_variables_from_dataset(
    dataset: iTHORDataset, max_files: Optional[int] = None
) -> np.ndarray:
    """
    Load simulator variables (latents) from the iTHORDataset.

    Args:
        dataset: Pre-loaded iTHORDataset instance
        max_files: Maximum number of files to process (for alignment with NF encodings)

    Returns:
        Simulator variables array with shape (n_samples, n_vars)
    """
    print("Loading simulator variables from dataset...")

    # Calculate expected number of samples if max_files is set
    if max_files is not None:
        # Assuming 100 samples per file (this is typical for iTHOR sequences)
        expected_samples = max_files * 100
        actual_samples = min(expected_samples, len(dataset))
    else:
        actual_samples = len(dataset)

    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    all_targets = []
    total_processed = 0

    for batch_idx, batch in enumerate(data_loader):
        if batch_idx % 50 == 0:
            print(f"Processing batch {batch_idx + 1}/{len(data_loader)}")

        # Unpack batch: (encodings, robot_state, targets, latents)
        encodings, robot_state, targets, latents = batch

        # Use latents as ground truth simulator variables
        batch_latents = latents.numpy()
        all_targets.append(batch_latents)

        total_processed += len(batch_latents)

        # Stop if we've processed enough samples (for max_files alignment)
        if max_files is not None and total_processed >= actual_samples:
            # Trim the last batch if necessary
            if total_processed > actual_samples:
                excess = total_processed - actual_samples
                all_targets[-1] = all_targets[-1][:-excess]
            break

    sim_vars = np.concatenate(all_targets, axis=0)
    print(f"Loaded simulator variables shape: {sim_vars.shape}")

    return sim_vars


def load_aligned_data(
    data_dir: str,
    split: str,
    max_files: Optional[int] = None,
    nf_files: Optional[List[str]] = None,
    seq_files: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load perfectly aligned BISCUIT variables and simulator variables.

    Args:
        data_dir: Path to iTHOR dataset directory
        split: Dataset split ('test', 'train', 'val')
        max_files: Maximum number of files to load (None for all)
        nf_files: Optional list of NF encoding files already discovered by caller
        seq_files: Optional list of aligned sequence files to avoid re-computing

    Returns:
        Tuple of (biscuit_vars, sim_vars, var_names)
    """
    # If caller didn't pass lists, validate existence for helpful error messages
    if nf_files is None:
        has_nf_encodings, nf_files_local = check_nf_encodings_exist(data_dir, split)
        if not has_nf_encodings:
            raise FileNotFoundError(f"No NF encodings found in {split} split!")
        nf_files = nf_files_local
    print(f"Found {len(nf_files)} NF encoding files in {split} directory")

    # Load dataset to get simulator variables; get variable names from utils to avoid heavy calls
    # Note: The ordering in get_all_variable_info must match the dataset latent ordering
    from utils.model_evaluation import get_all_variable_info

    dataset = load_ithor_dataset(
        data_dir, split=split, max_files=None
    )  # Load full dataset
    var_info = get_all_variable_info()
    all_vars = list(var_info.keys())

    # Determine aligned sequence file list preference/order
    aligned_seq_files: Optional[List[str]] = None
    if seq_files is not None:
        aligned_seq_files = seq_files
    else:
        # If we only have nf_files, build aligned sequence ordering and keep only those with encodings
        aligned_all_seq = get_aligned_seq_files(data_dir, split)
        nf_basenames = {os.path.basename(f) for f in nf_files}

        def seq_has_nf(s: str) -> bool:
            candidate_nf = os.path.basename(s).replace(".npz", "_nf_encodings.npz")
            return candidate_nf in nf_basenames

        aligned_seq_files = [s for s in aligned_all_seq if seq_has_nf(s)]

    # Load NF encodings in aligned order
    biscuit_vars = load_nf_encodings_aligned(
        data_dir, split, max_files, seq_files=aligned_seq_files
    )

    # Load simulator variables in aligned order
    sim_vars = load_simulator_variables_from_dataset(dataset, max_files)

    # Ensure shapes match exactly
    min_len = min(len(biscuit_vars), len(sim_vars))
    if len(biscuit_vars) != len(sim_vars):
        print(f"Warning: Shape mismatch. Trimming to {min_len} samples.")
        biscuit_vars = biscuit_vars[:min_len]
        sim_vars = sim_vars[:min_len]

    print(f"Final aligned data shapes:")
    print(f"  BISCUIT variables (from NF encodings): {biscuit_vars.shape}")
    print(f"  Simulator variables: {sim_vars.shape}")

    return biscuit_vars, sim_vars, all_vars


def load_causality_data(
    data_dir: str,
    split: str = "test",
    max_files: Optional[int] = None,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, np.ndarray]]:
    """
    All-inclusive function to load causality analysis data with caching and error handling.

    This function handles:
    - Checking for cached numpy arrays
    - Loading from NF encodings if available
    - Proper alignment between BISCUIT and simulator variables
    - Automatic caching for future use
    - Clear error messages if prerequisites are missing

    Args:
        data_dir: Path to iTHOR dataset directory
        split: Dataset split ('test', 'train', 'val')
        max_files: Maximum number of files to load (None for all)
        use_cache: Whether to use/create cache files

    Returns:
        Tuple of (biscuit_vars, sim_vars, var_names, var_name_to_data_dict)

    Raises:
        FileNotFoundError: If NF encodings don't exist and need to be generated
    """
    print(f"Loading causality data for {split} split...")

    # Define cache file paths
    cache_suffix = f"_{max_files}files" if max_files is not None else ""
    biscuit_vars_file = os.path.join(
        data_dir, f"biscuit_vars_{split}{cache_suffix}.npy"
    )
    sim_vars_file = os.path.join(data_dir, f"sim_vars_{split}{cache_suffix}.npy")

    # Try loading from cache first
    if (
        use_cache
        and os.path.exists(biscuit_vars_file)
        and os.path.exists(sim_vars_file)
    ):
        print("Loading cached variables from numpy files...")
        biscuit_vars = np.load(biscuit_vars_file)
        sim_vars = np.load(sim_vars_file)

        # Load variable names without instantiating the dataset
        from utils.model_evaluation import get_all_variable_info

        var_info = get_all_variable_info()
        all_vars = list(var_info.keys())

        print(f"Loaded cached data:")
        print(f"  BISCUIT variables: {biscuit_vars.shape}")
        print(f"  Simulator variables: {sim_vars.shape}")

    else:
        # Check if NF encodings exist
        has_nf_encodings, nf_files = check_nf_encodings_exist(data_dir, split)

        if has_nf_encodings:
            print("Loading aligned data from NF encodings...")
            # Precompute aligned sequence files to avoid re-scanning
            seq_files = get_aligned_seq_files(data_dir, split)
            biscuit_vars, sim_vars, all_vars = load_aligned_data(
                data_dir, split, max_files, nf_files=nf_files, seq_files=seq_files
            )

            # Save for future use if caching is enabled
            if use_cache:
                print("Saving variables to cache...")
                np.save(biscuit_vars_file, biscuit_vars)
                np.save(sim_vars_file, sim_vars)

        else:
            print(f"❌ No NF encodings found in {split} split!")
            print(
                "Please run the convert_encodings.py script first to generate NF encodings:"
            )
            print(
                f"  python src/utils/convert_encodings.py --data_dir {data_dir} --splits {split}"
            )
            print()
            print(
                "This script converts VAE encodings to causal (NF) encodings using the BISCUIT flow model."
            )
            raise FileNotFoundError(
                f"NF encodings not found for {split} split. Run convert_encodings.py first."
            )

    # Create variable name to data mapping
    sim_var_name_to_data = {}
    for i, var_name in enumerate(all_vars):
        sim_var_name_to_data[var_name] = sim_vars[:, i]

    print(f"Successfully loaded causality data:")
    print(f"  Split: {split}")
    print(f"  BISCUIT dimensions: {biscuit_vars.shape[1]}")
    print(f"  Causal variables: {len(all_vars)}")
    print(f"  Total samples: {len(biscuit_vars)}")

    return biscuit_vars, sim_vars, all_vars, sim_var_name_to_data


def create_var_info(causal_keys: List[str]) -> Dict[str, str]:
    """
    Create variable info dictionary with variable types.

    Args:
        causal_keys: List of causal variable names

    Returns:
        Dictionary mapping variable names to their types
    """
    var_info = {}
    for var_name in causal_keys:
        if any(suffix in var_name for suffix in ["_x", "_y", "_z"]):
            var_info[var_name] = "continuous"
        else:
            var_info[var_name] = "binary"

    return var_info
