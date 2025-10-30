"""
R² Analysis utilities for causality experiments.

This module provides functions for computing R² metrics and analyzing
relationships between latent variables and causal variables.

Note: R² computation uses the existing compute_r2() function from utils.py
for consistency across the codebase.
"""

import numpy as np
from typing import Dict, List, Tuple
from .utils import compute_r2


def create_variable_groups() -> Dict[str, List[str]]:
    """
    Create the standard variable groupings for iTHOR dataset analysis.
    
    Returns
    -------
    Dict[str, List[str]]
        Mapping from group names to lists of variable names.
    """
    group_map = {
        "Cabinet - Open": ["Cabinet_47fc321b_open"],
        "Egg - State": [
            "Egg_afaaaca3_broken", "Egg_afaaaca3_center_x",
            "Egg_afaaaca3_center_y", "Egg_afaaaca3_center_z",
            "Egg_afaaaca3_cooked", "Egg_afaaaca3_pickedup",
        ],
        "Microwave - Open": ["Microwave_d8b935e4_open"],
        "Microwave - Active": ["Microwave_d8b935e4_on"],
        "Plate - State": [
            "Plate_49b95a7a_center_x", "Plate_49b95a7a_center_y",
            "Plate_49b95a7a_center_z", "Plate_49b95a7a_pickedup",
        ],
        "Stove 1 - Active": ["StoveKnob_690d0d5d_on"],
        "Stove 2 - Active": ["StoveKnob_c8955f66_on"],
        "Stove 3 - Active": ["StoveKnob_38c1dbc2_on"],
        "Stove 4 - Active": ["StoveKnob_cf670576_on"],
        "Toaster - Active": ["Toaster_194647f5_on"],
    }
    return group_map


def compute_grouped_r2_metrics(full_R2: np.ndarray, 
                              group_map: Dict[str, List[str]], 
                              x_vars: List[str], 
                              var_to_lat: Dict[str, int]) -> Dict:
    """
    Compute R²-diag and R²-sep metrics for grouped variables.
    
    Parameters
    ----------
    full_R2 : np.ndarray
        Full R² matrix with shape (n_latent_dims, n_variables)
    group_map : Dict[str, List[str]]
        Mapping from group names to variable names
    x_vars : List[str]
        List of all variable names in order
    var_to_lat : Dict[str, int]
        Mapping from variable names to their best latent dimension
        
    Returns
    -------
    Dict
        Dictionary containing R2_diag, R2_sep, diag_vals, and sep_vals
    """
    diag_vals = []
    sep_vals = []

    print("\nComputing grouped R² metrics:")
    
    for g, vars_in_g in group_map.items():
        print(f"\nProcessing group: {g}")
        print(f"Variables in group: {vars_in_g}")
        
        # Matched R² for each variable in the group
        matched_r2s = []
        for v in vars_in_g:
            if v not in var_to_lat:
                print(f"  Warning: Variable {v} not found in var_to_lat mapping")
                continue
                
            lat_dim = var_to_lat[v]
            var_idx = x_vars.index(v)
            r2_val = full_R2[lat_dim, var_idx]
            print(f"  Variable {v}: using latent dim {lat_dim}, R² = {r2_val:.4f}")
            matched_r2s.append(r2_val)
        
        if not matched_r2s:
            print(f"  Warning: No valid variables found for group {g}")
            continue
            
        mean_r2 = np.mean(matched_r2s)
        print(f"Group {g} summary:")
        print(f"  Matched R² values: {matched_r2s}")
        print(f"  Mean R² for group: {mean_r2:.4f}")
        
        diag_vals.append(mean_r2)  # average inside the group

        # sep: for each variable, best latent that is *not* its matched one
        offs = []
        for v in vars_in_g:
            if v not in var_to_lat:
                continue
                
            matched_lat = var_to_lat[v]
            j = x_vars.index(v)
            # Get R² for all latents except the matched one
            other_r2s = np.delete(full_R2[:, j], matched_lat)
            if len(other_r2s) > 0:
                offs.append(np.max(other_r2s))
        
        if offs:
            sep_vals.append(np.mean(offs))

    R2_diag = np.mean(diag_vals) if diag_vals else np.nan
    R2_sep = np.mean(sep_vals) if sep_vals else np.nan
    
    print(f"\nFinal metrics:")
    print(f"R²-diag: {R2_diag:.4f}, R²-sep: {R2_sep:.4f}")
    
    return {
        'R2_diag': R2_diag,
        'R2_sep': R2_sep,
        'diag_vals': diag_vals,
        'sep_vals': sep_vals
    }


def compute_r2_matrix(biscuit_vars: np.ndarray, 
                     sim_var_name_to_data: Dict[str, np.ndarray],
                     var_names: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Compute the full R² matrix between latent dimensions and causal variables.
    
    Parameters
    ----------
    biscuit_vars : np.ndarray
        BISCUIT latent variables with shape (n_samples, n_latent_dims)
    sim_var_name_to_data : Dict[str, np.ndarray]
        Mapping from variable names to data arrays
    var_names : List[str]
        List of variable names to analyze
        
    Returns
    -------
    Tuple[np.ndarray, Dict[str, int]]
        Full R² matrix and mapping from variables to best latent dimensions
    """
    n_dims = biscuit_vars.shape[1]
    full_R2 = np.empty((n_dims, len(var_names)))
    
    print(f"Computing R² matrix for {n_dims} latent dims and {len(var_names)} variables...")
    
    for d in range(n_dims):
        if d % 10 == 0:
            print(f"  Processing latent dimension {d+1}/{n_dims}")
        
        z = biscuit_vars[:, d]
        for j, v in enumerate(var_names):
            if v in sim_var_name_to_data:
                y = sim_var_name_to_data[v].astype(float)
                full_R2[d, j] = compute_r2(z, y)
            else:
                full_R2[d, j] = np.nan
    
    # Find best latent dimension for each variable
    var_to_lat = {}
    for j, v in enumerate(var_names):
        if not np.isnan(full_R2[:, j]).all():
            var_to_lat[v] = np.nanargmax(full_R2[:, j])
    
    return full_R2, var_to_lat
