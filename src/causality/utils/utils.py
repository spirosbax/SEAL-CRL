import re
import os
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


# ---------- Pretty names ----------
CONCEPT_NAME_MAP = {
    "StoveKnob_cf670576_on": "Stove Back-Right",
    "StoveKnob_38c1dbc2_on": "Stove Front-Right",
    "StoveKnob_c8955f66_on": "Stove Back-Left",
    "StoveKnob_690d0d5d_on": "Stove Front-Left",
}
_SUFFIX_MAP = {
    "on": "On",
    "off": "Off",
    "open": "Open",
    "closed": "Closed",
    "pickedup": "PickedUp",
    "broken": "Broken",
    "cooked": "Cooked",
}
_HEX = re.compile(r"^[0-9a-fA-F]+$")


def pretty_concept_name(raw: str, overrides=CONCEPT_NAME_MAP):
    if raw in overrides:
        return overrides[raw]
    parts = raw.split("_")
    if len(parts) >= 3 and _HEX.match(parts[1]):
        base, rest = parts[0], parts[2:]
    else:
        base, rest = parts[0], parts[1:]
    base = "Stove" if base == "StoveKnob" else base
    rest_pretty = " ".join(_SUFFIX_MAP.get(x.lower(), x.capitalize()) for x in rest)
    return f"{base} {rest_pretty}".strip()


def compute_r2(x, y):
    """
    Compute R² score with NaN safety checks and robust error handling.
    
    Args:
        x, y: Arrays of predictor and target values
        
    Returns:
        float: R² score (≥0), or 0.0 if computation fails
    """
    # Handle NaN values by removing them pair-wise
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0  # Not enough valid data points
    
    x_clean = x[mask]
    y_clean = y[mask]
    
    # Check for constant target values
    if len(np.unique(y_clean)) == 1:
        return 0.0
    
    try:
        x_clean = x_clean.reshape(-1, 1)
        y_clean = y_clean.reshape(-1, 1)
        model = LinearRegression().fit(x_clean, y_clean)
        return max(0, r2_score(y_clean, model.predict(x_clean)))
    except Exception:
        return 0.0


def setup_output_dir(output_dir):
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
