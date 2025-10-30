# Causality Research with BISCUIT-VAE

A research codebase for causal representation learning and analysis on iTHOR data using BISCUIT-VAE encodings.

## Installation

Choose between **UV (modern, fast)** or **Conda (traditional)**:

### Option A: UV (Recommended - Modern & Fast)

UV is a blazing-fast Python package manager (10-100x faster than pip/conda).

**1) Clone repositories**
```bash
git clone https://github.com/spirosbax/causality.git
cd causality

# External dependencies
mkdir -p external
cd external
git clone https://github.com/phlippe/BISCUIT
git clone https://github.com/spirosbax/sample-efficient-learning-of-concepts
cd ..
```

**2) Install UV**
```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

**3) Create environment and install**
```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install causality package (CPU-only PyTorch)
uv pip install -e .

# OR install with CUDA support (requires system CUDA 11.8)
uv pip install -e ".[cuda]"

# Install external dependencies
cd external/BISCUIT
# Create minimal setup.py for BISCUIT
python - << 'PY'
from setuptools import setup, find_packages
setup(name='biscuit', version='0.1', packages=find_packages())
PY
uv pip install -e .
cd ../..

cd external/sample-efficient-learning-of-concepts
uv pip install -e .
cd ../..
```

**4) Verify installation**
```bash
python -c "from causality.data.ithor_loader import load_causality_data; print('✓ Causality package installed')"
python -c "import biscuit; print('✓ BISCUIT installed')"
python -c "from permutation_estimator.FeaturePermutationEstimator import FeaturePermutationEstimator; print('✓ Permutation estimators installed')"
```

---

### Option B: Conda (Traditional)

**1) Clone repositories**
```bash
git clone https://github.com/spirosbax/causality.git
cd causality

# External dependencies
mkdir -p external
cd external
git clone https://github.com/phlippe/BISCUIT
git clone https://github.com/spirosbax/sample-efficient-learning-of-concepts
cd ..
```

**2) Create environment**
```bash
# Use environment.yml for Linux or environment_mac.yml for macOS
conda env create -f environment.yml  # or environment_mac.yml
conda activate cbmbiscuit
```

**3) Install packages**
```bash
# BISCUIT
cd external/BISCUIT
python - << 'PY'
from setuptools import setup, find_packages
setup(name='biscuit', version='0.1', packages=find_packages())
PY
pip install -e .
cd ../..

# Sample-efficient learning of concepts
cd external/sample-efficient-learning-of-concepts
pip install -e .
cd ../..

# Main causality package
pip install -e .
```

**4) Verify installation**
```bash
python -c "from causality.data.ithor_loader import load_causality_data; print('✓ Causality package installed')"
python -c "import biscuit; print('✓ BISCUIT installed')"
python -c "from permutation_estimator.FeaturePermutationEstimator import FeaturePermutationEstimator; print('✓ Permutation estimators installed')"
```

---

### UV vs Conda

**UV advantages:**
- ⚡ 10-100x faster package resolution
- 🔒 Reproducible with `uv.lock`
- 🎯 Modern Python standard (PEP 517/518)
- 💾 Efficient disk usage

**Conda advantages:**
- 🔧 Pre-compiled CUDA binaries
- 📦 System library management
- 🏢 Widely used in research

## Data setup

### iTHOR Dataset

The iTHOR dataset from the BISCUIT paper is required for running experiments.

- **Source**: [Zenodo - BISCUIT Datasets](https://zenodo.org/records/8027138)
- **DOI**: `10.5281/zenodo.8027138`
- **File**: `ithor.zip` (30.8 GB)
- **Default path**: `src/data/ithor`

**Download and setup**:
```bash
# Download ithor.zip from Zenodo (30.8 GB)
# Extract to src/data/ithor/
unzip ithor.zip -d src/data/ithor/

# Or create a symlink if you have the data elsewhere
ln -s /absolute/path/to/ithor src/data/ithor
```

**Expected structure**:
```
src/data/ithor/
├── train/
├── val/
├── test/
├── train_indep/
├── val_indep/
└── test_indep/
```

## Encoding conversion (VAE → causal)

Convert VAE encodings to causal encodings using the BISCUIT normalizing flow.

- Purpose: Transform 40-D VAE latents to causal latents for downstream analyses.
- Script: `src/utils/convert_encodings.py`
- Inputs: `{split}_seq_{id}_encodings.npz`
- Outputs: `{split}_seq_{id}_nf_encodings.npz` with key `encodings`

### Pretrained Models

The BISCUIT pretrained models are included in the external/BISCUIT repository:

- **Normalizing Flow checkpoint**: `external/BISCUIT/pretrained_models/BISCUITNF_iTHOR/BISCUITNF_40l_64hid.ckpt`
- **Autoencoder checkpoint**: `external/BISCUIT/pretrained_models/AE_iTHOR/AE_40l_64hid.ckpt`

If these are not present, download them from the [BISCUIT repository](https://github.com/phlippe/BISCUIT).

### Basic usage:
```bash
conda activate cbmbiscuit
python src/utils/convert_encodings.py \
  --data_dir src/data/ithor \
  --batch_size 2000 \
  --splits train val test
```

### Key arguments:
- `--data_dir`: Path to iTHOR data (default: `src/data/ithor`)
- `--checkpoint`: NF checkpoint (default: auto-detected from BISCUIT pretrained_models)
- `--autoencoder_checkpoint`: AE checkpoint (default: auto-detected from BISCUIT pretrained_models)
- `--batch_size`: Processing batch size (default 2000)
- `--splits`: One or more of `train val test`
- `--device`: `cuda` or `cpu` (auto if omitted)

HPC/SLURM:
```bash
sbatch slurm_jobs/convert_encodings.sh
```
Monitor: `tail -f outfiles/convert_encodings_<job_id>.out`

Output structure:
```
src/data/ithor/
├── train/*_encodings.npz      # VAE encodings
├── train/*_nf_encodings.npz   # causal encodings (new)
├── val/...
└── test/...
```

## Scripts overview

Below are the main analysis scripts with purpose, key arguments, example usage, and expected outputs.

### 1) Robustness analysis (grouped, capacitated) — `experiments/robustness_analysis_groups.py`
- Purpose: Evaluate assignment and prediction stability under perturbations across multiple concepts with capacitated matching. Produces a publication-ready 2x4 grid (row 1: Jaccard; row 2: prediction correlation).
- Models: baseline (Spearman+Hungarian+LR), linear (lasso), logistic, kernel_rbf.
- Stove predicate handling: If multiple stove predicates are requested, runs separate subruns (each: one stove predicate + `Microwave_d8b935e4_open` control) to avoid latent competition; results are merged.
- Key args:
  - `--predicates`: Aggregate predicates and/or raw concepts. Defaults to `all_stoves_on at_least_three_stoves_on at_least_two_stoves_on egg_intact`.
  - `--models`: `baseline linear logistic kernel_rbf` (subset allowed)
  - `--n_train_values`: Training sizes (log-spaced recommended)
  - `--seeds`: Random seeds
  - `--n_trials`: Noisy refit trials per configuration
  - `--sigma`: Perturbation scale
  - `--output_dir`: Output directory
- Example:
```bash
conda activate cbmbiscuit
python experiments/robustness_analysis_groups.py \
  --output_dir results/paper_plots/robustness_groups
```
- Outputs in the output_dir:
  - `robust_groups_summary.csv`: Global Jaccard summary per model/N
  - `robust_groups_perconcept.csv`: Per-concept Jaccard and prediction correlation
  - `robust_groups_global_jaccard.(png|pdf)`
  - `robust_groups_perconcept_jaccard.(png|pdf)`
  - `robust_groups_perconcept_corr.(png|pdf)`
  - `robust_groups_grid_2x4.(png|pdf)`: Publication-ready grid (All Stoves On, At Least 3, At Least 2, Egg Intact)

### 2) Spearman vs alignment comparison — `experiments/spearman_vs_alignment.py`
- Purpose: Compare baseline vs permutation estimators across concepts and N, and plot R² grids.
- Key args:
  - `--variable_types`: `binary|continuous|all`
  - `--models`: `baseline linear logistic kernel_rbf [kernel_laplacian]`
  - `--alpha`, `--kernel_gamma`, `--n_train_values`, `--seeds`
  - `--output_dir`: Base output directory
- Example:
```bash
conda activate cbmbiscuit
python experiments/spearman_vs_alignment.py \
  --variable_types binary \
  --models baseline linear logistic kernel_rbf \
  --output_dir results/spearman_vs_alignment
```
- Outputs:
  - `comparison_results.csv`: R² per concept/model/N
  - `spearman_vs_alignment_<type>_grid.(pdf|png)`
  - Text summary (stdout) of overall and best results

### 3) Causal grouping analysis — `experiments/causal_grouping_analysis.py`
- Purpose: Analyze grouped alignment and capacity effects for stove and egg predicates with a control concept.
- Key args:
  - `--predicates`: One or more aggregate predicates
  - `--other_concept`: Control raw concept (e.g., `Microwave_d8b935e4_open`)
  - `--models`, `--n_train_values`, `--seeds`, `--output_dir`
- Example:
```bash
conda activate cbmbiscuit
python experiments/causal_grouping_analysis.py \
  --predicates all_stoves_on at_least_three_stoves_on \
  --other_concept Microwave_d8b935e4_open \
  --output_dir results/causal_grouping_multi
```
- Outputs: CSV summaries and plots of grouped alignment performance.

### 4) Predicate class counts — `experiments/predicate_class_counts.py`
- Purpose: Compute and plot class balance for key binary concepts.
- Key args: `--split`, `--output_dir`
- Example:
```bash
conda activate cbmbiscuit
python experiments/predicate_class_counts.py --output_dir results/class_counts
```
- Outputs:
  - Per-concept PNGs in `results/class_counts/`

### 5) Kernel/linear estimator utilities — `src/utils/estimator_models.py`
- Purpose: Shared wrappers/utilities for permutation estimators used across experiments.
- Not directly executable; used by experiments.

## Tips

- macOS BLAS stability: if you see a bus error, limit threads when running Python:
```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
python -X faulthandler experiments/robustness_analysis_groups.py --help
```

- Caching: data loader caches aligned arrays for speed; use `--no_cache` to force reload.

- Reproducibility: set `--split_seed` and pass explicit `--seeds`.
