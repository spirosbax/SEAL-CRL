# Causal Grouping Analysis

This experiment evaluates different models on aggregated causal predicates, comparing baseline models (Linear/Logistic Regression, MLP) with advanced permutation estimators (Linear/Logistic/Kernel with group-lasso).

## Features

### Model Types
- **Baseline Models**: LinearRegression, LogisticRegression, MLPClassifier with various hidden layer sizes
- **Permutation Estimators**: Linear, Logistic, and Kernel RBF estimators with capacitated matching for N→1 latent alignment

### Class Balancing
For highly imbalanced predicates (like "all stoves on"), the experiment supports automatic class balancing:
- **Undersamples majority class** to achieve closer to 50-50 class distribution
- **Preserves all minority samples** while randomly sampling from majority class
- **Maintains stratification** in train/test splits after balancing

### Visualizations
- **Baseline comparison plots**: AUC and R² curves across training set sizes
- **Estimator comparison plots**: R² performance across concepts
- **Latent assignment heatmaps**: Shows which latent dimensions are assigned to each concept across different training sizes and seeds

## Usage

### Basic Usage
```bash
# Run both baseline and estimator analyses
python experiments/causal_grouping_analysis.py --mode both --predicate all_stoves_on

# Run only baseline analysis
python experiments/causal_grouping_analysis.py --mode baseline --predicate all_stoves_on

# Run only estimator analysis (requires second concept)
python experiments/causal_grouping_analysis.py --mode estimators --predicate all_stoves_on --other_concept Microwave_d8b935e4_open
```

### Class Balancing
```bash
# Enable class balancing for 50-50 split
python experiments/causal_grouping_analysis.py --balance_classes --predicate all_stoves_on

# Custom target ratio (e.g., 40-60 split)
python experiments/causal_grouping_analysis.py --balance_classes --target_ratio 0.4 --predicate all_stoves_on

# With verbose output to see balancing details
python experiments/causal_grouping_analysis.py --balance_classes --verbose --predicate all_stoves_on
```

### Advanced Options
```bash
# Custom hyperparameters
python experiments/causal_grouping_analysis.py \
    --balance_classes \
    --alpha 0.001 \
    --kernel_gamma 0.3 \
    --mlp_hidden_sizes 4 8 16 \
    --seeds 0 1 2 3 4

# Subset of training sizes for faster testing
python experiments/causal_grouping_analysis.py \
    --balance_classes \
    --n_train_values 5 10 20 40 80 \
    --verbose
```

## Output Files

The experiment generates several output files in the specified output directory:

### CSV Results
- `baseline_results.csv`: Raw results for all baseline models across seeds and training sizes
- `estimator_results.csv`: Raw results for permutation estimators across seeds and training sizes

### Visualizations
- `baseline_comparison.pdf`: Three-panel plot showing AUC and R² curves for baseline models
- `estimator_comparison.pdf`: R² comparison across concepts for permutation estimators
- `latent_assignment_heatmap.pdf`: Heatmap showing latent dimension assignments across training sizes and seeds

### Console Output
- Class distribution before/after balancing
- Logistic regression weight analysis (commented out by default)
- Summary statistics for all models
- Progress information (with `--verbose`)

## Parameters

### Data Parameters
- `--data_dir`: Directory containing iTHOR data (default: `src/data/ithor`)
- `--split`: Dataset split to use (default: `test`)
- `--max_files`: Maximum files to load (default: None for all)
- `--output_dir`: Results directory (default: `results/causal_grouping`)

### Experiment Parameters
- `--mode`: Analysis mode (`baseline`, `estimators`, `both`) (default: `both`)
- `--predicate`: Aggregate predicate to evaluate (default: `all_stoves_on`)
- `--other_concept`: Second concept for N→1 alignment (default: `Microwave_d8b935e4_open`)

### Model Parameters
- `--mlp_hidden_sizes`: MLP hidden layer sizes (default: `[2, 4, 8]`)
- `--alpha`: Regularization for permutation estimators (default: `2.154e-3`)
- `--kernel_gamma`: RBF kernel gamma (default: `0.5`)
- `--max_nystrom`: Max Nyström components (default: `64`)

### Training Parameters
- `--seeds`: Random seeds (default: `[0, 1, 2]`)
- `--n_train_values`: Training set sizes (default: `[5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240]`)
- `--test_fraction`: Test set fraction (default: `0.2`)
- `--split_seed`: Random seed for train/test split (default: `0`)

### Class Balancing Parameters
- `--balance_classes`: Enable class balancing (flag)
- `--target_ratio`: Target minority class ratio (default: `0.5` for 50-50 split)

### Visualization Parameters
- `--dpi`: Figure DPI (default: `600`)
- `--verbose`: Detailed progress output (flag)

## Class Balancing Details

When `--balance_classes` is enabled:

1. **Original distribution is analyzed** and reported
2. **Minority class samples are all kept** (no undersampling of minority)
3. **Majority class is undersampled** to achieve target ratio
4. **Balanced subset is used** for both train/test split and all analyses
5. **Stratification is maintained** during train/test splitting

### Example Output
```
Original class distribution:
  Class 0: 4500 samples (90.0%)
  Class 1: 500 samples (10.0%)

Balanced class distribution:
  Class 0: 500 samples (50.0%)
  Class 1: 500 samples (50.0%)
Total samples: 1000 (reduced from 5000)
```

This ensures that models are evaluated on meaningful, balanced data rather than learning to simply predict the majority class.
