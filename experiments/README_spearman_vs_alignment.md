# Spearman vs Alignment Methods Comparison

This experiment compares different methods for aligning BISCUIT latent variables with ground truth causal concepts from the iTHOR dataset.

## Methods Compared

1. **Baseline (Hungarian LR)**: 
   - Computes Spearman correlation between latent dimensions and concepts on training data
   - Uses Hungarian algorithm to find optimal 1-to-1 assignment based on absolute correlation
   - Fits individual Linear Regression models for each concept using assigned dimension

2. **Linear**: 
   - Uses `FeaturePermutationEstimator` with Lasso regularization
   - Learns linear transformations to align latent variables with concepts
   - Single joint optimization across all concepts

3. **Kernel RBF**: 
   - Uses `KernelizedPermutationEstimator` with RBF kernel
   - Applies kernel feature maps before linear alignment
   - Uses Nyström approximation for computational efficiency

4. **Kernel Laplacian**: 
   - Uses `KernelizedPermutationEstimator` with Laplacian kernel
   - Alternative kernel choice for non-linear feature mapping

## Key Features

- **Sample Efficiency Analysis**: Evaluates performance across different training set sizes (5, 10, 20, 40, 80, 160, 320, 640, 1280 samples)
- **Multiple Seeds**: Runs experiments with multiple random seeds for robust statistics
- **Publication-Ready Plots**: Creates journal-quality grid plots and summary visualizations
- **Comprehensive Evaluation**: Tests on 12 iTHOR concepts including object states and interactions

## Usage

### Basic Usage
```bash
conda activate cbmbiscuit
python experiments/spearman_vs_alignment.py
```

### Custom Configuration
```bash
python experiments/spearman_vs_alignment.py \
    --data_dir src/data/ithor \
    --split test \
    --models baseline linear kernel_rbf \
    --seeds 0 1 2 3 4 \
    --n_train_values 10 20 50 100 200 500 1000 \
    --alpha 0.001 \
    --output_dir results/my_comparison \
    --show_ranges \
    --create_learning_curves
```

### Key Arguments

- `--models`: Choose which methods to compare (baseline, linear, kernel_rbf, kernel_laplacian)
- `--seeds`: Random seeds for multiple runs (default: [0, 1, 2])
- `--n_train_values`: Training set sizes to evaluate
- `--alpha`: Regularization parameter for permutation estimators (default: 0.002154)
- `--kernel_gamma`: Gamma parameter for kernel methods (default: 0.5)
- `--show_ranges`: Show min/max ranges across seeds in plots
- `--create_learning_curves`: Generate detailed learning curve plots

## Output Files

The experiment generates several output files:

1. **comparison_results.csv**: Raw results with columns [model, seed, concept, n_train, r2]
2. **spearman_vs_alignment_grid.pdf/.png**: Main grid plot comparing all methods
3. **model_summary_comparison.pdf**: Summary bar chart of overall performance
4. **learning_curves.pdf**: Detailed learning curves (if requested)

## Expected Results

Based on the original analysis, you should expect:

- **Baseline**: Simple but effective, especially with limited training data
- **Linear**: Improved performance with joint optimization across concepts
- **Kernel RBF**: Best performance due to non-linear feature mapping
- **Performance scales**: Better methods show more improvement with larger training sets

## Dependencies

The experiment requires:
- Standard scientific Python stack (numpy, pandas, matplotlib, sklearn)
- Spearman correlation (scipy.stats)
- Hungarian algorithm (scipy.optimize)
- Permutation estimators (external/sample-efficient-learning-of-concepts)

## Implementation Notes

- **Data Splits**: Uses reproducible train/test splits with configurable test fraction
- **Standardization**: Features are standardized for baseline method
- **Jitter**: Small noise added to constant targets to avoid numerical issues
- **Memory Efficiency**: Processes data in batches and releases memory appropriately
- **Error Handling**: Robust handling of edge cases (constant targets, insufficient data)

## Customization

The experiment is designed to be easily customizable:

- **New Models**: Add new methods in `src/utils/model_evaluation.py`
- **Different Concepts**: Modify concept order in `create_concept_order()`
- **Alternative Kernels**: Extend kernel options in model configuration
- **Custom Visualizations**: Add new plot types in `src/utils/visualization.py`
