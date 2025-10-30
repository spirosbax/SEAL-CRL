#!/bin/bash

#SBATCH --partition=gpu_a100
#SBATCH --job-name=create_env
#SBATCH --time=04:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --output=outfiles/create_env_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

# Set working directory to repository root
cd ~/causality/

# Create environment from scratch
conda create -n cbmbiscuit python=3.9 -y

# Activate environment
source activate cbmbiscuit

# Install specific PyTorch versions with CUDA 11.8
echo "Installing PyTorch 2.4.1 with CUDA 11.8..."
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=11.8 -c pytorch -c nvidia --yes
echo "Installing other dependencies..."
conda install lightning -c conda-forge

# Test PyTorch installation
echo ""
echo "=== Testing PyTorch Installation ==="
echo "Activating environment and testing PyTorch..."

# Create a temporary Python script to test PyTorch
cat > test_pytorch.py << 'EOF'
import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    
    # Test CUDA tensor creation
    x = torch.randn(3, 3).cuda()
    print(f"CUDA tensor created successfully: {x.device}")
else:
    print("CUDA is not available")

# Test basic PyTorch operations
x = torch.randn(2, 3)
y = torch.randn(2, 3)
z = x + y
print(f"Basic PyTorch operation successful: {z.shape}")

print("PyTorch installation test completed successfully!")
EOF

# Run the test script
python test_pytorch.py

# Clean up test file
rm test_pytorch.py

echo ""
echo "=== Environment Setup Complete ==="
echo "To activate the environment in the future, run:"
echo "conda activate cbmbiscuit"
echo ""
echo "To test PyTorch again, run:"
echo "conda activate cbmbiscuit && python -c \"import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')\""

