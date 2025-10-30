#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --job-name=convert_encodings
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=outfiles/convert_encodings_%A.out

module purge
module load 2024
module load Anaconda3/2024.06-1

# Set working directory to repository root
cd ~/causality/

# Activate environment
source activate cbmbiscuit

# Run the conversion script
python src/utils/convert_encodings.py --data_dir /home/sbaxevanakis/causality/src/data/ithor --checkpoint /home/sbaxevanakis/causality/external/BISCUIT/pretrained_models/BISCUITNF_iTHOR/BISCUITNF_40l_64hid.ckpt --autoencoder_checkpoint /home/sbaxevanakis/causality/external/BISCUIT/pretrained_models/AE_iTHOR/AE_40l_64hid.ckpt --batch_size 2000 --device cuda
