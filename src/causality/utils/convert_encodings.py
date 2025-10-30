#!/usr/bin/env python3
"""
Script to convert VAE encodings to causal encodings for the iTHOR dataset.

This script:
1. Loads all VAE encoding files from train/val/test splits
2. Uses the BISCUIT model's flow.forward() to convert VAE encodings to causal encodings
3. Saves the causal encodings in the same format and structure
4. Processes in large batches for efficiency
"""

import os
import sys
import torch
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm
import argparse

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'external', 'BISCUIT'))

from models.biscuit_nf import BISCUITNF


def load_model(checkpoint_path, autoencoder_checkpoint, device):
    """Load a trained BISCUIT-VAE model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    model = BISCUITNF.load_from_checkpoint(
        checkpoint_path,
        autoencoder_checkpoint=autoencoder_checkpoint
    )
    model.to(device)
    model.freeze()
    model.eval()
    
    print("Model loaded successfully")
    return model


def get_encoding_files(data_dir, splits=['train', 'val', 'test']):
    """Get all VAE encoding files from specified splits"""
    all_files = []
    
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory {split_dir} not found, skipping...")
            continue
            
        # Find all VAE encoding files
        pattern = os.path.join(split_dir, f'{split}_seq_*_encodings.npz')
        files = sorted(glob(pattern))
        
        print(f"Found {len(files)} VAE encoding files in {split} split")
        all_files.extend(files)
    
    return all_files


def load_vae_encodings_batch(files, batch_size=1000):
    """Load VAE encodings from multiple files and batch them efficiently"""
    all_encodings = []
    file_info = []  # Track which encodings belong to which files
    
    for file_path in files:
        try:
            data = np.load(file_path, allow_pickle=True)
            encodings = data['encodings']  # Shape: (seq_len, 40)
            
            # Store info about this file's encodings
            start_idx = len(all_encodings)
            end_idx = start_idx + len(encodings)
            file_info.append({
                'file_path': file_path,
                'start_idx': start_idx, 
                'end_idx': end_idx,
                'original_shape': encodings.shape
            })
            
            # Add to batch
            all_encodings.extend(encodings)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    # Convert to numpy array and create batches
    all_encodings = np.array(all_encodings, dtype=np.float32)
    print(f"Loaded {len(all_encodings)} total frames for processing")
    
    # Create batches
    batches = []
    for i in range(0, len(all_encodings), batch_size):
        end_idx = min(i + batch_size, len(all_encodings))
        batches.append((i, end_idx, all_encodings[i:end_idx]))
    
    return batches, file_info, all_encodings


def convert_batch_to_causal(model, vae_batch, device):
    """Convert a batch of VAE encodings to causal encodings"""
    # Convert to torch tensor and move to device
    vae_tensor = torch.from_numpy(vae_batch).float().to(device)
    
    # Convert using the flow model
    with torch.no_grad():
        causal_encodings = model.flow.forward(vae_tensor)[0]
    
    # Convert back to numpy
    return causal_encodings.cpu().numpy()


def save_causal_encodings(file_info, all_causal_encodings, output_suffix="_nf_encodings"):
    """Save causal encodings to files with the same structure as VAE encodings"""
    
    for info in file_info:
        # Extract causal encodings for this file
        start_idx = info['start_idx']
        end_idx = info['end_idx']
        causal_data = all_causal_encodings[start_idx:end_idx]
        
        # Create output file path
        vae_file_path = info['file_path']
        output_file_path = vae_file_path.replace('_encodings.npz', f'{output_suffix}.npz')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # Save in the same format as VAE encodings
        np.savez_compressed(output_file_path, encodings=causal_data)
        
        print(f"Saved {causal_data.shape} causal encodings to {output_file_path}")


def convert_encodings(data_dir, checkpoint_path, autoencoder_checkpoint, 
                     batch_size=2000, device=None, splits=['train', 'val', 'test']):
    """Main function to convert all VAE encodings to causal encodings"""
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Processing splits: {splits}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Load the model
    model = load_model(checkpoint_path, autoencoder_checkpoint, device)
    
    # Get all VAE encoding files
    encoding_files = get_encoding_files(data_dir, splits)
    print(f"Total files to process: {len(encoding_files)}")
    print()
    
    # Process in chunks to manage memory
    chunk_size = 50  # Process 50 files at a time
    
    for chunk_start in range(0, len(encoding_files), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(encoding_files))
        chunk_files = encoding_files[chunk_start:chunk_end]
        
        print(f"Processing files {chunk_start+1}-{chunk_end} of {len(encoding_files)}")
        
        # Load VAE encodings for this chunk
        print("Loading VAE encodings...")
        batches, file_info, all_vae_encodings = load_vae_encodings_batch(chunk_files, batch_size)
        
        # Convert to causal encodings
        print(f"Converting {len(all_vae_encodings)} frames in {len(batches)} batches...")
        all_causal_encodings = np.zeros_like(all_vae_encodings)
        
        for batch_idx, (start_idx, end_idx, vae_batch) in enumerate(tqdm(batches, desc="Converting batches")):
            causal_batch = convert_batch_to_causal(model, vae_batch, device)
            all_causal_encodings[start_idx:end_idx] = causal_batch
        
        # Save causal encodings
        print("Saving causal encodings...")
        save_causal_encodings(file_info, all_causal_encodings, "_nf_encodings")
        
        print(f"Completed chunk {chunk_start//chunk_size + 1}")
        print()
    
    print("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(description="Convert VAE encodings to causal encodings")
    parser.add_argument("--data_dir", type=str, default="src/data/ithor",
                       help="Directory containing the iTHOR data")
    parser.add_argument("--checkpoint", type=str, 
                       default="external/BISCUIT/pretrained_models/BISCUITNF_iTHOR/BISCUITNF_40l_64hid.ckpt",
                       help="Path to BISCUIT model checkpoint")
    parser.add_argument("--autoencoder_checkpoint", type=str,
                       default="external/BISCUIT/pretrained_models/AE_iTHOR/AE_40l_64hid.ckpt", 
                       help="Path to autoencoder checkpoint")
    parser.add_argument("--batch_size", type=int, default=2000,
                       help="Batch size for processing (default: 2000)")
    parser.add_argument("--splits", type=str, nargs='+', default=['train', 'val', 'test'],
                       help="Splits to process (default: train val test)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu, default: auto-detect)")
    
    args = parser.parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = None
    
    convert_encodings(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        batch_size=args.batch_size,
        device=device,
        splits=args.splits
    )


if __name__ == "__main__":
    main()
