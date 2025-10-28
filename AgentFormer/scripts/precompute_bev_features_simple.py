#!/usr/bin/env python
"""
Simple BEV feature precomputation - sequential loading with batched inference.

This version uses simple sequential data loading (no DataLoader workers)
but still batches the GPU inference for speed.

Usage:
    python scripts/precompute_bev_features_simple.py --split train --gpu 0 --batch_size 16
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
import time
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bevdepth.datasets.nusc_det_dataset import NuscDetDataset
from model.bev.base_lss_fpn import BaseLSSFPN
from utils.config import Config


def load_bev_encoder(cfg, device):
    """Load and freeze BEV encoder"""
    print("Loading BEV encoder...")
    bev_encoder = BaseLSSFPN(**cfg.bev_encoder)

    for param in bev_encoder.parameters():
        param.requires_grad = False

    bev_encoder.eval()
    bev_encoder.to(device)

    print(f"BEV encoder loaded and frozen. Output channels: {bev_encoder.output_channels}")
    return bev_encoder


def load_used_tokens(split, token_file=None):
    """Load the list of sample tokens that are actually used"""
    if token_file is None:
        token_file = f'bev_features/used_tokens_{split}.json'

    if not os.path.exists(token_file):
        raise FileNotFoundError(
            f"Token file not found: {token_file}\n"
            f"Please run: python scripts/collect_used_samples.py --split {split}"
        )

    print(f"Loading used tokens from: {token_file}")
    with open(token_file, 'r') as f:
        data = json.load(f)

    tokens = data['tokens']
    print(f"Loaded {len(tokens)} tokens for {split} split")
    return set(tokens)


def create_dataset(cfg, split):
    """Create BEV dataset for the given split"""
    data_root = cfg.data_root_nuscenes_pred

    if split == 'train':
        info_paths = os.path.join(data_root, 'nuscenes_infos_train_subset.pkl')
    elif split == 'val':
        info_paths = os.path.join(data_root, 'nuscenes_infos_val_subset.pkl')
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(info_paths):
        raise FileNotFoundError(f"Dataset file not found: {info_paths}")

    print(f"Creating dataset from: {info_paths}")

    # Force single camera for precomputation to save memory
    ida_aug_conf_single_cam = cfg.ida_aug_conf.copy()
    ida_aug_conf_single_cam['cams'] = ['CAM_FRONT']  # Force single camera
    ida_aug_conf_single_cam['Ncams'] = 1

    dataset = NuscDetDataset(
        ida_aug_conf=ida_aug_conf_single_cam,
        bda_aug_conf=cfg.bda_aug_conf,
        classes=cfg.object_classes,
        data_root='nuscenes',
        info_paths=info_paths,
        is_train=False,
        img_conf=cfg.get('img_conf', {}),
        num_sweeps=1,
        return_depth=True,
    )

    print(f"Dataset created with {len(dataset)} samples")
    return dataset, info_paths


def filter_dataset_indices(info_paths, used_tokens):
    """Get indices of samples we actually need"""
    import pickle

    print(f"Reading tokens from pickle: {info_paths}")
    with open(info_paths, 'rb') as f:
        data_infos = pickle.load(f)

    token_to_idx = {}
    for idx, info in enumerate(data_infos):
        sample_token = info['sample_token']
        token_to_idx[sample_token] = idx

    used_indices = []
    for token in used_tokens:
        if token in token_to_idx:
            used_indices.append(token_to_idx[token])

    print(f"Filtered to {len(used_indices)} samples")
    return used_indices


def extract_features_simple(bev_encoder, dataset, indices, output_dir, device, batch_size=16):
    """Extract BEV features with simple sequential loading + batched inference"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting BEV features to: {output_dir}")
    print(f"Processing {len(indices)} samples in batches of {batch_size}")
    print(f"Method: Sequential loading + GPU batched inference\n")

    total_size_mb = 0
    extracted = 0
    skipped = 0
    start_time = time.time()

    # Process in batches
    num_batches = (len(indices) + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            batch_start_time = time.time()

            # Get batch indices
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(indices))
            batch_indices = indices[batch_start:batch_end]

            # Load data sequentially (this shows progress!)
            batch_data = []
            batch_tokens = []
            batch_skip = []

            print(f"\n  Batch {batch_idx+1}/{num_batches}: Loading {len(batch_indices)} samples...")
            for idx in batch_indices:
                try:
                    data = dataset[idx]
                    img_metas = data[7]
                    sample_token = img_metas['token']

                    output_file = output_dir / f"{sample_token}.pt"
                    if output_file.exists():
                        batch_skip.append(True)
                        skipped += 1
                        continue

                    batch_data.append(data)
                    batch_tokens.append(sample_token)
                    batch_skip.append(False)
                except Exception as e:
                    print(f"    Error loading sample {idx}: {e}")
                    batch_skip.append(True)
                    continue

            if len(batch_data) == 0:
                print(f"  All samples in batch already exist, skipping...")
                continue

            print(f"  Loaded {len(batch_data)} samples, running BEV inference...")

            # Stack into batch
            try:
                sweep_imgs = torch.stack([item[0] for item in batch_data], dim=0).to(device, non_blocking=True)
                sensor2ego_mats = torch.stack([item[1] for item in batch_data], dim=0).to(device, non_blocking=True)
                intrins = torch.stack([item[2] for item in batch_data], dim=0).to(device, non_blocking=True)
                ida_mats = torch.stack([item[3] for item in batch_data], dim=0).to(device, non_blocking=True)
                sensor2sensor_mats = torch.stack([item[4] for item in batch_data], dim=0).to(device, non_blocking=True)
                bda_mats = torch.stack([item[5] for item in batch_data], dim=0).to(device, non_blocking=True)
                timestamps = torch.stack([item[6] for item in batch_data], dim=0).to(device, non_blocking=True)

                # Add num_sweeps dimension
                if len(sweep_imgs.shape) == 5:
                    sweep_imgs = sweep_imgs.unsqueeze(1)

                mats_dict = {
                    'sensor2ego_mats': sensor2ego_mats,
                    'intrin_mats': intrins,
                    'ida_mats': ida_mats,
                    'sensor2sensor_mats': sensor2sensor_mats,
                    'bda_mat': bda_mats,
                }

                for k, v in mats_dict.items():
                    if k != 'bda_mat' and len(v.shape) == 4:
                        mats_dict[k] = v.unsqueeze(1)

                if len(timestamps.shape) == 2:
                    timestamps = timestamps.unsqueeze(1)

                # Run BEV encoder
                print(f"  sweep_imgs shape: {sweep_imgs.shape}")

                t1 = time.time()
                bev_feature_maps = bev_encoder(sweep_imgs, mats_dict, timestamps=timestamps, is_return_depth=False)
                t2 = time.time()
                print(f"  BEV inference: {t2-t1:.1f}s")

                bev_feature_maps = bev_feature_maps.cpu()
                t3 = time.time()
                print(f"  GPU->CPU transfer: {t3-t2:.1f}s")

                # Save features
                for i, sample_token in enumerate(batch_tokens):
                    output_file = output_dir / f"{sample_token}.pt"
                    torch.save(bev_feature_maps[i], output_file)

                    file_size_mb = output_file.stat().st_size / (1024 * 1024)
                    total_size_mb += file_size_mb
                    extracted += 1

                t4 = time.time()
                print(f"  Saving {len(batch_tokens)} files: {t4-t3:.1f}s")

                batch_time = time.time() - batch_start_time
                samples_per_sec = len(batch_tokens) / batch_time
                print(f"  Batch completed in {batch_time:.1f}s ({samples_per_sec:.2f} samples/s)")

            except Exception as e:
                print(f"  ERROR processing batch: {e}")
                import traceback
                traceback.print_exc()
                continue

    elapsed_time = time.time() - start_time
    samples_per_sec = extracted / elapsed_time if elapsed_time > 0 else 0

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"  Extracted: {extracted} samples")
    print(f"  Skipped: {skipped} samples")
    print(f"  Total time: {elapsed_time/60:.1f} minutes")
    print(f"  Throughput: {samples_per_sec:.2f} samples/second")
    print(f"  Total disk usage: {total_size_mb/1024:.2f} GB")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Simple BEV precomputation (sequential loading)')
    parser.add_argument('--cfg', type=str, default='nuscenes_5sample_agentformer_pre_bev')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='bev_features')
    parser.add_argument('--token_file', type=str, default=None)

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.2f} GB")

    cfg = Config(args.cfg, tmp=False, create_dirs=False)
    used_tokens = load_used_tokens(args.split, args.token_file)
    dataset, info_paths = create_dataset(cfg, args.split)
    used_indices = filter_dataset_indices(info_paths, used_tokens)
    bev_encoder = load_bev_encoder(cfg, device)

    output_dir = os.path.join(args.output_dir, args.split)
    extract_features_simple(bev_encoder, dataset, used_indices, output_dir, device, args.batch_size)

    print("\nDone! You can now use these pre-computed features for training.")


if __name__ == '__main__':
    main()
