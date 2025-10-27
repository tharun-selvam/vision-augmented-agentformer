#!/usr/bin/env python
"""
Pre-compute BEV features for all nuScenes samples.

This script extracts BEV feature maps using the frozen BEV encoder and saves them to disk.
This avoids running the BEV encoder during training, saving GPU memory and time.

Usage:
    python scripts/precompute_bev_features.py --split train_subset --gpu 0
    python scripts/precompute_bev_features.py --split val_subset --gpu 0
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bevdepth.datasets.nusc_det_dataset import NuscDetDataset
from model.bev.base_lss_fpn import BaseLSSFPN
from utils.config import Config


def load_bev_encoder(cfg, device):
    """Load and freeze BEV encoder"""
    print("Loading BEV encoder...")
    bev_encoder = BaseLSSFPN(**cfg.bev_encoder)

    # Freeze all parameters
    for param in bev_encoder.parameters():
        param.requires_grad = False

    # Set to eval mode
    bev_encoder.eval()
    bev_encoder.to(device)

    print(f"BEV encoder loaded and frozen. Output channels: {bev_encoder.output_channels}")
    return bev_encoder


def create_dataset(cfg, split, device):
    """Create BEV dataset for the given split"""
    data_root = cfg.data_root_nuscenes_pred

    # Determine pkl file
    if split == 'train_subset':
        info_paths = os.path.join(data_root, 'nuscenes_infos_train_subset.pkl')
    elif split == 'val_subset':
        info_paths = os.path.join(data_root, 'nuscenes_infos_val_subset.pkl')
    elif split == 'train':
        info_paths = os.path.join(data_root, 'nuscenes_infos_train.pkl')
    elif split == 'val':
        info_paths = os.path.join(data_root, 'nuscenes_infos_val.pkl')
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(info_paths):
        raise FileNotFoundError(f"Dataset file not found: {info_paths}")

    print(f"Creating dataset from: {info_paths}")

    dataset = NuscDetDataset(
        ida_aug_conf=cfg.ida_aug_conf,
        bda_aug_conf=cfg.bda_aug_conf,
        classes=cfg.object_classes,
        data_root='/home/tharun/Documents/BTP/AgentFormer/nuscenes',
        info_paths=info_paths,
        is_train=False,  # Important: set to False for deterministic augmentation
        img_conf=cfg.get('img_conf', {}),
        num_sweeps=1,
        return_depth=True,
    )

    print(f"Dataset created with {len(dataset)} samples")
    return dataset


def extract_features(bev_encoder, dataset, output_dir, device):
    """Extract BEV features for all samples in dataset"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting BEV features to: {output_dir}")
    print(f"Total samples: {len(dataset)}")

    # Statistics
    total_size_mb = 0
    skipped = 0
    extracted = 0

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Extracting features"):
            # Get sample
            data = dataset[idx]

            # data is a list: [sweep_imgs, sensor2ego_mats, intrins, ida_mats,
            #                  sensor2sensor_mats, bda_mat, timestamps, img_metas, ...]
            sweep_imgs = data[0]
            img_metas = data[7]
            sample_token = img_metas['token']

            # Check if already exists
            output_file = output_dir / f"{sample_token}.pt"
            if output_file.exists():
                skipped += 1
                continue

            # Prepare mats_dict
            mats_dict = {
                'sensor2ego_mats': data[1],
                'intrin_mats': data[2],
                'ida_mats': data[3],
                'sensor2sensor_mats': data[4],
                'bda_mat': data[5],
            }

            # Move to device and add num_sweeps dimension if needed
            sweep_imgs = sweep_imgs.to(device)
            # Dataset returns [B, num_cams, C, H, W], but BEV encoder expects [B, num_sweeps, num_cams, C, H, W]
            if len(sweep_imgs.shape) == 5:
                sweep_imgs = sweep_imgs.unsqueeze(1)  # Add num_sweeps=1 dimension

            # Also add num_sweeps dimension to mats
            mats_dict_processed = {}
            for k, v in mats_dict.items():
                if isinstance(v, torch.Tensor):
                    v = v.to(device)
                    if k == 'bda_mat':
                        # bda_mat is [4, 4] but needs batch dimension: [4, 4] -> [B, 4, 4]
                        if len(v.shape) == 2:
                            v = v.unsqueeze(0)
                    else:
                        # Add sweep dimension for camera matrices: [B, num_cams, 4, 4] -> [B, num_sweeps, num_cams, 4, 4]
                        if len(v.shape) == 3:
                            v = v.unsqueeze(1)
                        elif len(v.shape) == 4 and v.shape[1] != 1:  # [B, num_cams, 4, 4] case
                            v = v.unsqueeze(1)
                mats_dict_processed[k] = v
            mats_dict = mats_dict_processed

            # Extract BEV features
            try:
                # Add sweep dimension to timestamps: [B, num_cams] -> [B, num_sweeps, num_cams]
                timestamps = data[6].to(device)
                if len(timestamps.shape) == 2:
                    timestamps = timestamps.unsqueeze(1)

                bev_feature_map = bev_encoder(sweep_imgs, mats_dict, timestamps=timestamps, is_return_depth=False)

                # Move to CPU and save
                bev_feature_map = bev_feature_map.cpu()
                torch.save(bev_feature_map, output_file)

                # Update statistics
                file_size_mb = output_file.stat().st_size / (1024 * 1024)
                total_size_mb += file_size_mb
                extracted += 1

            except Exception as e:
                print(f"\nError processing sample {idx} (token: {sample_token}): {e}")
                continue

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"  Extracted: {extracted} samples")
    print(f"  Skipped (already exist): {skipped} samples")
    print(f"  Total disk usage: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)")
    print(f"  Average size per sample: {total_size_mb/max(extracted,1):.2f} MB")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute BEV features')
    parser.add_argument('--cfg', type=str, default='nuscenes_5sample_agentformer_pre_bev',
                       help='Config name (without .yml)')
    parser.add_argument('--split', type=str, default='train_subset',
                       choices=['train_subset', 'val_subset', 'train', 'val'],
                       help='Dataset split to process')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--output_dir', type=str, default='bev_features',
                       help='Output directory for features')

    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config (Config class expects just the config name, not full path)
    print(f"Loading config: {args.cfg}")
    cfg = Config(args.cfg, tmp=False, create_dirs=False)

    # Create output directory per split
    output_dir = os.path.join(args.output_dir, args.split)

    # Load BEV encoder
    bev_encoder = load_bev_encoder(cfg, device)

    # Create dataset
    dataset = create_dataset(cfg, args.split, device)

    # Extract features
    extract_features(bev_encoder, dataset, output_dir, device)

    print("\nDone! You can now use these pre-computed features for training.")
    print(f"Set use_precomputed_bev: true in your config to use them.")


if __name__ == '__main__':
    main()
