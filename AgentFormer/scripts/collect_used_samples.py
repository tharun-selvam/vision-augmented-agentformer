#!/usr/bin/env python
"""
Collect which nuScenes sample tokens are actually used by AgentFormer training.

This script runs through the AgentFormer dataloader and saves a list of all
sample_tokens that are accessed during training. This allows us to only
precompute BEV features for samples that will actually be used.

Usage:
    python scripts/collect_used_samples.py --cfg nuscenes_5sample_agentformer_pre_bev --split train
    python scripts/collect_used_samples.py --cfg nuscenes_5sample_agentformer_pre_bev --split val
"""

import os
import sys
import argparse
import json
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import Config
from bevdepth.datasets.nusc_det_dataset import NuscDetDataset


def collect_used_samples(cfg, split='train'):
    """
    Iterate through AgentFormer data and collect which nuScenes samples are used.
    """
    print(f"\n{'='*60}")
    print(f"Collecting used samples for {split} split")
    print(f"{'='*60}\n")

    # Create BEV dataset
    data_root = cfg.data_root_nuscenes_pred

    if split == 'train':
        info_paths = os.path.join(data_root, 'nuscenes_infos_train_subset.pkl')
    elif split == 'val':
        info_paths = os.path.join(data_root, 'nuscenes_infos_val_subset.pkl')
    else:
        raise ValueError(f"Unknown split: {split}")

    print(f"Loading dataset from: {info_paths}")

    nuscenes_data_root = 'nuscenes'

    dataset = NuscDetDataset(
        ida_aug_conf=cfg.ida_aug_conf,
        bda_aug_conf=cfg.bda_aug_conf,
        classes=cfg.object_classes,
        data_root=nuscenes_data_root,
        info_paths=info_paths,
        is_train=False,
        img_conf=cfg.get('img_conf', {}),
        num_sweeps=1,
        return_depth=True,
    )

    print(f"Total nuScenes samples in dataset: {len(dataset)}")

    # Load AgentFormer data to see which samples are actually used
    from data.nuscenes_pred_split import get_nuscenes_pred_split
    from data.preprocessor import preprocess

    seq_train, seq_val, seq_test = get_nuscenes_pred_split(data_root)

    if split == 'train':
        sequences = seq_train
    elif split == 'val':
        sequences = seq_val
    else:
        sequences = seq_test

    print(f"AgentFormer sequences: {len(sequences)}")

    # Process sequences and collect frame info
    used_sample_indices = set()
    total_trajectory_samples = 0

    for seq_name in tqdm(sequences, desc="Processing sequences"):
        try:
            preprocessor = preprocess(data_root, seq_name, cfg, None, split, 'training')

            # Calculate how many training samples this sequence produces
            num_seq_samples = preprocessor.num_fr - (cfg.min_past_frames + cfg.min_future_frames - 1)
            total_trajectory_samples += num_seq_samples

            # For each training sample, we need BEV features
            # The current code uses: bev_sample_idx = sample_index % len(dataset)
            # We'll collect all possible indices that might be accessed
            for i in range(num_seq_samples):
                sample_idx = total_trajectory_samples - num_seq_samples + i
                bev_idx = sample_idx % len(dataset)
                used_sample_indices.add(bev_idx)

        except Exception as e:
            print(f"Warning: Error processing sequence {seq_name}: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Collection complete!")
    print(f"  Total trajectory samples: {total_trajectory_samples}")
    print(f"  Unique BEV samples needed: {len(used_sample_indices)}")
    print(f"  Reduction: {len(dataset)} -> {len(used_sample_indices)} ({100*len(used_sample_indices)/len(dataset):.1f}%)")
    print(f"{'='*60}\n")

    # Get actual sample tokens
    print("Collecting sample tokens...")
    used_tokens = []
    for idx in tqdm(sorted(used_sample_indices), desc="Extracting tokens"):
        try:
            data = dataset[idx]
            img_metas = data[7]
            sample_token = img_metas['token']
            used_tokens.append(sample_token)
        except Exception as e:
            print(f"Warning: Error getting token for index {idx}: {e}")
            continue

    return used_tokens


def main():
    parser = argparse.ArgumentParser(description='Collect used nuScenes samples')
    parser.add_argument('--cfg', type=str, default='nuscenes_5sample_agentformer_pre_bev',
                       help='Config name (without .yml)')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val'],
                       help='Dataset split to process')
    parser.add_argument('--output_dir', type=str, default='bev_features',
                       help='Output directory for token lists')

    args = parser.parse_args()

    # Load config
    print(f"Loading config: {args.cfg}")
    cfg = Config(args.cfg, tmp=False, create_dirs=False)

    # Collect used samples
    used_tokens = collect_used_samples(cfg, args.split)

    # Save to file
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f'used_tokens_{args.split}.json')

    with open(output_file, 'w') as f:
        json.dump({
            'split': args.split,
            'num_tokens': len(used_tokens),
            'tokens': used_tokens
        }, f, indent=2)

    print(f"Saved {len(used_tokens)} tokens to: {output_file}")
    print("\nNow run precomputation with:")
    print(f"  python scripts/precompute_bev_features_smart.py --split {args.split} --gpu 0")


if __name__ == '__main__':
    main()
