#!/usr/bin/env python3
"""
Quick test to verify the dataloader works with the subset dataset.
"""

import sys
import os
from utils.config import Config

def test_dataloader():
    # Load the config
    cfg_name = 'nuscenes_5sample_agentformer_pre'
    print(f"Loading config {cfg_name}...")
    cfg = Config(cfg_name, create_dirs=False)

    print(f"\nConfig dataset: {cfg.dataset}")
    print(f"Data root: {cfg.data_root_nuscenes_pred}")

    # Check if subset pkl files exist
    for split in ['train', 'val']:
        subset_pkl = os.path.join(cfg.data_root_nuscenes_pred, f'nuscenes_infos_{split}_subset.pkl')
        full_pkl = os.path.join(cfg.data_root_nuscenes_pred, f'nuscenes_infos_{split}.pkl')

        print(f"\n{split.upper()} split:")
        print(f"  Subset PKL exists: {os.path.exists(subset_pkl)}")
        if os.path.exists(subset_pkl):
            print(f"  Subset PKL path: {subset_pkl}")
        print(f"  Full PKL exists: {os.path.exists(full_pkl)}")

    # Try to initialize the dataloader
    print("\n" + "="*60)
    print("Attempting to initialize dataloader...")
    print("="*60)

    try:
        from data.dataloader import data_generator
        from utils.utils import prepare_seed

        prepare_seed(cfg.seed)

        # Create a simple logger
        class SimpleLog:
            def write(self, msg):
                print(msg)
            def flush(self):
                pass

        log = SimpleLog()

        # Initialize train dataloader
        print("\nInitializing TRAIN dataloader...")
        train_gen = data_generator(cfg, log, split='train', phase='training')
        print(f"✓ Train dataloader initialized successfully!")
        print(f"  Total samples: {train_gen.num_total_samples}")

        # Initialize val dataloader
        print("\nInitializing VAL dataloader...")
        val_gen = data_generator(cfg, log, split='val', phase='testing')
        print(f"✓ Val dataloader initialized successfully!")
        print(f"  Total samples: {val_gen.num_total_samples}")

        # Try to get one sample
        print("\nTrying to fetch a sample from train dataloader...")
        seq_idx, frame_idx = train_gen.get_seq_and_frame(0)
        print(f"✓ Successfully fetched sample indices: seq={seq_idx}, frame={frame_idx}")

        print("\n" + "="*60)
        print("✓ DATALOADER TEST PASSED!")
        print("="*60)
        return True

    except Exception as e:
        print(f"\n✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("✗ DATALOADER TEST FAILED!")
        print("="*60)
        return False


if __name__ == '__main__':
    success = test_dataloader()
    sys.exit(0 if success else 1)
