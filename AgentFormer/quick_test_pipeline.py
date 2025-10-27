#!/usr/bin/env python3
"""
Quick pipeline test - runs just a few iterations to verify the complete workflow.
This takes only a few minutes instead of hours.
"""

import os
import sys
import torch
from torch import optim

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from model.model_lib import model_dict
from utils.torch import *
from utils.config import Config
from utils.utils import prepare_seed, print_log, get_timestring

# Disable CUDA benchmarking for faster startup
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MAX_ITERS = 50  # Only run 50 iterations for quick test


def quick_test_stage1():
    """Test Stage 1: VAE training"""
    print("\n" + "="*60)
    print("STAGE 1: VAE Pre-training Test (50 iterations)")
    print("="*60)

    cfg = Config('nuscenes_5sample_agentformer_pre_test', create_dirs=True)
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    log = open(os.path.join(cfg.log_dir, 'quick_test.txt'), 'w')
    print_log(f"Quick test - Stage 1 VAE", log)

    # Data
    generator = data_generator(cfg, log, split='train', phase='training')
    print_log(f"Total samples: {generator.num_total_samples}", log)

    # Model
    model_id = cfg.get('model_id', 'agentformer')
    model = model_dict[model_id](cfg)
    model.set_device(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Quick training
    print(f"\nRunning {MAX_ITERS} iterations...")
    for i in range(MAX_ITERS):
        data = generator()
        if data is not None:
            model.set_data(data)
            model.forward()
            total_loss, _, _ = model.compute_loss()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"  Iter {i+1}/{MAX_ITERS}: loss = {total_loss.item():.4f}")

    # Save checkpoint
    model_path = os.path.join(cfg.model_dir, 'model_0002.p')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_cp = {'model_dict': model.state_dict(), 'epoch': 2}
    torch.save(model_cp, model_path)
    print(f"\n✓ Stage 1 checkpoint saved: {model_path}")
    log.close()

    return 'nuscenes_5sample_agentformer_pre_test'


def quick_test_stage2(pred_cfg_name):
    """Test Stage 2: DLow training"""
    print("\n" + "="*60)
    print("STAGE 2: DLow Trajectory Sampler Test (50 iterations)")
    print("="*60)

    cfg = Config('nuscenes_5sample_agentformer_test', create_dirs=True)
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    log = open(os.path.join(cfg.log_dir, 'quick_test.txt'), 'w')
    print_log(f"Quick test - Stage 2 DLow", log)

    # Data
    generator = data_generator(cfg, log, split='train', phase='training')
    print_log(f"Total samples: {generator.num_total_samples}", log)

    # Model
    model_id = cfg.get('model_id', 'dlow')
    model = model_dict[model_id](cfg)
    model.set_device(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # Quick training
    print(f"\nRunning {MAX_ITERS} iterations...")
    for i in range(MAX_ITERS):
        data = generator()
        if data is not None:
            model.set_data(data)
            model.forward()
            total_loss, _, _ = model.compute_loss()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f"  Iter {i+1}/{MAX_ITERS}: loss = {total_loss.item():.4f}")

    # Save checkpoint
    model_path = os.path.join(cfg.model_dir, 'model_0002.p')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_cp = {'model_dict': model.state_dict(), 'epoch': 2}
    torch.save(model_cp, model_path)
    print(f"\n✓ Stage 2 checkpoint saved: {model_path}")
    log.close()

    return 'nuscenes_5sample_agentformer_test'


def quick_test_evaluation(cfg_name):
    """Test Evaluation"""
    print("\n" + "="*60)
    print("STAGE 3: Evaluation Test (20 samples)")
    print("="*60)

    cfg = Config(cfg_name, create_dirs=False)
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    log = sys.stdout

    # Data - try val split first, fall back to train split if empty
    generator = data_generator(cfg, log, split='val', phase='testing')
    if generator.num_total_samples == 0:
        print("No validation samples found, using train split for quick test...")
        generator = data_generator(cfg, log, split='train', phase='testing')
    print(f"Test samples: {generator.num_total_samples}")

    # Model
    model_id = cfg.get('model_id', 'dlow')
    model = model_dict[model_id](cfg)

    # Load checkpoint
    model_path = os.path.join(cfg.model_dir, 'model_0002.p')
    if os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model_cp = torch.load(model_path, map_location='cpu')
        model.load_state_dict(model_cp['model_dict'])
    else:
        print(f"WARNING: Model checkpoint not found at {model_path}")
        return

    model.set_device(device)
    model.eval()

    # Quick evaluation on 20 samples
    print(f"\nEvaluating on 20 samples...")
    ade_errors = []
    fde_errors = []

    with torch.no_grad():
        for i in range(20):
            data = generator()
            if data is not None:
                model.set_data(data)
                predictions, _ = model.inference(mode='infer', sample_num=5)

                # Compute simple ADE/FDE (simplified for quick test)
                # predictions is a dict with 'infer_dec_motion' key
                if isinstance(predictions, dict) and 'infer_dec_motion' in predictions:
                    pred = predictions['infer_dec_motion']  # [K, T, N, 2] where K=sample_num
                    gt = model.data['fut_motion_orig']  # [T, N, 2]

                    # Take best of K samples (minimum error)
                    pred = pred.permute(1, 2, 0, 3)  # [T, N, K, 2]
                    gt_expanded = gt.unsqueeze(2)  # [T, N, 1, 2]

                    # Compute errors for all K samples
                    errors_all = torch.norm(pred - gt_expanded, dim=-1)  # [T, N, K]
                    errors_best = errors_all.min(dim=2)[0]  # [T, N] - best of K samples

                    ade = errors_best.mean().item()
                    fde = errors_best[-1].mean().item()
                    ade_errors.append(ade)
                    fde_errors.append(fde)

                    if (i + 1) % 5 == 0:
                        print(f"  Sample {i+1}/20: ADE={ade:.3f}, FDE={fde:.3f}")

    if ade_errors:
        print(f"\n✓ Evaluation complete!")
        print(f"  Average ADE: {sum(ade_errors)/len(ade_errors):.3f}")
        print(f"  Average FDE: {sum(fde_errors)/len(fde_errors):.3f}")
    else:
        print("  Warning: No errors computed (check data format)")


def main():
    print("\n" + "="*70)
    print(" QUICK PIPELINE TEST - AgentFormer Complete Workflow")
    print(" This tests all 3 stages in ~5-10 minutes")
    print("="*70)

    try:
        # Stage 1: VAE training
        pred_cfg = quick_test_stage1()

        # Stage 2: DLow training
        final_cfg = quick_test_stage2(pred_cfg)

        # Stage 3: Evaluation
        quick_test_evaluation(final_cfg)

        print("\n" + "="*70)
        print(" ✓ PIPELINE TEST COMPLETE - All stages working!")
        print("="*70)
        print("\nNext steps:")
        print("1. Full training: python train.py --cfg nuscenes_5sample_agentformer_pre --gpu 0")
        print("2. Then: python train.py --cfg nuscenes_5sample_agentformer --gpu 0")
        print("3. Finally: python test.py --cfg nuscenes_5sample_agentformer --gpu 0")

    except Exception as e:
        print(f"\n✗ PIPELINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
