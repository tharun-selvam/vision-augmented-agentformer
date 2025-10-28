# Vision-Augmented AgentFormer Quickstart

Minimal setup guide to train and evaluate AgentFormer with and without BEV visual features.

## Prerequisites

- NVIDIA GPU with 32GB VRAM (RTX 5090 or equivalent)
- 100GB+ free disk space
- Conda environment

## Setup

```bash
# Clone and setup environment
cd AgentFormer
conda env create -f environment.yml
conda activate agentformer_py310
```

## Step 1: Download nuScenes Dataset

```bash
# Download nuScenes v1.0 mini split (10 scenes, ~4GB)
python download_nuscenes.py --dataroot nuscenes --version v1.0-mini

# Or download full trainval split (~350GB)
# python download_nuscenes.py --dataroot nuscenes --version v1.0-trainval
```

## Step 2: Preprocess Data

```bash
# Generate info pickle files for trajectory prediction
python scripts/gen_info_subset.py --version v1.0-mini

# This creates:
# - datasets/nuscenes_pred/nuscenes_infos_train_subset.pkl
# - datasets/nuscenes_pred/nuscenes_infos_val_subset.pkl
```

## Step 3: Train Vanilla AgentFormer (Baseline)

```bash
# Stage 1: VAE pre-training (30 epochs, ~2 hours)
python train.py --cfg nuscenes_5sample_agentformer_pre --gpu 0

# Stage 2: Full model with DLow sampler (50 epochs, ~3 hours)
python train.py --cfg nuscenes_5sample_agentformer --gpu 0
```

## Step 4: Evaluate Baseline

```bash
python test.py --cfg nuscenes_5sample_agentformer --gpu 0

# Results saved to: results/nuscenes_5sample_agentformer/test_results.txt
# Expected metrics: ADE ~4.3m, FDE ~8.4m
```

## Step 5: Precompute BEV Features

```bash
# Collect which samples are actually used (~1,034 out of 28,130)
python scripts/collect_used_samples.py --split train --cfg nuscenes_5sample_agentformer_pre_bev
python scripts/collect_used_samples.py --split val --cfg nuscenes_5sample_agentformer_pre_bev

# Precompute BEV features (single camera, ~4-6 hours)
python scripts/precompute_bev_features_simple.py --split train --gpu 0 --batch_size 8
python scripts/precompute_bev_features_simple.py --split val --gpu 0 --batch_size 8

# BEV features saved to: bev_features/train/*.pt and bev_features/val/*.pt
```

## Step 6: Train Vision-Augmented AgentFormer

```bash
# Stage 1: VAE pre-training with BEV (30 epochs, ~20 hours with precomputed BEV)
python train.py --cfg nuscenes_5sample_agentformer_pre_bev --gpu 0

# Stage 2: Full model with BEV (50 epochs, ~30 hours)
python train.py --cfg nuscenes_5sample_agentformer_bev --gpu 0
```

**Note:** If precomputation fails due to disk space, edit config to use on-the-fly BEV extraction:
```yaml
# In cfg/nuscenes/5_sample/nuscenes_5sample_agentformer_pre_bev.yml
use_precomputed_bev: false  # Change to false for on-the-fly extraction
```

## Step 7: Evaluate Vision-Augmented Model

```bash
python test.py --cfg nuscenes_5sample_agentformer_bev --gpu 0

# Results saved to: results/nuscenes_5sample_agentformer_bev/test_results.txt
```

## Step 8: Compare Results

```bash
# View baseline results
cat results/nuscenes_5sample_agentformer/test_results.txt

# View vision-augmented results
cat results/nuscenes_5sample_agentformer_bev/test_results.txt

# Expected improvement: ~5-10% reduction in ADE/FDE with visual context
```

## Quick Commands Summary

```bash
# Full pipeline (run in order)
python download_nuscenes.py --dataroot nuscenes --version v1.0-mini
python scripts/gen_info_subset.py --version v1.0-mini
python train.py --cfg nuscenes_5sample_agentformer_pre --gpu 0
python train.py --cfg nuscenes_5sample_agentformer --gpu 0
python test.py --cfg nuscenes_5sample_agentformer --gpu 0
python scripts/collect_used_samples.py --split train --cfg nuscenes_5sample_agentformer_pre_bev
python scripts/precompute_bev_features_simple.py --split train --gpu 0 --batch_size 8
python train.py --cfg nuscenes_5sample_agentformer_pre_bev --gpu 0
python train.py --cfg nuscenes_5sample_agentformer_bev --gpu 0
python test.py --cfg nuscenes_5sample_agentformer_bev --gpu 0
```

## Troubleshooting

**Out of disk space during BEV precomputation:**
- Use on-the-fly extraction: set `use_precomputed_bev: false` in config
- Or reduce batch size: `--batch_size 4`

**CUDA out of memory:**
- Reduce `max_train_agent` in config (default: 16 → try 8)
- Use single camera: `cams: ['CAM_FRONT']`

**Training too slow:**
- Check GPU utilization: `nvidia-smi`
- Reduce image resolution in config: `final_dim: [64, 192]`

## Configuration Files

- `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer_pre.yml` - Baseline Stage 1
- `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer.yml` - Baseline Stage 2
- `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer_pre_bev.yml` - BEV Stage 1
- `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer_bev.yml` - BEV Stage 2

## Expected Timeline

| Step | Time | Notes |
|------|------|-------|
| Download nuScenes mini | 10 min | 4GB download |
| Preprocess data | 5 min | Generate pickle files |
| Train baseline Stage 1 | 2 hours | 30 epochs |
| Train baseline Stage 2 | 3 hours | 50 epochs |
| Evaluate baseline | 10 min | Get ADE/FDE |
| Precompute BEV | 4-6 hours | 1,034 samples |
| Train BEV Stage 1 | 20 hours | 30 epochs |
| Train BEV Stage 2 | 30 hours | 50 epochs |
| Evaluate BEV model | 10 min | Get ADE/FDE |
| **Total** | **~60 hours** | Full pipeline |
