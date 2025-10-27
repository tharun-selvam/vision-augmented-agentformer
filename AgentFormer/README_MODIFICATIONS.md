# Vision-Augmented AgentFormer

This repository contains modifications to the original [AgentFormer](https://github.com/Khrylx/AgentFormer) to integrate BEVDepth visual features for trajectory prediction on nuScenes dataset.

## Overview

This project implements an ablation study comparing:
1. **Baseline AgentFormer**: Original trajectory prediction using only trajectory and map features
2. **Vision-Augmented AgentFormer**: Enhanced with BEVDepth visual embeddings from multi-view camera images

## Key Modifications

### 1. Conditional BEVDepth Integration
- **File**: `data/dataloader.py`
- **Changes**: Added `use_bev` flag to toggle between original AgentFormer dataloader and BEVDepth-based NuscDetDataset
- **Purpose**: Enables fair ablation study by using same codebase for both approaches

### 2. Dataset Filtering for 10% nuScenes Subset
- **Files**:
  - `scripts/filter_nuscenes_subset.py` - Filters metadata to 85 available scenes
  - `scripts/gen_info_subset.py` - Generates subset .pkl files
  - `data/nuscenes_pred_split.py` - Updated to filter scene list
- **Purpose**: Work with partial nuScenes dataset (Part 1 of 10, ~30GB, 85 scenes)

### 3. Training Pipeline Fixes
- **File**: `train.py`
- **Changes**:
  - Fixed variable naming (`agent_former` → `model`)
  - Added missing `model.forward()` call before loss computation
  - Fixed logging for nuScenes dataset
- **Purpose**: Enable stable training on nuScenes

### 4. Quick Pipeline Test
- **File**: `quick_test_pipeline.py`
- **Purpose**: Verify complete training workflow (pre-training → trajectory sampler → evaluation) in ~5-10 minutes
- **Details**: Runs 50 iterations per stage instead of full epochs

## System Requirements

### Hardware
- **Minimum**: NVIDIA GPU with 6GB VRAM (tested on RTX 4050 Mobile)
- **Recommended**: NVIDIA GPU with 16GB+ VRAM (e.g., RTX 3090, RTX 4090, RTX 5090)
- **Storage**: ~50GB for dataset and results
- **RAM**: 16GB+ recommended

### Software
- **OS**: Linux (tested on Ubuntu with kernel 6.8.0-85)
- **CUDA**: 11.1+ (compatible with PyTorch 1.9.0)
- **Python**: 3.7

### Training Time Estimates
- **RTX 4050 Mobile (6GB)**: ~5.5 hours/epoch
- **RTX 5090 (32GB)**: ~20-40 minutes/epoch (estimated 10-15x speedup)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd AgentFormer
```

### 2. Set Up Conda Environment

```bash
# Create environment
conda create -n agentformer python=3.7
conda activate agentformer

# Install PyTorch (CUDA 11.1)
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
pip install -r requirements.txt

# Install nuScenes devkit
pip install nuscenes-devkit
```

### 3. Download nuScenes Dataset

**Option A: Full Dataset (850 scenes, ~300GB)**
```bash
# Download from https://www.nuscenes.org/nuscenes#download
# Not recommended unless you need full dataset
```

**Option B: 10% Subset (85 scenes, ~30GB) - Recommended**
```bash
# Download Part 1 of nuScenes trainval
# 1. Go to https://www.nuscenes.org/nuscenes#download
# 2. Download "v1.0-trainval_meta" (metadata)
# 3. Download "Trainval part 1" only (~30GB)
# 4. Extract to datasets/nuscenes_pred/
```

### 4. Filter Dataset to Available Scenes

```bash
# Filter metadata to match your 85 scenes
python scripts/filter_nuscenes_subset.py

# Generate .pkl info files for the subset
python scripts/gen_info_subset.py
```

Expected output:
- `datasets/nuscenes_pred/v1.0-trainval-subset/` - Filtered metadata
- `data/nuscenes_infos_train_subset.pkl` - 2462 train samples, 62 scenes
- `data/nuscenes_infos_val_subset.pkl` - 914 val samples, 23 scenes
- `data/nuscenes_infos_all_subset.pkl` - 3376 total samples, 85 scenes

## Dataset Structure

After setup, your directory should look like:

```
AgentFormer/
├── datasets/
│   └── nuscenes_pred/
│       ├── v1.0-trainval/          # Original metadata
│       ├── v1.0-trainval-subset/   # Filtered metadata (85 scenes)
│       ├── samples/                # Camera images and sensor data
│       ├── sweeps/                 # Additional sensor sweeps
│       └── maps/                   # HD maps
├── data/
│   ├── nuscenes_infos_train_subset.pkl
│   ├── nuscenes_infos_val_subset.pkl
│   └── nuscenes_infos_all_subset.pkl
├── cfg/
│   └── nuscenes/
│       └── 5_sample/
│           ├── nuscenes_5sample_agentformer_pre.yml      # Stage 1 config
│           ├── nuscenes_5sample_agentformer.yml          # Stage 2 config
│           ├── nuscenes_5sample_agentformer_pre_test.yml # Quick test Stage 1
│           └── nuscenes_5sample_agentformer_test.yml     # Quick test Stage 2
└── results/                        # Training outputs (created automatically)
```

## Usage

### Quick Pipeline Test (5-10 minutes)

Before committing to full training, verify the pipeline works:

```bash
conda activate agentformer
python quick_test_pipeline.py
```

This runs 50 iterations per stage to test:
- ✓ Stage 1: VAE pre-training
- ✓ Stage 2: DLow trajectory sampler training
- ✓ Stage 3: Evaluation

### Full Training Pipeline

#### Stage 1: VAE Pre-training (Baseline, no BEVDepth)

```bash
# Pre-train the VAE encoder/decoder
python train.py --cfg nuscenes_5sample_agentformer_pre --gpu 0

# Expected output:
# - Checkpoints saved to: results/nuscenes_5sample_agentformer_pre/models/
# - Logs saved to: results/nuscenes_5sample_agentformer_pre/log/
# - Training time: ~5.5 hours/epoch on RTX 4050 Mobile
```

**Config**: `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer_pre.yml`
- `use_bev: false` - Uses original AgentFormer dataloader (no visual features)
- `num_epochs: 30` (default)
- `model_id: agentformer`

#### Stage 2: DLow Trajectory Sampler Training

```bash
# Train the DLow diversity sampler
python train.py --cfg nuscenes_5sample_agentformer --gpu 0

# This automatically loads the pre-trained model from Stage 1
```

**Config**: `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer.yml`
- `use_bev: false` - Baseline (no visual features)
- `pred_cfg: nuscenes_5sample_agentformer_pre` - Points to Stage 1 checkpoint
- `pred_epoch: 30` - Loads epoch 30 from Stage 1
- `model_id: dlow`

#### Stage 3: Evaluation

```bash
# Evaluate on validation set
python test.py --cfg nuscenes_5sample_agentformer --gpu 0

# Outputs ADE/FDE metrics
```

### Training with BEVDepth (Vision-Augmented)

To train with BEVDepth visual features, modify the config files:

**1. Edit `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer_pre.yml`:**
```yaml
# Add this line to enable BEVDepth dataloader
use_bev: true

# Update data paths
data_root_nuscenes_pred: datasets/nuscenes_pred
info_train_path: data/nuscenes_infos_train_subset.pkl
info_val_path: data/nuscenes_infos_val_subset.pkl
```

**2. Edit `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer.yml`:**
```yaml
use_bev: true
# Keep pred_cfg and pred_epoch pointing to BEVDepth pre-trained model
```

**3. Run training:**
```bash
# Stage 1: Pre-training with BEVDepth
python train.py --cfg nuscenes_5sample_agentformer_pre --gpu 0

# Stage 2: DLow with BEVDepth
python train.py --cfg nuscenes_5sample_agentformer --gpu 0

# Stage 3: Evaluation
python test.py --cfg nuscenes_5sample_agentformer --gpu 0
```

## Configuration Files

### Key Config Parameters

| Parameter | Description | Baseline | Vision-Augmented |
|-----------|-------------|----------|------------------|
| `use_bev` | Enable BEVDepth dataloader | `false` | `true` |
| `model_id` | Model architecture | `agentformer` (Stage 1), `dlow` (Stage 2) | Same |
| `pred_cfg` | Pre-trained model config | `nuscenes_5sample_agentformer_pre` | Same |
| `pred_epoch` | Epoch to load from Stage 1 | `30` | `30` |
| `num_epochs` | Training epochs | `30` | `30` |
| `lr` | Learning rate | `1e-4` | `1e-4` |
| `past_frames` | History length | `4` | `4` |
| `future_frames` | Prediction horizon | `12` | `12` |

### Creating Custom Configs

To create a new config for different settings:

```bash
# Copy existing config
cp cfg/nuscenes/5_sample/nuscenes_5sample_agentformer_pre.yml \
   cfg/nuscenes/5_sample/my_custom_config.yml

# Edit parameters
nano cfg/nuscenes/5_sample/my_custom_config.yml

# Train with custom config
python train.py --cfg my_custom_config --gpu 0
```

## Ablation Study Guide

### Running Complete Ablation Study

```bash
# 1. Train Baseline (no BEVDepth)
python train.py --cfg nuscenes_5sample_agentformer_pre --gpu 0  # Stage 1
python train.py --cfg nuscenes_5sample_agentformer --gpu 0      # Stage 2
python test.py --cfg nuscenes_5sample_agentformer --gpu 0       # Eval

# 2. Train Vision-Augmented (with BEVDepth)
# First, modify configs to set use_bev: true
python train.py --cfg nuscenes_5sample_agentformer_pre --gpu 0  # Stage 1
python train.py --cfg nuscenes_5sample_agentformer --gpu 0      # Stage 2
python test.py --cfg nuscenes_5sample_agentformer --gpu 0       # Eval

# 3. Compare results
# Compare ADE/FDE metrics from both runs
```

### Expected Results Structure

```
results/
├── nuscenes_5sample_agentformer_pre/     # Baseline Stage 1
│   ├── models/
│   │   ├── model_0001.p
│   │   ├── ...
│   │   └── model_0030.p
│   ├── log/
│   │   └── log.txt
│   └── tb/                                # TensorBoard logs
├── nuscenes_5sample_agentformer/         # Baseline Stage 2
│   ├── models/
│   ├── log/
│   └── tb/
├── nuscenes_5sample_agentformer_pre_bev/ # Vision-Augmented Stage 1
│   └── ...
└── nuscenes_5sample_agentformer_bev/     # Vision-Augmented Stage 2
    └── ...
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Errors

```bash
# Reduce batch size in config
# Edit cfg/nuscenes/5_sample/*.yml
batch_size: 8  # Try smaller values: 4, 2, 1
```

#### 2. "No samples found" Error

```bash
# Verify dataset structure
ls datasets/nuscenes_pred/v1.0-trainval-subset/

# Re-run filtering scripts
python scripts/filter_nuscenes_subset.py
python scripts/gen_info_subset.py
```

#### 3. CUDA Out of Memory

```bash
# Check GPU memory
nvidia-smi

# Kill existing processes
pkill -f train.py

# Use smaller batch size or reduce model size
```

#### 4. "Module not found" Errors

```bash
# Ensure conda environment is activated
conda activate agentformer

# Reinstall dependencies
pip install -r requirements.txt
```

#### 5. Checkpoint Not Found

```bash
# Check checkpoint path in config
# Edit pred_cfg and pred_epoch in Stage 2 config
pred_cfg: nuscenes_5sample_agentformer_pre  # Must match Stage 1 config name
pred_epoch: 30  # Must match saved checkpoint epoch
```

### Getting Help

- **Original AgentFormer**: https://github.com/Khrylx/AgentFormer
- **nuScenes Dataset**: https://www.nuscenes.org/nuscenes
- **BEVDepth**: https://github.com/Megvii-BaseDetection/BEVDepth

## File Descriptions

### Modified Files

| File | Purpose | Changes |
|------|---------|---------|
| `data/dataloader.py` | Data loading | Added conditional BEV loading with `use_bev` flag |
| `data/nuscenes_pred_split.py` | Scene filtering | Filter to 85 available scenes |
| `train.py` | Training script | Fixed variable names, added forward() call, logging fixes |
| `quick_test_pipeline.py` | Quick testing | New file for pipeline verification |

### New Scripts

| File | Purpose |
|------|---------|
| `scripts/filter_nuscenes_subset.py` | Filter metadata to 85 scenes |
| `scripts/gen_info_subset.py` | Generate subset .pkl files |
| `quick_test_pipeline.py` | Quick pipeline test (50 iterations) |

### Config Files

| File | Purpose |
|------|---------|
| `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer_pre.yml` | Stage 1 training config |
| `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer.yml` | Stage 2 training config |
| `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer_pre_test.yml` | Quick test Stage 1 |
| `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer_test.yml` | Quick test Stage 2 |

## Performance Benchmarks

### Training Time (per epoch)

| Hardware | Stage 1 (VAE) | Stage 2 (DLow) |
|----------|---------------|----------------|
| RTX 4050 Mobile (6GB) | ~5.5 hours | ~5.5 hours |
| RTX 5090 (32GB) | ~20-40 min | ~20-40 min |

### Memory Usage

| Configuration | GPU Memory | System RAM |
|---------------|------------|------------|
| Baseline (no BEV) | ~4-5 GB | ~8 GB |
| Vision-Augmented (BEV) | ~5-6 GB | ~12 GB |

## Citation

If you use this code, please cite the original AgentFormer paper:

```bibtex
@inproceedings{yuan2021agent,
  title={Agent-Former: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting},
  author={Yuan, Ye and Weng, Xinshuo and Ou, Yanglan and Kitani, Kris},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

And the BEVDepth paper if using visual features:

```bibtex
@article{li2022bevdepth,
  title={BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection},
  author={Li, Yinhao and Ge, Zheng and Yu, Guanyi and Yang, Jinrong and Wang, Zengran and Shi, Yukang and Sun, Jianjian and Li, Zeming},
  journal={arXiv preprint arXiv:2206.10092},
  year={2022}
}
```

## License

This project inherits the license from the original AgentFormer repository.

## Acknowledgments

- Original AgentFormer implementation by Ye Yuan et al.
- BEVDepth implementation by Megvii Technology
- nuScenes dataset by Motional
