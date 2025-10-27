# Pre-computed BEV Features Guide

This guide explains how to use pre-computed BEV features to save GPU memory and training time.

## Overview

Instead of computing BEV features during every training step, you can:
1. **Pre-compute features once** for all samples (~30 minutes on RTX 4050)
2. **Save them to disk** (~24 GB for train_subset)
3. **Load during training** (saves ~3-4GB GPU memory + faster training)

## Benefits

- **GPU Memory**: Saves 3-4GB VRAM (BEV encoder not loaded)
- **Training Speed**: ~10-15% faster (no BEV forward pass)
- **Storage**: 24GB disk space for 2,462 samples (~9.77 MB/sample)

## Quick Start

### 1. Extract BEV Features

```bash
cd AgentFormer
conda activate agentformer

# Extract features for train_subset (2,462 samples)
python scripts/precompute_bev_features.py --split train_subset --gpu 0

# Extract features for val_subset (914 samples)
python scripts/precompute_bev_features.py --split val_subset --gpu 0
```

**Progress:**
- You'll see a progress bar: `Extracting features: 45%|████▌ | 1100/2462 [30:15<33:45, 1.48s/it]`
- Takes ~30-45 minutes for train_subset on RTX 4050

**Output:**
- Features saved to `bev_features/train_subset/{sample_token}.pt`
- Each file contains a tensor of shape `[1, 256, 1, 100, 100]`

### 2. Enable Pre-computed Features in Config

Edit your config file (e.g., `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer_pre_bev.yml`):

```yaml
use_precomputed_bev: true  # Enable loading pre-computed features
bev_features_dir: 'bev_features'  # Directory where features are stored
```

### 3. Train with Pre-computed Features

```bash
python train.py --cfg nuscenes_5sample_agentformer_pre_bev --gpu 0
```

The model will automatically:
- Load BEV features from disk instead of computing them
- Skip loading the BEV encoder (saves GPU memory)
- Use cached features during training

## Extraction Script Options

```bash
python scripts/precompute_bev_features.py \
    --split train_subset \      # Dataset split: train_subset, val_subset, train, val
    --gpu 0 \                   # GPU device ID
    --cfg nuscenes_5sample_agentformer_pre_bev \  # Config file (default)
    --output_dir bev_features   # Output directory (default)
```

## Storage Requirements

| Split | Samples | Disk Space |
|-------|---------|------------|
| train_subset | 2,462 | ~24 GB |
| val_subset | 914 | ~9 GB |
| train (full) | 28,130 | ~275 GB |
| val (full) | 6,019 | ~59 GB |

## Technical Details

### Feature Format

Each saved feature file contains:
- **Tensor shape**: `[1, 256, 1, 100, 100]`
  - Batch dimension: 1
  - Channels: 256 (BEV feature channels)
  - Depth: 1 (single sweep)
  - Height: 100 (BEV grid height)
  - Width: 100 (BEV grid width)
- **Data type**: float32
- **File size**: ~9.77 MB per sample

### Fallback Implementation

The extraction script uses a Python fallback for voxel pooling if the CUDA extension is not available. The fallback:
1. Weights features by depth probabilities
2. Interpolates geometry to match feature resolution
3. Scatters features into BEV voxel grid

This is slower than the CUDA version but produces identical results.

## Troubleshooting

### Out of Disk Space

If you run out of disk space during extraction:
1. Extract only train_subset first
2. Train and validate
3. Extract val_subset separately if needed

### Features Not Loading

Check:
1. Config has `use_precomputed_bev: true`
2. Features exist in `bev_features/{split}/`
3. Directory path is correct in config

### Extraction Errors

If extraction fails:
- **Shape mismatch errors**: Fixed in current version
- **CUDA OOM**: Reduce batch size or use CPU (slower)
- **Missing samples**: Some samples may fail, extraction continues

### Verifying Extraction

Check a single sample:
```python
import torch
feature = torch.load('bev_features/train_subset/{sample_token}.pt')
print(f"Shape: {feature.shape}")  # Should be [1, 256, 1, 100, 100]
print(f"Size: {feature.numel() * feature.element_size() / 1024 / 1024:.2f} MB")
```

## Performance Comparison

| Mode | GPU Memory | Train Time/Epoch | Disk Space |
|------|-----------|------------------|------------|
| **On-the-fly** | ~5.5 GB | 5.5 hours | 0 GB |
| **Pre-computed** | ~2 GB | 4.5 hours | 24 GB |

*Times measured on RTX 4050 Mobile (6GB) with train_subset*

## Advanced Usage

### Custom BEV Encoder

To use a different BEV encoder for feature extraction:

1. Modify `scripts/precompute_bev_features.py`:
```python
# Load your custom BEV encoder
from my_module import MyBEVEncoder
bev_encoder = MyBEVEncoder(**cfg.bev_encoder)
```

2. Extract features with custom config

### Distributed Extraction

Extract features in parallel on multiple GPUs:

```bash
# GPU 0: samples 0-1231
python scripts/precompute_bev_features.py --split train_subset --gpu 0 --start_idx 0 --end_idx 1231

# GPU 1: samples 1231-2462
python scripts/precompute_bev_features.py --split train_subset --gpu 1 --start_idx 1231 --end_idx 2462
```

Note: Current script doesn't support --start_idx/--end_idx, would need to be added.

## See Also

- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Complete setup instructions
- [README_MODIFICATIONS.md](README_MODIFICATIONS.md) - All code changes
- [BEV_INTEGRATION_COMPLETE.md](BEV_INTEGRATION_COMPLETE.md) - BEV integration details
