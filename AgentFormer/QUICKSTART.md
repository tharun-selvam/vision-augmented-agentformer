# Vision-Augmented AgentFormer Quickstart

Minimal setup guide to train and evaluate AgentFormer with and without BEV visual features.

## Prerequisites

- NVIDIA GPU with 24-32GB VRAM
- 100GB+ free disk space
- Conda environment

### ✅ **RTX 5090 Supported**

**The RTX 5090 (compute capability sm_120) is NOW supported by PyTorch 2.9+** (released October 2025).

**Supported GPUs:**
- ✅ RTX 5090 (32GB VRAM, sm_120) - Requires PyTorch 2.9+ with CUDA 12.8
- ✅ RTX 4090 (24GB VRAM, sm_89)
- ✅ RTX A6000 (48GB VRAM, sm_86)
- ✅ A100 (40GB/80GB VRAM, sm_80)
- ✅ H100 (80GB VRAM, sm_90)

## Setup

### 1. Create Conda Environment

```bash
cd AgentFormer
conda env create -f environment.yml
conda activate agentformer_py310
```

### 2. Install PyTorch with CUDA Support

**For RTX 5090 (Compute Capability 12.0) or newer GPUs:**
```bash
# RTX 5090 requires PyTorch 2.9+ with CUDA 12.8 for sm_120 support
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
```

**For older GPUs (RTX 3090, 4090, etc.):**
```bash
# Older GPUs can use PyTorch 2.0.1 with CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

**Check your GPU compute capability:**
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

### 3. Install OpenMMLab Dependencies

**For PyTorch 2.9+ (RTX 5090):**
```bash
# Install OpenMMLab packages (mmcv will build from source, takes ~30 minutes)
pip install --no-cache-dir mmcv==2.1.0 --no-binary mmcv
pip install mmdet==3.3.0 mmdet3d==1.4.0 mmengine
pip install 'numpy<2.0'  # Fix numpy version for compatibility
```

**For PyTorch 2.0.1 (Older GPUs):**
```bash
# Install mmcv-full (pre-built wheel for CUDA 11.8, PyTorch 2.0)
pip install --only-binary=mmcv-full mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html

# Install mmdet
pip install mmdet==2.28.2

# Install other OpenMMLab packages
pip install mmengine mmsegmentation==0.30.0
```

### 4. Install Additional Dependencies

```bash
# Install nuScenes devkit and utilities
pip install nuscenes-devkit easydict glob2
```

### 5. Verify Installation

```bash
# Test that all imports work
python -c "import mmcv; import mmdet; import mmengine; print('✓ All packages imported successfully')"

# Test the data collection script
python scripts/collect_used_samples.py --help
```

**Expected output:** You should see the help message without any import errors.

## Step 1: Download nuScenes Dataset

```bash
# Download nuScenes v1.0 mini split (10 scenes, ~4GB)
python download_nuscenes.py --dataroot nuscenes

# Or download full trainval split (~350GB)
# python download_nuscenes.py --dataroot nuscenes --version v1.0-trainval
```

you can download from google drive too
`gdown -f "link" `

extract the `.tar` under `AgentFormer/nuscenes`


## Step 2: Preprocess Data

```bash
python data/process_nuscenes.py --data_root nuscenes
# Generate info pickle files for trajectory prediction
python scripts/gen_info_subset.py --version v1.0-trainval
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

### Installation Issues

**CUDA version mismatch during mmcv-full installation:**
- Solution: Use the pre-built wheel with `--only-binary=mmcv-full` flag as shown above
- Do NOT try to compile from source

**Import error: "No module named 'mmdet3d'":**
- Check that the `.pth` file was created correctly
- Verify: `cat $CONDA_PREFIX/lib/python3.10/site-packages/mmdetection3d.pth`
- Should output: `/workspace/vision-augmented-agentformer/mmdetection3d`

**NumPy version errors:**
- Make sure you downgraded to numpy<2.0
- Check version: `python -c "import numpy; print(numpy.__version__)"`
- Should be 1.26.4 or similar

**Missing module errors (easydict, glob2, etc.):**
- Re-run: `pip install nuscenes-devkit easydict glob2`

### Runtime Issues

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

## Installed Package Versions

After following the setup instructions, you should have:

**For RTX 5090 (PyTorch 2.9+):**
| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10.x | From conda environment |
| PyTorch | 2.9.0 | With CUDA 12.8 support for RTX 5090 |
| torchvision | 0.24.0 | Compatible with PyTorch 2.9.0 |
| mmcv | 2.1.0 | Built from source (~30 min compile time) |
| mmdet | 3.3.0 | Object detection framework |
| mmdet3d | 1.4.0 | 3D object detection framework |
| mmengine | 0.10.7 | OpenMMLab engine |
| nuscenes-devkit | 1.1.11 | nuScenes dataset toolkit |
| numpy | 1.26.4 | Downgraded for compatibility |
| easydict | 1.13 | Configuration management |
| glob2 | 0.7 | File pattern matching |

**For Older GPUs (PyTorch 2.0.1):**
| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.10.x | From conda environment |
| PyTorch | 2.0.1 | With CUDA 11.8 support |
| torchvision | 0.15.2 | Compatible with PyTorch 2.0.1 |
| mmcv-full | 1.7.2 | Pre-built wheel (no compilation) |
| mmdet | 2.28.2 | Object detection framework |
| mmengine | 0.10.7 | OpenMMLab engine |
| mmsegmentation | 0.30.0 | Semantic segmentation framework |
| nuscenes-devkit | 1.1.11 | nuScenes dataset toolkit |
| numpy | 1.26.4 | Downgraded for compatibility |
| easydict | 1.13 | Configuration management |
| glob2 | 0.7 | File pattern matching |
| mmdet3d | 0.17.3 (local) | Via Python path, no compilation |

**Note:** For RTX 5090 systems, use mmdet3d 1.4.0 from pip which is compatible with PyTorch 2.9+ and mmcv 2.1.0.
