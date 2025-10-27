# Quick Setup Guide

This guide will help you set up the Vision-Augmented AgentFormer on a new machine.

## Prerequisites Checklist

- [ ] Linux machine (Ubuntu 18.04+)
- [ ] NVIDIA GPU with CUDA support (6GB+ VRAM)
- [ ] ~50GB free disk space
- [ ] Internet connection for downloading dataset and packages

## Step-by-Step Setup

### 1. Install Miniconda (if not already installed)

```bash
# Download Miniconda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts, restart terminal
source ~/.bashrc
```

### 2. Clone This Repository

```bash
# Clone to your desired location
git clone https://github.com/YOUR_USERNAME/BTP-Vision-Augmented-Trajectory-Prediction.git
cd BTP-Vision-Augmented-Trajectory-Prediction/AgentFormer
```

### 3. Create Conda Environment

```bash
# Create environment with Python 3.7
conda create -n agentformer python=3.7 -y
conda activate agentformer

# Verify CUDA version (must be 11.0+)
nvidia-smi
```

**Note**: This setup has been tested with CUDA 13.0 and NVIDIA driver 580.95.05.

### 4. Install PyTorch with CUDA 11.7

```bash
# Install PyTorch 1.13.1 with CUDA 11.7 support
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 1.13.1+cu117
CUDA Available: True
```

### 5. Install MMCV and MMDetection Suite

**Important**: Install in this exact order to avoid dependency conflicts.

```bash
# Install MMCV 2.0.0 (pre-built wheel for PyTorch 1.13 + CUDA 11.7)
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html

# Install MMEngine
pip install mmengine==0.10.7

# Install MMDetection
pip install mmdet==3.3.0

# Install MMDetection3D
pip install mmdet3d==1.4.0
```

Verify installation:
```bash
python -c "import mmcv; print(f'MMCV: {mmcv.__version__}'); import mmdet; print(f'MMDet: {mmdet.__version__}'); import mmdet3d; print(f'MMDet3D: {mmdet3d.__version__}')"
```

Expected output:
```
MMCV: 2.0.0
MMDet: 3.3.0
MMDet3D: 1.4.0
```

### 6. Install Other Dependencies

```bash
# Install all other required packages
pip install -r requirements.txt

# Note: PyTorch and MMCV are already installed, pip will skip them
```

### 7. Compile CUDA Extensions (Optional)

**Note**: Only needed if you want to use on-the-fly BEV feature computation. Skip if using pre-computed features.

```bash
# Set CUDA_HOME environment variable
export CUDA_HOME=/usr/local/cuda

# Compile voxel pooling CUDA extension
python setup.py develop
```

### 8. Download nuScenes Dataset

**Important**: You only need 10% of the dataset (~30GB) for this project.

#### Method 1: Direct Download (Recommended)

1. Go to https://www.nuscenes.org/nuscenes
2. Create account and sign in
3. Go to "Download" page
4. Download **only these files**:
   - `v1.0-trainval_meta.tgz` (~2GB) - Metadata
   - `v1.0-trainval01_blobs.tgz` through `v1.0-trainval10_blobs.tgz` (~30GB total) - First 10% of data

5. Extract to `nuscenes/` directory:
```bash
# Create dataset directory
mkdir -p nuscenes
cd nuscenes

# Extract metadata
tar -xzf ../v1.0-trainval_meta.tgz

# Extract data blobs (1-10 only)
for i in {01..10}; do
    tar -xzf ../v1.0-trainval${i}_blobs.tgz
done

cd ..
```

### 9. Verify Dataset Structure

After extraction, verify your directory structure:

```bash
ls nuscenes/
```

Expected structure:
```
nuscenes/
├── v1.0-trainval/          # Metadata JSON files
│   ├── attribute.json
│   ├── calibrated_sensor.json
│   ├── category.json
│   ├── ego_pose.json
│   ├── instance.json
│   ├── log.json
│   ├── map.json
│   ├── sample.json
│   ├── sample_annotation.json
│   ├── sample_data.json
│   ├── scene.json
│   ├── sensor.json
│   └── visibility.json
├── samples/                # Sensor data
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── LIDAR_TOP/
│   └── RADAR_*/
├── sweeps/                 # Additional sweeps
│   └── [similar structure]
└── maps/                   # HD maps
    ├── basemap/
    └── expansion/
```

### 10. Filter Dataset to 10% Subset

```bash
# Create filtered metadata for available scenes
python scripts/filter_nuscenes_subset.py
```

This creates `data/v1.0-trainval-subset/` with metadata for 85 scenes.

Expected output:
```
Found 85 scenes with complete data
Copying metadata files to v1.0-trainval-subset...
Created filtered subset with 85 scenes
```

### 11. Generate Processed Dataset Files

```bash
# Generate AgentFormer-specific data files
python scripts/gen_info_subset.py
```

This creates pickle files in `datasets/nuscenes_pred/`:
- `nuscenes_infos_train_subset.pkl` (2,462 samples)
- `nuscenes_infos_val_subset.pkl` (914 samples)

Expected output:
```
Processing train split...
Saved to datasets/nuscenes_pred/nuscenes_infos_train_subset.pkl
Train: 2462 samples from 62 scenes

Processing val split...
Saved to datasets/nuscenes_pred/nuscenes_infos_val_subset.pkl
Val: 914 samples from 23 scenes

Total: 3376 samples from 85 scenes
```

### 12. Pre-compute BEV Features (Recommended)

Pre-computing BEV features saves GPU memory (3-4GB) and training time.

```bash
# Extract BEV features for train_subset (~1-2 hours)
conda run -n agentformer python scripts/precompute_bev_features.py --split train_subset --gpu 0

# Extract BEV features for val_subset (~20-30 minutes)
conda run -n agentformer python scripts/precompute_bev_features.py --split val_subset --gpu 0
```

This creates `bev_features/train_subset/` and `bev_features/val_subset/` directories (~24GB total).

See `PRECOMPUTED_BEV_GUIDE.md` for more details.

### 13. Verify Setup with Quick Test

```bash
# Run quick pipeline test (~5-10 minutes)
conda run -n agentformer python quick_test_pipeline.py
```

Expected output:
```
============================================================
STAGE 1: VAE Pre-training Test (50 iterations)
============================================================
...
✓ Stage 1 checkpoint saved

============================================================
STAGE 2: DLow Trajectory Sampler Test (50 iterations)
============================================================
...
✓ Stage 2 checkpoint saved

============================================================
STAGE 3: Evaluation Test (20 samples)
============================================================
...
✓ Evaluation complete!

======================================================================
 ✓ PIPELINE TEST COMPLETE - All stages working!
======================================================================
```

### 14. Start Training

If the quick test passes, you're ready to train:

```bash
# Baseline AgentFormer (without BEV features)
conda run -n agentformer python train.py --cfg nuscenes_baseline --gpu 0

# Vision-Augmented AgentFormer (with pre-computed BEV features)
conda run -n agentformer python train.py --cfg nuscenes_bev --gpu 0
```

## Troubleshooting Setup Issues

### CUDA Not Available

```bash
# Check NVIDIA driver
nvidia-smi

# If command not found, install NVIDIA drivers
sudo ubuntu-drivers autoinstall
sudo reboot

# After reboot, reinstall PyTorch with correct CUDA version
conda activate agentformer
pip uninstall torch torchvision
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

### MMCV Import Errors

```bash
# Ensure correct MMCV version
conda activate agentformer
pip uninstall mmcv mmcv-full
pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
```

### Dataset Not Found

```bash
# Check if dataset extracted correctly
ls nuscenes/v1.0-trainval/

# If empty, re-extract
cd nuscenes
tar -xzf ../v1.0-trainval_meta.tgz
for i in {01..10}; do tar -xzf ../v1.0-trainval${i}_blobs.tgz; done
```

### Out of Disk Space

```bash
# Check available space
df -h

# Clean up if needed
conda clean --all
pip cache purge

# Or mount additional storage for datasets
```

### MMDetection Build Errors

If you encounter build errors with mmdet or mmdet3d:

```bash
# Install build dependencies
conda install -c conda-forge gxx_linux-64 gcc_linux-64

# Reinstall mmdet/mmdet3d
pip install --no-cache-dir mmdet==3.3.0
pip install --no-cache-dir mmdet3d==1.4.0
```

## System Requirements

### Tested Configuration

- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.7.12
- **CUDA**: 13.0 (driver 580.95.05)
- **GPU**: NVIDIA RTX 4050 Laptop GPU (6GB VRAM)
- **PyTorch**: 1.13.1+cu117
- **MMCV**: 2.0.0

### Minimum Requirements

- **RAM**: 16GB (8GB RAM + 8GB swap recommended)
- **GPU Memory**: 6GB VRAM minimum
- **Disk Space**: 50GB (30GB for dataset + 20GB for features and checkpoints)
- **CUDA**: 11.0 or higher

## Next Steps

Once setup is complete, see the following guides:

- `README.md` - Project overview and quick start
- `README_MODIFICATIONS.md` - Detailed code changes and architecture
- `PRECOMPUTED_BEV_GUIDE.md` - Using pre-computed BEV features
- Training configurations in `cfg/` directory

## Estimated Time

- Setup (steps 1-6): ~30 minutes
- Dataset download (step 8): ~2-4 hours (depends on internet speed)
- Dataset processing (steps 9-11): ~10 minutes
- BEV feature extraction (step 12): ~1-2 hours
- Quick test (step 13): ~5-10 minutes

**Total**: ~4-7 hours (mostly waiting for downloads and feature extraction)

## Verification Checklist

Before starting full training, verify:

- [ ] `conda activate agentformer` works
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` returns `True`
- [ ] `nuscenes/v1.0-trainval/` exists with JSON files
- [ ] `data/v1.0-trainval-subset/` exists
- [ ] Files exist:
  - [ ] `datasets/nuscenes_pred/nuscenes_infos_train_subset.pkl`
  - [ ] `datasets/nuscenes_pred/nuscenes_infos_val_subset.pkl`
  - [ ] `bev_features/train_subset/` (if using pre-computed features)
- [ ] `python quick_test_pipeline.py` completes successfully
- [ ] GPU memory: `nvidia-smi` shows at least 6GB available

If all checks pass, you're ready to train!
