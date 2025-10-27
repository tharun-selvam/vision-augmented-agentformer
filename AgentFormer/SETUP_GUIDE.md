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
git clone https://github.com/YOUR_USERNAME/vision-augmented-agentformer.git
cd vision-augmented-agentformer
```

### 3. Create Conda Environment

```bash
# Create environment with Python 3.7
conda create -n agentformer python=3.7 -y
conda activate agentformer

# Verify CUDA version (must be 11.1+)
nvidia-smi
```

### 4. Install PyTorch with CUDA

```bash
# For CUDA 11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 1.9.0+cu111
CUDA Available: True
```

### 5. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Install nuScenes devkit
pip install nuscenes-devkit
```

### 6. Download nuScenes Dataset

**Important**: You only need 10% of the dataset (~30GB) for this project.

#### Method 1: Direct Download (Recommended)

1. Go to https://www.nuscenes.org/nuscenes
2. Create account and sign in
3. Go to "Download" page
4. Download:
   - `v1.0-trainval_meta.tgz` (~2GB) - Metadata
   - `v1.0-trainval_01.tgz` (~30GB) - Part 1 only

#### Method 2: Using wget

```bash
# Create dataset directory
mkdir -p datasets/nuscenes_pred
cd datasets/nuscenes_pred

# Download metadata (replace with actual download link from nuScenes)
wget <NUSCENES_META_URL> -O v1.0-trainval_meta.tgz

# Download Part 1 only (replace with actual download link)
wget <NUSCENES_PART1_URL> -O v1.0-trainval_01.tgz

# Extract
tar -xzf v1.0-trainval_meta.tgz
tar -xzf v1.0-trainval_01.tgz

# Return to project root
cd ../..
```

### 7. Verify Dataset Structure

After extraction, verify your directory structure:

```bash
ls -R datasets/nuscenes_pred/
```

Expected structure:
```
datasets/nuscenes_pred/
├── v1.0-trainval/
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
├── samples/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   ├── CAM_BACK_RIGHT/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── LIDAR_TOP/
│   └── RADAR_*/
├── sweeps/
│   └── [similar structure]
└── maps/
    ├── basemap/
    └── expansion/
```

### 8. Filter Dataset to 85 Available Scenes

```bash
# Run filtering script
python scripts/filter_nuscenes_subset.py
```

Expected output:
```
Found 85 scenes with complete data
Copying metadata files to v1.0-trainval-subset...
Created filtered subset with 85 scenes
```

### 9. Generate .pkl Info Files

```bash
# Generate subset info files
python scripts/gen_info_subset.py
```

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

### 10. Verify Setup with Quick Test

```bash
# Run quick pipeline test (~5-10 minutes)
python quick_test_pipeline.py
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

### 11. Start Training

If the quick test passes, you're ready to train:

```bash
# Stage 1: VAE Pre-training
python train.py --cfg nuscenes_5sample_agentformer_pre --gpu 0
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
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### Import Errors

```bash
# Ensure conda environment is activated
conda activate agentformer

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Dataset Not Found

```bash
# Check if dataset extracted correctly
ls datasets/nuscenes_pred/v1.0-trainval/

# If empty, re-extract
cd datasets/nuscenes_pred
tar -xzf v1.0-trainval_meta.tgz
tar -xzf v1.0-trainval_01.tgz
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

## Next Steps

Once setup is complete, see the main `README_MODIFICATIONS.md` for:
- Full training instructions
- Configuration options
- Ablation study guide
- Performance benchmarks

## Estimated Time

- Setup (steps 1-5): ~30 minutes
- Dataset download (step 6): ~2-4 hours (depends on internet speed)
- Dataset processing (steps 7-9): ~10 minutes
- Quick test (step 10): ~5-10 minutes

**Total**: ~3-5 hours (mostly waiting for downloads)

## Verification Checklist

Before starting full training, verify:

- [ ] `conda activate agentformer` works
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` returns `True`
- [ ] `datasets/nuscenes_pred/v1.0-trainval-subset/` exists
- [ ] Files exist:
  - [ ] `datasets/nuscenes_pred/nuscenes_infos_train_subset.pkl`
  - [ ] `datasets/nuscenes_pred/nuscenes_infos_val_subset.pkl`
  - [ ] `datasets/nuscenes_pred/nuscenes_infos_all_subset.pkl`
- [ ] `python quick_test_pipeline.py` completes successfully
- [ ] GPU memory: `nvidia-smi` shows at least 6GB available

If all checks pass, you're ready to train!
