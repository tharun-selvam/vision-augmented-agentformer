# BTP: Vision-Augmented Trajectory Prediction

This repository contains my B.Tech Project on integrating vision features into trajectory prediction using AgentFormer and BEVDepth.

## 📁 Repository Structure

```
BTP/
├── AgentFormer/           # Vision-augmented AgentFormer (main project)
├── BEVDepth/              # BEVDepth for BEV feature extraction
├── mmdetection3d/         # MMDetection3D utilities
├── AgentFormer.pdf        # AgentFormer paper (download separately)
└── visionTrap.pdf         # VisionTrap paper (download separately)
```

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/BTP.git
cd BTP
```

### 2. Setup AgentFormer (Main Project)

```bash
cd AgentFormer
conda create -n agentformer python=3.7 -y
conda activate agentformer

# Install PyTorch
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
pip install -r requirements.txt
pip install nuscenes-devkit
```

### 3. Download nuScenes Dataset

1. Go to https://www.nuscenes.org/nuscenes#download
2. Download:
   - `v1.0-trainval_meta.tgz` (metadata)
   - `v1.0-trainval01.tgz` to `v1.0-trainval10.tgz` (10% subset, ~30GB)
3. Extract to `AgentFormer/datasets/nuscenes_pred/`

### 4. Setup Dataset

```bash
cd AgentFormer
python scripts/filter_nuscenes_subset.py
python scripts/gen_info_subset.py
```

### 5. Train

```bash
# Baseline (trajectory + map only)
python train.py --cfg nuscenes_5sample_agentformer --gpu 0

# Vision-augmented (+ BEV features)
python train.py --cfg nuscenes_5sample_agentformer_pre_bev --gpu 0
```

## 📚 Documentation

| Project | Documentation |
|---------|---------------|
| **AgentFormer** | [AgentFormer/README.md](AgentFormer/README.md) |
| Setup Guide | [AgentFormer/SETUP_GUIDE.md](AgentFormer/SETUP_GUIDE.md) |
| Pre-computed Features | [AgentFormer/PRECOMPUTED_BEV_GUIDE.md](AgentFormer/PRECOMPUTED_BEV_GUIDE.md) |
| Modifications | [AgentFormer/README_MODIFICATIONS.md](AgentFormer/README_MODIFICATIONS.md) |

## ✨ Key Features

### Vision-Augmented AgentFormer

- **Conditional BEV Integration**: Toggle between baseline and vision-augmented modes
- **Pre-computed Features**: Save GPU memory (3-4GB) and training time
- **10% Dataset Support**: Works with 85-scene subset (~30GB instead of 300GB)
- **Memory Efficient**: Runs on 6GB GPU (RTX 4050 Mobile)

### Performance

| Mode | GPU Memory | Train Time/Epoch | Disk Space |
|------|-----------|------------------|------------|
| **Baseline** | ~2 GB | 4 hours | 0 GB |
| **Vision (on-the-fly)** | ~5.5 GB | 5.5 hours | 0 GB |
| **Vision (pre-computed)** | ~2 GB | 4.5 hours | 24 GB |

*Measured on RTX 4050 Mobile (6GB) with train_subset (2,462 samples)*

## 🔧 System Requirements

- **GPU**: 6GB+ VRAM (tested on RTX 4050 Mobile 6GB)
- **Storage**: ~50GB (30GB dataset + 24GB features)
- **RAM**: 16GB+ recommended
- **OS**: Linux (Ubuntu 18.04+)
- **CUDA**: 11.1+

## 📦 Subprojects

### 1. AgentFormer (Main)

Modified AgentFormer with BEVDepth integration for trajectory prediction.

**Key Changes:**
- Conditional BEV feature integration
- Pre-computed feature support
- Dataset filtering for 10% subset
- Memory-efficient training pipeline

See [AgentFormer/README.md](AgentFormer/README.md) for details.

### 2. BEVDepth

BEVDepth repository for BEV feature extraction. Used to pre-compute visual features.

**Setup:**
```bash
cd BEVDepth
# Follow BEVDepth setup instructions if needed
```

### 3. MMDetection3D

MMDetection3D utilities for 3D object detection (if needed).

## 📝 Usage Examples

### Baseline Mode (No Vision)

```bash
cd AgentFormer
python train.py --cfg nuscenes_5sample_agentformer --gpu 0
```

### Vision-Augmented (On-the-fly)

```bash
cd AgentFormer
python train.py --cfg nuscenes_5sample_agentformer_pre --gpu 0
```

### Vision-Augmented (Pre-computed)

```bash
cd AgentFormer

# 1. Extract BEV features once
python scripts/precompute_bev_features.py --split train_subset --gpu 0

# 2. Train with pre-computed features
python train.py --cfg nuscenes_5sample_agentformer_pre_bev --gpu 0
```

## 🐛 Troubleshooting

### CUDA Out of Memory

1. Use pre-computed BEV features
2. Reduce batch size in config
3. Use gradient checkpointing

### BEVDepth Import Error

If using pre-computed features:
```yaml
# In config file
use_bev: true
use_precomputed_bev: true  # BEVDepth not required
```

If computing on-the-fly:
```bash
# Make sure BEVDepth is in path
cd BTP
export PYTHONPATH=$PYTHONPATH:$(pwd)/BEVDepth
```

### Dataset Not Found

1. Download nuScenes from https://www.nuscenes.org/
2. Extract to `AgentFormer/datasets/nuscenes_pred/`
3. Run `python scripts/filter_nuscenes_subset.py`

## 📄 License

- **AgentFormer**: MIT License (see AgentFormer/LICENSE)
- **BEVDepth**: Apache 2.0 License
- **MMDetection3D**: Apache 2.0 License

## 🙏 Acknowledgments

- [AgentFormer](https://github.com/Khrylx/AgentFormer) by Ye Yuan
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) by Megvii
- [nuScenes](https://www.nuscenes.org/) dataset by Motional

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: PDFs and large dataset files are not included in the repository due to size constraints. Download them separately from the links above.
