# Vision-Augmented AgentFormer

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch 1.9.0](https://img.shields.io/badge/pytorch-1.9.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

AgentFormer with BEVDepth visual features for improved trajectory prediction on nuScenes dataset.

> **Note**: This is a modified version of the original [AgentFormer](https://github.com/Khrylx/AgentFormer) repository. See [README_ORIGINAL.md](README_ORIGINAL.md) for the original README.

## Overview

This repository extends AgentFormer to integrate visual features from BEVDepth, enabling trajectory prediction that leverages both scene geometry and visual context from multi-view cameras.

**Key Features:**
- 🔄 **Conditional BEVDepth Integration**: Toggle between baseline and vision-augmented modes
- 📊 **Ablation Study Ready**: Compare baseline vs vision-augmented on identical dataset
- ⚡ **Quick Pipeline Test**: Verify complete workflow in 5-10 minutes
- 📦 **10% Dataset Support**: Works with 85-scene subset (~30GB instead of 300GB)
- 📝 **Complete Documentation**: Detailed setup, usage, and troubleshooting

## Quick Start

```bash
# 1. Clone and setup environment
git clone https://github.com/YOUR_USERNAME/vision-augmented-agentformer.git
cd vision-augmented-agentformer
conda create -n agentformer python=3.7 -y
conda activate agentformer

# 2. Install dependencies
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
pip install nuscenes-devkit

# 3. Download nuScenes (10% subset)
# Download v1.0-trainval_meta.tgz and v1.0-trainval_01.tgz from https://www.nuscenes.org/
# Extract to datasets/nuscenes_pred/

# 4. Filter dataset to available scenes
python scripts/filter_nuscenes_subset.py
python scripts/gen_info_subset.py

# 5. Quick test (5-10 minutes)
python quick_test_pipeline.py

# 6. Start training
python train.py --cfg nuscenes_5sample_agentformer_pre --gpu 0
```

## Documentation

📚 **Comprehensive documentation available:**

| Document | Purpose |
|----------|---------|
| **[SETUP_GUIDE.md](SETUP_GUIDE.md)** | Step-by-step setup for new machines |
| **[README_MODIFICATIONS.md](README_MODIFICATIONS.md)** | Detailed documentation of all changes |
| **[CHANGES_SUMMARY.md](CHANGES_SUMMARY.md)** | Summary of modified files |
| **[GITHUB_UPLOAD_INSTRUCTIONS.md](GITHUB_UPLOAD_INSTRUCTIONS.md)** | GitHub upload guide |
| **[README_ORIGINAL.md](README_ORIGINAL.md)** | Original AgentFormer README |

## System Requirements

- **GPU**: 6GB+ VRAM (tested on RTX 4050 Mobile 6GB)
- **Storage**: ~50GB
- **RAM**: 16GB+ recommended
- **OS**: Linux (Ubuntu 18.04+)
- **CUDA**: 11.1+

**Training Time:**
- RTX 4050 Mobile (6GB): ~5.5 hours/epoch
- RTX 5090 (32GB): ~30 minutes/epoch (estimated)

## Key Modifications

### 1. Conditional BEVDepth Integration

Toggle between baseline and vision-augmented modes:

```python
# Config file: set use_bev flag
use_bev: false  # Baseline (trajectory + map only)
use_bev: true   # Vision-augmented (+ BEV visual features)
```

### 2. Dataset Filtering

Support for 10% nuScenes subset (85 scenes):
- Train: 2462 samples from 62 scenes
- Val: 914 samples from 23 scenes

### 3. Training Pipeline Fixes

Fixed bugs in original code:
- Variable naming issues
- Missing forward() call
- nuScenes logging errors

### 4. Quick Pipeline Test

New `quick_test_pipeline.py` for fast verification:
- Tests all 3 stages in 5-10 minutes
- Runs 50 iterations per stage
- Validates complete workflow

## Usage

### Baseline Training (No BEVDepth)

```bash
# Stage 1: VAE pre-training
python train.py --cfg nuscenes_5sample_agentformer_pre --gpu 0

# Stage 2: DLow trajectory sampler
python train.py --cfg nuscenes_5sample_agentformer --gpu 0

# Stage 3: Evaluation
python test.py --cfg nuscenes_5sample_agentformer --gpu 0
```

### Vision-Augmented Training (With BEVDepth)

Edit config file to add:
```yaml
use_bev: true
data_root_nuscenes_pred: datasets/nuscenes_pred
info_train_path: data/nuscenes_infos_train_subset.pkl
info_val_path: data/nuscenes_infos_val_subset.pkl
```

Then run the same training commands.

## Project Structure

```
vision-augmented-agentformer/
├── cfg/                           # Configuration files
├── data/
│   ├── dataloader.py             # ✨ Modified: conditional BEV loading
│   ├── nuscenes_pred_split.py    # ✨ Modified: 85-scene filtering
│   └── bev_utils.py              # ✨ New: BEV utilities
├── model/
│   └── bev/                      # ✨ New: BEVDepth models
├── scripts/                       # ✨ New: Dataset processing
│   ├── filter_nuscenes_subset.py
│   └── gen_info_subset.py
├── train.py                       # ✨ Modified: Fixed bugs
├── quick_test_pipeline.py        # ✨ New: Quick verification
└── README.md                      # This file
```

## Troubleshooting

**CUDA Out of Memory**: Reduce batch size in config
```yaml
batch_size: 4  # or 2, 1
```

**Dataset Not Found**: Re-run filtering scripts
```bash
python scripts/filter_nuscenes_subset.py
python scripts/gen_info_subset.py
```

**Import Errors**: Reinstall dependencies
```bash
conda activate agentformer
pip install -r requirements.txt --force-reinstall
```

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for more troubleshooting.

## Citation

```bibtex
@inproceedings{yuan2021agent,
  title={AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting},
  author={Yuan, Ye and Weng, Xinshuo and Ou, Yanglan and Kitani, Kris},
  booktitle={ICCV},
  year={2021}
}

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

- [AgentFormer](https://github.com/Khrylx/AgentFormer) - Original implementation
- [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) - Visual feature extraction
- [nuScenes](https://www.nuscenes.org/) - Dataset

---

**Status**: ✅ Pipeline verified | 🚧 Training ready | 📊 Ablation study pending
