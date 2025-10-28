# Pretrained Models

This directory contains pretrained AgentFormer models trained on nuScenes dataset (10% subset, 85 scenes).

## Vanilla AgentFormer (Baseline)

Trained without BEV visual features using only trajectory history and HD map.

### Performance
- **ADE (Average Displacement Error):** 4.31m
- **FDE (Final Displacement Error):** 8.37m

### Available Models

**Stage 1 (VAE Pre-training):** `vanilla/stage1/`
- `model_0010.p` - Epoch 10 (74MB)
- `model_0020.p` - Epoch 20 (74MB)
- `model_0030.p` - Epoch 30 (74MB) ⭐ Best

**Stage 2 (Full Model with DLow):** `vanilla/stage2/`
- `model_0005.p` - Epoch 5 (4.0MB)
- `model_0010.p` - Epoch 10 (4.0MB)
- `model_0015.p` - Epoch 15 (4.0MB)
- `model_0020.p` - Epoch 20 (4.0MB)
- `model_0025.p` - Epoch 25 (4.0MB)
- `model_0030.p` - Epoch 30 (4.0MB)
- `model_0035.p` - Epoch 35 (4.0MB)
- `model_0040.p` - Epoch 40 (4.0MB)
- `model_0045.p` - Epoch 45 (4.0MB)
- `model_0050.p` - Epoch 50 (4.0MB) ⭐ Best

**Total Size:** 337MB (13 checkpoints)

### Usage

**Evaluate Stage 2 model:**
```bash
python test.py --cfg nuscenes_5sample_agentformer --gpu 0
```

**Continue training from checkpoint:**
```bash
python train.py --cfg nuscenes_5sample_agentformer --gpu 0 --start_epoch 50
```

### Training Details

**Stage 1 (VAE Pre-training):**
- Config: `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer_pre.yml`
- Epochs: 30
- Learning rate: 1e-4
- Optimizer: Adam
- Training time: ~2 hours on RTX 5090

**Stage 2 (Full Model):**
- Config: `cfg/nuscenes/5_sample/nuscenes_5sample_agentformer.yml`
- Epochs: 50
- Learning rate: 1e-4
- Optimizer: Adam
- Training time: ~3 hours on RTX 5090
- Uses pretrained weights from Stage 1

## Vision-Augmented Models (Coming Soon)

Models trained with BEV visual features will be added after training completes.

## Download Instructions

These models are hosted on GitHub Releases due to file size constraints.

**Download via wget:**
```bash
cd AgentFormer/pretrained_models
wget https://github.com/tharun-selvam/vision-augmented-agentformer/releases/download/v1.0/vanilla_models.tar.gz
tar -xzf vanilla_models.tar.gz
```

**Download via browser:**
Visit: https://github.com/tharun-selvam/vision-augmented-agentformer/releases

## File Structure

```
pretrained_models/
├── README.md
├── vanilla/
│   ├── stage1/           # VAE pre-training checkpoints
│   │   ├── model_0010.p
│   │   ├── model_0020.p
│   │   └── model_0030.p
│   ├── stage2/           # Full model checkpoints
│   │   ├── model_0005.p
│   │   ├── model_0010.p
│   │   ├── ...
│   │   └── model_0050.p
│   ├── stage1_epoch30.p  # Symlink to best Stage 1
│   └── stage2_epoch50.p  # Symlink to best Stage 2
└── bev/                  # Coming soon
    ├── stage1/
    └── stage2/
```

## Model Format

Models are saved as Python pickle files (`.p`) containing:
- `model_dict`: Model state dictionary
- `opt_dict`: Optimizer state dictionary
- `scheduler_dict`: Learning rate scheduler state
- `epoch`: Training epoch number

## Citation

If you use these pretrained models, please cite:

```bibtex
@inproceedings{yuan2021agentformer,
  title={AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting},
  author={Yuan, Ye and Weng, Xinshuo and Ou, Yanglan and Kitani, Kris},
  booktitle={ICCV},
  year={2021}
}
```
