# Pretrained Models

This directory contains pretrained AgentFormer models trained on nuScenes dataset (10% subset, 85 scenes).

## Vanilla AgentFormer (Baseline)

Trained without BEV visual features using only trajectory history and HD map.

### Performance
- **ADE (Average Displacement Error):** 4.31m
- **FDE (Final Displacement Error):** 8.37m

### Available Models

| Model | Epochs | Size | Description |
|-------|--------|------|-------------|
| `vanilla/stage1_epoch30.p` | 30 | 74MB | Stage 1: VAE pre-training |
| `vanilla/stage2_epoch50.p` | 50 | 4.0MB | Stage 2: Full model with DLow sampler |

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
│   ├── stage1_epoch30.p  # VAE pre-training checkpoint
│   └── stage2_epoch50.p  # Full model checkpoint
└── bev/                  # Coming soon
    ├── stage1_epoch30.p
    └── stage2_epoch50.p
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
