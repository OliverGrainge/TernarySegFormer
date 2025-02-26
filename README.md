# Ternary SegFormer

This repository contains the implementation of a 2-bit ternary quantized SegFormer model for semantic segmentation. The project focuses on model compression while maintaining segmentation performance.

## Overview

The repository provides tools for:
- Training SegFormer models on ImageNet
- Real-time semantic segmentation using webcam input
- Support for both standard and ternary (2-bit) quantized SegFormer variants

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ternary-segformer
cd ternary-segformer

# Install dependencies
pip install -r requirements.txt
```


### Training

Train the model using the following command:

```bash
python train.py --training_method imagenet --model [segformer|ternary_segformer]
```

Parameters:
- `--training_method`: Currently supports 'imagenet' (required)
- `--model`: Choose between 'segformer' (standard) or 'ternary_segformer' (2-bit quantized) (required)

### Real-time Demo

Run real-time semantic segmentation using your webcam:

```bash
python demo_online.py --model [segformer|ternary_segformer] --checkpoint_path path/to/checkpoint --device [cpu|cuda:0]
```

Parameters:
- `--model`: Choose between 'segformer' or 'ternary_segformer'
- `--checkpoint_path`: Path to the trained model checkpoint
- `--device`: Device to run inference on (default: 'cpu')

## Model Architecture

The project implements two main variants:
1. Standard SegFormer: Based on the original SegFormer architecture
2. Ternary SegFormer: A 2-bit quantized version for reduced model size and faster inference

## Training Details

- Uses PyTorch Lightning for training
- Supports mixed precision training (bfloat16)
- Includes learning rate monitoring and model checkpointing
- Integration with Weights & Biases for experiment tracking

## Requirements

- PyTorch
- PyTorch Lightning
- Transformers
- OpenCV
- Weights & Biases
- NumPy

## License

MIT License
