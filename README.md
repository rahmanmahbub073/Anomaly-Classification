# Network Project with Multi-GPU Training Support

This project builds and trains deep learning models using TensorFlow for a networking task. It supports multiple models such as U-Net, ResNet, Inception, and TCN. The training pipeline leverages TensorFlow's multi-GPU training capabilities for efficient and distributed computation.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Available Models](#available-models)
- [Project Structure](#project-structure)
- [Training and Evaluation](#training-and-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Multi-GPU Training**: Enables multi-GPU support with TensorFlow’s `MirroredStrategy`.
- **Model Selection**: Choose from U-Net, ResNet, Inception, and TCN architectures.
- **Cross-Validation**: 5-fold cross-validation for robust model evaluation.
- **Dynamic Model Loading**: Load models dynamically by name.
- **Plotting and Logging**: Logs results and generates accuracy/loss plots.

## Requirements
- Python 3.9+
- TensorFlow with GPU support (CUDA and cuDNN installed)
- NVIDIA GPUs (4 GPUs recommended for multi-GPU training)
- Additional Python packages: `numpy`, `scipy`, `sklearn`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/network-project.git
   cd network-project
2. Create and activate a virtual environment (optional but recommended):
   python -m venv env
  source env/bin/activate  # On Windows, use `env\Scripts\activate`
3. Install required packages:
   pip install -r requirements.txt
4. Verify GPU setup with TensorFlow:
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
## Usage
1. Training
To start training, run train.py, specifying the model name as an argument:
python train.py --model_name model_unet  # Options: model_unet, model_resnet, model_inception, model_tcn
2. Model Evaluation
Evaluation is automatically run after training. To evaluate a pre-trained model, use evaluate.py:
python evaluate.py --model_name model_unet

## Available Models
U-Net: A convolutional network for tasks requiring precise output shape control.
ResNet: Deep residual network with skip connections.
Inception: Inception-based architecture for capturing multi-scale features.
TCN: Temporal convolutional network for sequential data.

## Project Structure

network-project/
├── data/                         # Data files and datasets
├── models/                       # Model architectures
│   ├── model_unet.py
│   ├── model_resnet.py
│   ├── model_inception.py
│   └── model_tcn.py
├── utils/                        # Utility functions
│   └── logging_utils.py          # Logging and plot-saving utilities
├── logs/                         # Model checkpoints and logs
├── train.py                      # Main training script with multi-GPU support
├── evaluate.py                   # Evaluation script for trained models
|── data_utils.py                # Data loading functions
└── README.md                     # Project documentation

## Training and Evaluation
1. Training: Initiate training by running train.py and specifying your chosen model. This will perform 5-fold cross-validation, save plots, and log results.
2. Evaluation: The evaluate.py script provides detailed metrics, including confusion matrices and classification reports.
