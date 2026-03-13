#!/bin/bash

# --- Opal Vanguard GPU Launcher ---
# Explicitly mapping pip-installed NVIDIA libraries to LD_LIBRARY_PATH
# to ensure TensorFlow 2.21+ can see the Blackwell GPU.

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dev2/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/dev2/.local/lib/python3.12/site-packages/nvidia/cublas/lib:/home/dev2/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/dev2/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib

echo "Opal Vanguard: Attempting GPU Detection..."
python3 -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"

echo "Opal Vanguard: Starting Heavy Training Run..."
python3 train_resnet.py
