#!/bin/bash

# --- Opal Vanguard GPU Launcher ---
# Explicitly mapping pip-installed NVIDIA libraries to LD_LIBRARY_PATH
# to ensure TensorFlow 2.21+ can see the Blackwell GPU.

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/home/dev2/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/home/dev2/.local/lib/python3.12/site-packages/nvidia/cublas/lib:/home/dev2/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:/home/dev2/.local/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/home/dev2/.local/lib/python3.12/site-packages/nvidia/cuda_nvcc/nvvm/lib64

echo "--------------------------------------------------"
echo "Opal Vanguard: Hardware Verification"
GPU_COUNT=$(python3 -c "import tensorflow as tf; print(len(tf.config.list_physical_devices('GPU')))" 2>/dev/null)
echo "Num GPUs Available: $GPU_COUNT"
echo "--------------------------------------------------"

if [ "$GPU_COUNT" -eq "0" ]; then
    echo "WARNING: No GPUs detected. Training will run on CPU (VERY SLOW)."
else
    echo "SUCCESS: GPU acceleration active."
fi

echo ""
read -p "Do you wish to proceed with training? (y/n): " confirm
if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
    echo "Opal Vanguard: Starting Heavy Training Run..."
    python3 train_resnet.py
else
    echo "Training aborted by user."
    exit 1
fi
