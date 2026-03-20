import os
import tensorflow as tf
import numpy as np
import h5py
from src.data_loader import RadioMLDataLoader
from src.resnet_lstm_v9 import build_resnet_lstm_v9

# 57-Class Master List
MASTER_LIST = [
    'ook', '4ask', '8ask', '16ask', '32ask', '64ask', '2fsk', '2gfsk', '2msk', '2gmsk',
    '4fsk', '4gfsk', '4msk', '4gmsk', '8fsk', '8gfsk', '8msk', '8gmsk', '16fsk', '16gfsk',
    '16msk', '16gmsk', 'bpsk', 'qpsk', '8psk', '16psk', '32psk', '64psk', '16qam', '32qam',
    '32qam_cross', '64qam', '128qam_cross', '256qam', '512qam_cross', '1024qam', 'ofdm-64',
    'ofdm-72', 'ofdm-128', 'ofdm-180', 'ofdm-256', 'ofdm-300', 'ofdm-512', 'ofdm-600',
    'ofdm-900', 'ofdm-1024', 'ofdm-1200', 'ofdm-2048', 'fm', 'am-dsb-sc', 'am-dsb',
    'am-lsb', 'am-usb', 'lfm_data', 'lfm_radar', 'chirpss', 'tone'
]

DATASET = "data/VDF_SPECTER_GOLDEN.h5"
MODEL_PATH = "models/vanguard_v9_specialist_final.keras"

def main():
    print(f"Opal Vanguard: Reverse-Engineering Hardware Labels")
    
    # 1. Load Model
    model = build_resnet_lstm_v9(num_classes=57)
    model.load_weights(MODEL_PATH)
    print(f"Loaded {MODEL_PATH}")

    # 2. Extract Samples
    with h5py.File(DATASET, 'r') as f:
        # We take 5 samples from every index group (0-23)
        indices = []
        y_all = f['Y'][:]
        y_classes = np.argmax(y_all, axis=1)
        
        for i in range(24):
            match_idx = np.where(y_classes == i)[0]
            if len(match_idx) > 0:
                indices.extend(match_idx[:5])
        
        x_raw = f['X'][sorted(indices)]
        y_raw = f['Y'][sorted(indices)]

    # 3. Predict with Adaptive Scaling
    # First, scale each sample to [-1, 1] based on its own peak
    x_adaptive = []
    for sample in x_raw:
        max_val = np.max(np.abs(sample))
        if max_val > 0:
            sample = sample / max_val
        x_adaptive.append(sample)
    x_adaptive = np.array(x_adaptive)
    
    # Then apply the project standard Soft-Clip
    x_norm = x_adaptive / (1.0 + np.abs(x_adaptive))
    preds = model.predict(x_norm, batch_size=32)
    
    # 4. Report
    print("\n--- Model's Truth Mapping ---")
    current_idx = -1
    y_true_classes = np.argmax(y_raw, axis=1)
    
    for i in range(len(indices)):
        hw_idx = y_true_classes[i]
        if hw_idx != current_idx:
            print(f"\nHardware Index {hw_idx} is predicted as:")
            current_idx = hw_idx
            
        top_idx = np.argmax(preds[i])
        confidence = preds[i][top_idx]
        print(f"  - {MASTER_LIST[top_idx]} ({confidence:.2f})")

if __name__ == "__main__":
    main()
