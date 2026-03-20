import os
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.data_loader import RadioMLDataLoader
from src.resnet_lstm_v9 import build_resnet_lstm_v9
from src.resnet_lstm_polar_v9 import build_resnet_lstm_polar_v9
from src.resnet_opal_vanguard import build_resnet_vanguard

# 57-Class Master List (V9 Model Vocabulary)
MASTER_LIST = [
    'ook', '4ask', '8ask', '16ask', '32ask', '64ask', '2fsk', '2gfsk', '2msk', '2gmsk',
    '4fsk', '4gfsk', '4msk', '4gmsk', '8fsk', '8gfsk', '8msk', '8gmsk', '16fsk', '16gfsk',
    '16msk', '16gmsk', 'bpsk', 'qpsk', '8psk', '16psk', '32psk', '64psk', '16qam', '32qam',
    '32qam_cross', '64qam', '128qam_cross', '256qam', '512qam_cross', '1024qam', 'ofdm-64',
    'ofdm-72', 'ofdm-128', 'ofdm-180', 'ofdm-256', 'ofdm-300', 'ofdm-512', 'ofdm-600',
    'ofdm-900', 'ofdm-1024', 'ofdm-1200', 'ofdm-2048', 'fm', 'am-dsb-sc', 'am-dsb',
    'am-lsb', 'am-usb', 'lfm_data', 'lfm_radar', 'chirpss', 'tone'
]

# Hardware List (Definitive order from spectrum_sentry.py)
HARDWARE_LIST = [
    '32PSK', '16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK',
    'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM',
    '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM',
    'AM-DSB-WC', 'OOK', '16QAM'
]

DATASET = "data/VDF_SPECTER_GOLDEN.h5"
MODELS_TO_TEST = [
    {"name": "V8.5_ResNet", "path": "models/vanguard_v8_specter_final_57pct.keras", "type": "resnet_v8"},
    {"name": "V9.1_Specialist", "path": "models/vanguard_v9_specialist_final.keras", "type": "iq_only"},
    {"name": "V9.2_Sovereign", "path": "models/vanguard_v9_sovereign_final.keras", "type": "multi_modal"},
    {"name": "V9.4_Transfused", "path": "models/v9_specialist_sovereign_checkpoint.keras", "type": "multi_modal"}
]

def run_evaluation(model_info, x_raw, y_raw):
    print(f"\n--- Evaluating: {model_info['name']} ---")
    
    # 1. Build & Load
    if model_info['type'] == "resnet_v8":
        model = build_resnet_vanguard(num_classes=57)
    elif model_info['type'] == "iq_only":
        model = build_resnet_lstm_v9(num_classes=57)
    else:
        model = build_resnet_lstm_polar_v9(num_classes=57)
    
    model.load_weights(model_info['path'])

    # 2. Prep Inputs
    # Soft-Clip Normalization
    x_norm = x_raw / (1.0 + np.abs(x_raw))
    
    if model_info['type'] == "multi_modal":
        i_comp, q_comp = x_norm[:, :, 0], x_norm[:, :, 1]
        amplitude = np.sqrt(i_comp**2 + q_comp**2)
        amplitude = amplitude / (1.0 + np.abs(amplitude))
        phase = np.arctan2(q_comp, i_comp) / np.pi
        x_polar = np.stack([amplitude, phase], axis=-1)
        inputs = [x_norm, x_polar]
    else:
        inputs = x_norm

    # 3. Predict
    predictions = model.predict(inputs, batch_size=128)
    y_pred_idx = np.argmax(predictions, axis=1)
    y_true_hw_idx = np.argmax(y_raw, axis=1)
    
    # 4. CROSS-VOCABULARY MAPPING
    # Map Hardware True labels to their Master List index
    hw_to_master = {}
    for i, name in enumerate(HARDWARE_LIST):
        normalized_name = name.lower().replace("-", "").replace(" ", "")
        # Find matching name in MASTER_LIST
        for j, m_name in enumerate(MASTER_LIST):
            m_normalized = m_name.lower().replace("-", "").replace(" ", "")
            if normalized_name in m_normalized or m_normalized in normalized_name:
                hw_to_master[i] = j
                break
    
    y_true_master_idx = np.array([hw_to_master.get(idx, -1) for idx in y_true_hw_idx])
    
    # Only calculate accuracy for classes that exist in both lists
    valid_mask = y_true_master_idx != -1
    acc = np.mean(y_pred_idx[valid_mask] == y_true_master_idx[valid_mask]) * 100
    print(f" Vocabulary-Aligned Hardware Accuracy: {acc:.2f}%")
    
    # 5. Confusion Matrix (Mapped to Hardware Names)
    # Filter only samples belonging to our 24 HW classes
    cm = confusion_matrix(y_true_hw_idx[valid_mask], 
                          [next((k for k, v in hw_to_master.items() if v == p), 23) for p in y_pred_idx[valid_mask]], 
                          labels=range(len(HARDWARE_LIST)))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    return cm_norm, acc

def main():
    with h5py.File(DATASET, 'r') as f:
        sample_size = 10000
        indices = np.random.choice(f['X'].shape[0], sample_size, replace=False)
        x_raw = f['X'][sorted(indices)]
        y_raw = f['Y'][sorted(indices)]

    results = []
    for model_info in MODELS_TO_TEST:
        if os.path.exists(model_info['path']):
            res = run_evaluation(model_info, x_raw, y_raw)
            results.append((model_info['name'], res[0], res[1]))

    # Plot
    fig, axes = plt.subplots(1, len(results), figsize=(10 * len(results), 8))
    for i, (name, cm, acc) in enumerate(results):
        sns.heatmap(cm, cmap='magma', ax=axes[i], cbar=False)
        axes[i].set_title(f"{name}\nAlign-Acc: {acc:.1f}%")
    plt.tight_layout()
    plt.savefig("reports/aligned_comparison_matrix.png")
    print("\nSUCCESS: Aligned Comparison saved to reports/aligned_comparison_matrix.png")

if __name__ == "__main__":
    main()
