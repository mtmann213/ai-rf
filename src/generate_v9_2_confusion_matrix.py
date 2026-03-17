import os
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.data_loader import RadioMLDataLoader
from src.resnet_lstm_polar_v9 import build_resnet_lstm_polar_v9

# Configuration
DATASET = "data/VDF_SPECTER_GOLDEN.h5"
WEIGHTS_PATH = "models/vanguard_v9_sovereign_final.keras"
OUTPUT_PNG = "reports/confusion_matrix_v9_2_real.png"
NUM_CLASSES_MODEL = 57
NUM_CLASSES_DATA = 24

def main():
    print(f"Opal Vanguard: Generating V9.2 SOVEREIGN EYE Failure Analysis Matrix")
    
    # 1. Build & Load Model
    model = build_resnet_lstm_polar_v9(iq_shape=(1024, 2), polar_shape=(1024, 2), num_classes=NUM_CLASSES_MODEL)
    model.load_weights(WEIGHTS_PATH)
    print(f"Successfully loaded {WEIGHTS_PATH}")

    # 2. Setup Data Loader
    loader = RadioMLDataLoader(DATASET, num_classes=NUM_CLASSES_DATA)
    with h5py.File(DATASET, 'r') as f:
        n_samples = f['X'].shape[0]
        # Use 10k samples for a fast diagnostic
        sample_size = min(10000, n_samples)
        indices = np.random.choice(n_samples, sample_size, replace=False)
        x_raw = f['X'][sorted(indices)]
        y_raw = f['Y'][sorted(indices)]

    x_norm = loader.normalize(x_raw)
    
    # 3. Calculate Polar Stream
    i_comp = x_norm[:, :, 0]
    q_comp = x_norm[:, :, 1]
    amplitude = np.sqrt(i_comp**2 + q_comp**2)
    phase = np.arctan2(q_comp, i_comp)
    # Normalize
    amplitude = amplitude / (1.0 + np.abs(amplitude))
    phase = phase / np.pi
    x_polar = np.stack([amplitude, phase], axis=-1)
    
    # 4. Predict & Analyze
    print(f"Analyzing {sample_size} hardware snapshots (IQ + Polar)...")
    predictions = model.predict([x_norm, x_polar], batch_size=128)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_raw, axis=1)

    # 5. Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES_DATA))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    # 6. Plotting
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Purples', # Purple for Sovereign Eye era
                xticklabels=loader.modulations, 
                yticklabels=loader.modulations)
    plt.title(f'Vanguard V9.2 Sovereign Eye Hardware Confusion Matrix (Peak Acc: {np.max(y_true == y_pred)*100:.2f}%)')
    plt.ylabel('True Modulation')
    plt.xlabel('Predicted Modulation')
    plt.savefig(OUTPUT_PNG)
    print(f"Success: Confusion Matrix saved to {OUTPUT_PNG}")

    # 7. Report the Top 5 Failures
    print("\n--- Top V9.2 Diagnostic Failures ---")
    confused_pairs = []
    for i in range(NUM_CLASSES_DATA):
        for j in range(NUM_CLASSES_DATA):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((loader.modulations[i], loader.modulations[j], cm[i, j]))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    for i, (m1, m2, count) in enumerate(confused_pairs[:5]):
        print(f" {i+1}. {m1} misidentified as {m2} ({count} times)")

if __name__ == "__main__":
    main()
