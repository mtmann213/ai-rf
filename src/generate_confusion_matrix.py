import os
import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data_loader import RadioMLDataLoader
from resnet_opal_vanguard import build_resnet_vanguard

# Configuration
DATASET = "VDF_SPECTER_GOLDEN.h5"
WEIGHTS_PATH = "vanguard_v8_specter_final_57pct.keras"
OUTPUT_PNG = "confusion_matrix_v8_5_real.png"
NUM_CLASSES_MODEL = 57
NUM_CLASSES_DATA = 24

def main():
    print(f"Opal Vanguard: Generating V8.5 Failure Analysis Matrix")
    
    # 1. Build & Load Model
    model = build_resnet_vanguard((1024, 2), NUM_CLASSES_MODEL)
    model.load_weights(WEIGHTS_PATH)
    print(f"Successfully loaded {WEIGHTS_PATH}")

    # 2. Setup Data Loader
    loader = RadioMLDataLoader(DATASET, num_classes=NUM_CLASSES_DATA)
    with h5py.File(DATASET, 'r') as f:
        n_samples = f['X'].shape[0]
        # Use a 10k sample subset for a snappy diagnostic run
        sample_size = min(10000, n_samples)
        indices = np.random.choice(n_samples, sample_size, replace=False)
        x_raw = f['X'][sorted(indices)]
        y_raw = f['Y'][sorted(indices)]

    x_norm = loader.normalize(x_raw)
    
    # 3. Predict & Analyze
    print(f"Analyzing {sample_size} hardware snapshots...")
    predictions = model.predict(x_norm, batch_size=128)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_raw, axis=1)

    # 4. Generate Confusion Matrix (First 24 Classes)
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES_DATA))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # Handle zero-count classes

    # 5. Plotting
    plt.figure(figsize=(18, 14))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=loader.modulations, 
                yticklabels=loader.modulations)
    plt.title(f'Vanguard V8.5 Hardware Confusion Matrix (Acc: {np.mean(y_true == y_pred)*100:.2f}%)')
    plt.ylabel('True Modulation')
    plt.xlabel('Predicted Modulation')
    plt.savefig(OUTPUT_PNG)
    print(f"Success: Confusion Matrix saved to {OUTPUT_PNG}")

    # 6. Report the Top 3 "Fatal Confusions"
    print("\n--- Top Diagnostic Failures ---")
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
