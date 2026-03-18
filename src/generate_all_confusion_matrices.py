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

# Configuration
DATASET = "data/VDF_SPECTER_GOLDEN.h5"
MODELS_TO_TEST = [
    {"name": "V8.5_ResNet", "path": "models/vanguard_v8_specter_final_57pct.keras", "type": "resnet_v8"},
    {"name": "V9.1_Specialist", "path": "models/vanguard_v9_specialist_final.keras", "type": "iq_only"},
    {"name": "V9.2_Sovereign", "path": "models/vanguard_v9_sovereign_final.keras", "type": "multi_modal"},
    {"name": "V9.4_Transfused", "path": "models/v9_specialist_sovereign_checkpoint.keras", "type": "multi_modal"}
]
NUM_CLASSES_MODEL = 57
NUM_CLASSES_DATA = 24

def run_evaluation(model_info, x_raw, y_raw, loader):
    print(f"\n--- Evaluating Model: {model_info['name']} ---")
    
    # 1. Build & Load
    from src.resnet_opal_vanguard import build_resnet_vanguard
    if model_info['type'] == "resnet_v8":
        model = build_resnet_vanguard(num_classes=NUM_CLASSES_MODEL)
    elif model_info['type'] == "iq_only":
        model = build_resnet_lstm_v9(num_classes=NUM_CLASSES_MODEL)
    else:
        model = build_resnet_lstm_polar_v9(num_classes=NUM_CLASSES_MODEL)
    
    if os.path.exists(model_info['path']):
        model.load_weights(model_info['path'])
        print(f" Loaded weights from {model_info['path']}")
    else:
        print(f" ERROR: Weights not found at {model_info['path']}")
        return None

    # 2. Normalize & Prep Inputs
    x_norm = loader.normalize(x_raw)
    
    if model_info['type'] == "multi_modal":
        i_comp, q_comp = x_norm[:, :, 0], x_norm[:, :, 1]
        amplitude = np.sqrt(i_comp**2 + q_comp**2)
        phase = np.arctan2(q_comp, i_comp)
        amplitude = amplitude / (1.0 + np.abs(amplitude))
        phase = phase / np.pi
        x_polar = np.stack([amplitude, phase], axis=-1)
        inputs = [x_norm, x_polar]
    else:
        inputs = x_norm

    # 3. Predict & Filter to 24 Classes
    # We only care about how well it identifies the signals that are ACTUALLY there.
    predictions = model.predict(inputs, batch_size=128)
    
    # Slice the predictions to only the first 24 classes before taking the argmax
    # This forces the model to choose from the "Known Hardware" vocabulary.
    predictions_filtered = predictions[:, :NUM_CLASSES_DATA]
    y_pred = np.argmax(predictions_filtered, axis=1)
    y_true = np.argmax(y_raw, axis=1)
    
    acc = np.mean(y_true == y_pred) * 100
    print(f" Filtered Hardware Accuracy (24-class): {acc:.2f}%")
    
    # 4. Generate Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES_DATA))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    return cm_norm, acc

def main():
    print(f"Opal Vanguard: Master Performance Comparison")
    
    # Setup Data
    loader = RadioMLDataLoader(DATASET, num_classes=NUM_CLASSES_DATA)
    with h5py.File(DATASET, 'r') as f:
        n_samples = f['X'].shape[0]
        sample_size = 10000
        indices = np.random.choice(n_samples, sample_size, replace=False)
        x_raw = f['X'][sorted(indices)]
        y_raw = f['Y'][sorted(indices)]

    results = []
    for model_info in MODELS_TO_TEST:
        res = run_evaluation(model_info, x_raw, y_raw, loader)
        if res:
            results.append((model_info['name'], res[0], res[1]))

    # Plot Comparison
    fig, axes = plt.subplots(1, len(results), figsize=(12 * len(results), 10))
    if len(results) == 1: axes = [axes]
    
    for i, (name, cm, acc) in enumerate(results):
        sns.heatmap(cm, annot=False, cmap='viridis', ax=axes[i], cbar=False)
        axes[i].set_title(f"{name} ({acc:.1f}%)")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
        
    plt.tight_layout()
    plt.savefig("reports/master_comparison_matrix.png")
    print("\nSuccess: Master Comparison Matrix saved to reports/master_comparison_matrix.png")

if __name__ == "__main__":
    main()
