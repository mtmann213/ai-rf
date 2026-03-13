import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import sionna
from sionna.utils import BinarySource, Mapper, Constellation
from sionna.channel import AWGN

# Import configuration from training script
from train_opal_vanguard import MODULATIONS, NUM_CLASSES, INPUT_LENGTH, INPUT_SHAPE, get_sionna_constellation

def generate_evaluation_data(mod_idx, batch_size, ebno_db):
    """Generates synthetic Sionna samples for a specific modulation and SNR."""
    mod_type = MODULATIONS[mod_idx]
    const = get_sionna_constellation(mod_type)
    mapper = Mapper(constellation=const)
    source = BinarySource()
    channel = AWGN()
    
    b = source([batch_size, INPUT_LENGTH * const.num_bits_per_symbol])
    x = mapper(b)
    y = channel([x, ebno_db])
    
    y_iq = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1)
    return y_iq

def run_benchmarking(model_path='opal_vanguard_base.h5'):
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please run train_opal_vanguard.py first.")
        return

    print(f"Opal Vanguard: Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    snr_range = np.arange(-20, 31, 5)
    overall_accuracies = []
    
    # Store predictions for the final Confusion Matrix (using a high SNR for clarity)
    eval_snr = 20.0
    all_preds = []
    all_labels = []

    print("\n--- Starting SNR Benchmarking ---")
    for snr in snr_range:
        correct = 0
        total = 0
        
        for idx, mod in enumerate(MODULATIONS):
            x_test = generate_evaluation_data(idx, 100, float(snr))
            y_test = np.full((100,), idx)
            
            preds = model.predict(x_test, verbose=0)
            pred_labels = np.argmax(preds, axis=1)
            
            correct += np.sum(pred_labels == y_test)
            total += 100
            
            # Capture data for Confusion Matrix at the target SNR
            if snr == eval_snr:
                all_preds.extend(pred_labels)
                all_labels.extend(y_test)
                
        acc = correct / total
        overall_accuracies.append(acc)
        print(f"SNR {snr:3}dB | Accuracy: {acc:.4f}")

    # Plot 1: Accuracy vs SNR (Waterfall Curve)
    plt.figure(figsize=(10, 6))
    plt.plot(snr_range, overall_accuracies, marker='o', linestyle='-', linewidth=2)
    plt.title('Opal Vanguard: Classification Accuracy vs SNR (AWGN)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('accuracy_vs_snr.png')
    print("\nSaved accuracy_vs_snr.png")

    # Plot 2: Confusion Matrix at +20dB
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=MODULATIONS, yticklabels=MODULATIONS, cmap='Blues')
    plt.title(f'Opal Vanguard: Confusion Matrix at {eval_snr}dB SNR')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix_20dB.png')
    print("Saved confusion_matrix_20dB.png")

if __name__ == "__main__":
    run_benchmarking()
