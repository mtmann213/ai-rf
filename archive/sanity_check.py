import os
import argparse
import tensorflow as tf
import numpy as np
from data_loader import RadioMLDataLoader

# Configuration
DEFAULT_DATASET = "2018_01A/GOLD_XYZ_OSC.0001_1024.hdf5"
MODEL_PATH = 'best_resnet_v7.keras'

def sanity_check(dataset_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    print(f"Opal Vanguard: Running Zero-Shot Check on {dataset_path}")
    print(f"Loading model: {MODEL_PATH}")
    
    # Load model with weights from the current 52% training run
    model = tf.keras.models.load_model(MODEL_PATH)
    
    loader = RadioMLDataLoader(dataset_path)
    
    # Check the first 512 samples
    print(f"\nTesting 512 samples...")
    gen = loader.get_generator(np.arange(512), batch_size=512)
    x, y = next(gen)
    
    preds = model.predict(x, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(y, axis=1)
    
    acc = np.mean(pred_labels == true_labels)
    print(f"Zero-Shot Accuracy: {acc:.4f}")
    
    print("\nSample Comparisons (First 10):")
    for i in range(10):
        match = "✓" if pred_labels[i] == true_labels[i] else "✗"
        print(f"[{match}] Pred: {loader.modulations[pred_labels[i]]:<10} | True: {loader.modulations[true_labels[i]]:<10}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default=DEFAULT_DATASET, help='Path to the .h5 file to test')
    args = parser.parse_args()
    
    sanity_check(args.file)
