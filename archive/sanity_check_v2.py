import os
import argparse
import tensorflow as tf
import numpy as np
from data_loader import RadioMLDataLoader
from resnet_opal_vanguard import build_resnet_vanguard

# Configuration
INPUT_SHAPE = (1024, 2)
NUM_CLASSES = 24

def sanity_check(dataset_path, model_path):
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        return
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset {dataset_path} not found.")
        return

    print(f"Opal Vanguard: Running Validation Check on {dataset_path}")
    
    # 1. Build Architecture
    print("Building ResNet architecture...")
    model = build_resnet_vanguard(INPUT_SHAPE, NUM_CLASSES)
    
    # 2. Load Weights
    print(f"Loading weights from {model_path}...")
    try:
        # Try loading as weights first (our new V7.7.4 standard)
        model.load_weights(model_path)
    except:
        # Fallback for full model files
        print("Switching to full model load...")
        model = tf.keras.models.load_model(model_path)
    
    loader = RadioMLDataLoader(dataset_path)
    
    # Check 1024 samples from the end (Validation set style)
    print(f"\nTesting 1024 samples...")
    gen = loader.get_generator(np.arange(1024), batch_size=1024)
    x, y = next(gen)
    
    preds = model.predict(x, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = np.argmax(y, axis=1)
    
    acc = np.mean(pred_labels == true_labels)
    print(f"Hardware-Tuned Accuracy: {acc:.4f}")
    
    print("\nSample Comparisons (First 10):")
    for i in range(10):
        match = "✓" if pred_labels[i] == true_labels[i] else "✗"
        print(f"[{match}] Pred: {loader.modulations[pred_labels[i]]:<10} | True: {loader.modulations[true_labels[i]]:<10}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True, help='Path to the .h5 file to test')
    parser.add_argument('--model', type=str, default='vanguard_hardware_alpha.keras', help='Path to the model to test')
    args = parser.parse_args()
    
    sanity_check(args.file, args.model)
