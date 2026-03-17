import os
import tensorflow as tf
import numpy as np
import h5py
from data_loader import RadioMLDataLoader
from resnet_opal_vanguard import build_resnet_vanguard

# Configuration
DATASET = "VDF_SPECTER_GOLDEN.h5"
# We check for the latest checkpoint first, then the starting weights
WEIGHTS_TO_TEST = "vanguard_v8_deep_iq.h5"

NUM_CLASSES_MODEL = 57
NUM_CLASSES_DATA = 24 # Assuming the hardware file uses our 24-class USRP list

def main():
    print(f"Opal Vanguard: Hardware Validation Pulse")
    print(f"Testing weights: {WEIGHTS_TO_TEST}")
    print(f"Against Dataset: {DATASET}")

    # 1. Build V8 Architecture
    model = build_resnet_vanguard(input_shape=(1024, 2), num_classes=NUM_CLASSES_MODEL)
    
    # 2. Load Weights
    if os.path.exists(WEIGHTS_TO_TEST):
        model.load_weights(WEIGHTS_TO_TEST)
        print(f"Successfully loaded {WEIGHTS_TO_TEST}")
    else:
        print(f"ERROR: Weights file {WEIGHTS_TO_TEST} not found!")
        return

    # 3. Setup Dataset
    # We use the DataLoader to ensure consistent normalization
    loader = RadioMLDataLoader(DATASET, num_classes=NUM_CLASSES_DATA)
    
    # Get a batch of indices for a representative sample (e.g., 5000 samples)
    with h5py.File(DATASET, 'r') as f:
        n_samples = f['X'].shape[0]
        print(f"Total samples in dataset: {n_samples}")
        
    sample_size = min(10000, n_samples)
    indices = np.random.choice(n_samples, sample_size, replace=False)
    
    # 4. Run Inference
    print(f"Extracting {sample_size} hardware snapshots and normalizing...")
    
    # Note: We manually pull and normalize to handle the 24->57 label mapping for validation
    with h5py.File(DATASET, 'r') as f:
        x_raw = f['X'][sorted(indices)]
        y_raw = f['Y'][sorted(indices)]
        
    x_norm = loader.normalize(x_raw)
    
    print("Evaluating...")
    predictions = model.predict(x_norm, batch_size=128)
    pred_classes = np.argmax(predictions, axis=1)
    
    # For validation, we only care about the first 24 classes where the hardware labels exist
    true_classes = np.argmax(y_raw, axis=1)
    
    # Calculate accuracy
    correct = (pred_classes == true_classes).sum()
    accuracy = (correct / sample_size) * 100
    
    print(f"\n--- MISSION RESULTS ---")
    print(f"Hardware Generalization: {accuracy:.2f}%")
    print(f"Correct: {correct} / {sample_size}")
    
    # Detailed breakdown for top 5 classes
    print("\nTop Predictable Classes in Hardware:")
    for i in range(min(5, NUM_CLASSES_DATA)):
        class_mask = (true_classes == i)
        if class_mask.sum() > 0:
            class_acc = (pred_classes[class_mask] == i).sum() / class_mask.sum()
            label = loader.modulations[i] if i < len(loader.modulations) else f"Class_{i}"
            print(f" - {label}: {class_acc*100:.1f}%")

if __name__ == "__main__":
    main()
