import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data_loader import RadioMLDataLoader

# Configuration
DATASET_PATH = "2018_01A/GOLD_XYZ_OSC.0001_1024.hdf5"
MODEL_PATH = 'best_resnet_v7.keras'
BATCH_SIZE = 128

def run_benchmarking():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found.")
        return

    print(f"Opal Vanguard: Loading V7.1 model and dataset...")
    model = tf.keras.models.load_model(MODEL_PATH)
    loader = RadioMLDataLoader(DATASET_PATH)
    _, val_indices = loader.get_train_val_indices()
    
    # We'll evaluate on a large slice of the validation set (50,000 samples)
    eval_indices = val_indices[:50000]
    
    val_ds = tf.data.Dataset.from_generator(
        lambda: loader.get_generator(eval_indices, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 24), dtype=tf.float32)
        )
    )

    all_preds = []
    all_labels = []

    print(f"\n--- Starting Real-World Validation (50,000 samples) ---")
    steps = len(eval_indices) // BATCH_SIZE
    
    for x_batch, y_batch in val_ds.take(steps):
        preds = model.predict(x_batch, verbose=0)
        all_preds.extend(np.argmax(preds, axis=1))
        all_labels.extend(np.argmax(y_batch, axis=1))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    overall_acc = np.mean(all_preds == all_labels)
    print(f"\nOverall Validation Accuracy: {overall_acc:.4f}")

    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=loader.modulations, 
                yticklabels=loader.modulations, cmap='Greens')
    plt.title(f'Opal Vanguard V7.1: RadioML 2018.01A Confusion Matrix (Acc: {overall_acc:.4f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix_v7_real.png')
    print("Saved confusion_matrix_v7_real.png")

if __name__ == "__main__":
    run_benchmarking()
