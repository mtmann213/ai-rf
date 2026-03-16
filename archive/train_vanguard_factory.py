import os
import tensorflow as tf
import numpy as np
from data_loader import RadioMLDataLoader
from resnet_opal_vanguard import build_resnet_vanguard

# Configuration
VDF_DATASET = "phase2_aligned_final.h5"
BASE_MODEL_PATH = 'best_resnet_v7.keras'
NEW_MODEL_PATH = 'vanguard_hardware_alpha.keras'
BATCH_SIZE = 64
LEARNING_RATE = 0.00001 # Stage 2: Very low LR for fine-tuning

def main():
    if not os.path.exists(VDF_DATASET):
        print(f"Error: {VDF_DATASET} not found.")
        return

    print(f"Opal Vanguard: Starting Stage 2 Acclimation (Hardware Fine-Tuning)")
    
    # 1. Load the "Simulated Brain"
    print(f"Loading weights from {BASE_MODEL_PATH}...")
    base_model = tf.keras.models.load_model(BASE_MODEL_PATH)
    
    # 2. FEATURE LOCKING
    # Freeze the convolutional layers (the physics)
    # Only allow the dense layers at the end to learn hardware specifics
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            layer.trainable = False
            
    print("Feature layers locked. Classification head remains trainable.")

    # 3. Load VDF Data
    loader = RadioMLDataLoader(VDF_DATASET)
    train_indices, val_indices = loader.get_train_val_indices(test_size=0.1)
    
    train_ds = tf.data.Dataset.from_generator(
        lambda: loader.get_generator(train_indices, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 24), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: loader.get_generator(val_indices, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 24), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    # 4. Compile & Ignite
    # We use a very gentle optimizer to avoid smearing the learned filters
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
    
    base_model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    print("\n--- Starting Hardware Fine-Tuning (10 Epochs) ---")
    steps = len(train_indices) // BATCH_SIZE
    val_steps = len(val_indices) // BATCH_SIZE
    
    base_model.fit(train_ds, 
                  epochs=10, 
                  steps_per_epoch=steps,
                  validation_data=val_ds,
                  validation_steps=val_steps)
    
    base_model.save(NEW_MODEL_PATH)
    print(f"\nMission Success: Hardware-tuned model saved to {NEW_MODEL_PATH}")

if __name__ == "__main__":
    main()
