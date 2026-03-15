import os
import tensorflow as tf
import numpy as np
from data_loader import RadioMLDataLoader

# Configuration
VDF_DATASET = "VDF_GOLDEN_24CLASS.h5"
BASE_DATASET = "2018_01A/GOLD_XYZ_OSC.0001_1024.hdf5"
BASE_MODEL_PATH = 'best_resnet_v7.keras'
NEW_MODEL_PATH = 'vanguard_mixed_24class.keras'
BATCH_SIZE = 64
LEARNING_RATE = 0.00001 # Gentle learning rate for acclimation

def mixed_generator(loader_vdf, indices_vdf, loader_base, indices_base, batch_size):
    """Yields batches that are 50% Hardware Data and 50% Simulation Data."""
    gen_vdf = loader_vdf.get_generator(indices_vdf, batch_size // 2)
    gen_base = loader_base.get_generator(indices_base, batch_size // 2)
    
    while True:
        x_vdf, y_vdf = next(gen_vdf)
        x_base, y_base = next(gen_base)
        
        # Combine into a single batch
        x_mixed = np.concatenate([x_vdf, x_base], axis=0)
        y_mixed = np.concatenate([y_vdf, y_base], axis=0)
        
        # Shuffle the mixed batch
        p = np.random.permutation(len(x_mixed))
        yield x_mixed[p], y_mixed[p]

def main():
    print(f"Opal Vanguard: Launching Stage 3 Mixed Acclimation (8 Classes)")
    
    # 1. Load the "Foundational Brain" (Unfrozen)
    print(f"Loading weights from {BASE_MODEL_PATH}...")
    model = tf.keras.models.load_model(BASE_MODEL_PATH)
    
    # Ensure all layers are trainable for the "Full Thaw"
    for layer in model.layers:
        layer.trainable = True
    print("Full-Stack Thaw: All layers unlocked for fine-tuning.")

    # 2. Setup Loaders
    loader_vdf = RadioMLDataLoader(VDF_DATASET)
    loader_base = RadioMLDataLoader(BASE_DATASET)
    
    train_idx_vdf, val_idx_vdf = loader_vdf.get_train_val_indices(test_size=0.1)
    train_idx_base, val_idx_base = loader_base.get_train_val_indices(test_size=0.1)
    
    # 3. Create Dual-Engine Dataset
    train_ds = tf.data.Dataset.from_generator(
        lambda: mixed_generator(loader_vdf, train_idx_vdf, loader_base, train_idx_base, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 24), dtype=tf.float32)
        )
    )

    val_ds = tf.data.Dataset.from_generator(
        lambda: mixed_generator(loader_vdf, val_idx_vdf, loader_base, val_idx_base, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 24), dtype=tf.float32)
        )
    )

    # 4. Compile & Ignite
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
    
    # Check for existing checkpoint to resume
    checkpoint_path = 'mixed_vdf_checkpoint.keras'
    if os.path.exists(checkpoint_path):
        print(f"Resuming from existing checkpoint: {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path)
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # Setup Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    ]

    print("\n--- Starting Mixed Anchor Training (50 Epochs) ---")
    # We set steps to ensure we see all the new hardware data every epoch
    steps = len(train_idx_vdf) // (BATCH_SIZE // 2)
    val_steps = len(val_idx_vdf) // (BATCH_SIZE // 2)
    
    model.fit(train_ds, 
              epochs=50, 
              steps_per_epoch=steps,
              validation_data=val_ds,
              validation_steps=val_steps,
              callbacks=callbacks)
    
    model.save(NEW_MODEL_PATH)
    print(f"\nMission Success: Mixed Hardware/Sim model saved to {NEW_MODEL_PATH}")

if __name__ == "__main__":
    main()
