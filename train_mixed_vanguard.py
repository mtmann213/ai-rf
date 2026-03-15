import os
import tensorflow as tf
import numpy as np
import h5py
from data_loader import RadioMLDataLoader

# Configuration
VDF_DATASET = "VDF_GOLDEN_24CLASS.h5"
BASE_DATASET = "2018_01A/GOLD_XYZ_OSC.0001_1024.hdf5"
BASE_MODEL_PATH = 'best_resnet_v7.keras'
NEW_MODEL_PATH = 'vanguard_mixed_24class.keras'
CHECKPOINT_PATH = 'mixed_vdf_checkpoint.keras'
BATCH_SIZE = 64
LEARNING_RATE = 0.00005 

def load_vdf_to_ram(file_path):
    """Loads the entire hardware dataset into system RAM to prevent HDF5 race conditions."""
    print(f"Loading {file_path} into RAM...")
    with h5py.File(file_path, 'r') as f:
        x = f['X'][:]
        y = f['Y'][:]
    print(f"RAM Load Complete: {x.shape[0]} samples cached.")
    return x, y

def mixed_generator(x_vdf, y_vdf, indices_vdf, loader_base, indices_base, batch_size):
    """Yields batches: 50% from RAM (Hardware) and 50% from Disk (Simulation)."""
    gen_base = loader_base.get_generator(indices_base, batch_size // 2)
    
    while True:
        # 1. Get Simulation data from disk
        x_base, y_base = next(gen_base)
        
        # 2. Get Hardware data from RAM
        vdf_batch_idx = np.random.choice(indices_vdf, batch_size // 2, replace=False)
        x_vdf_batch = x_vdf[vdf_batch_idx]
        y_vdf_batch = y_vdf[vdf_batch_idx]
        
        # 3. Combine and Shuffle
        x_mixed = np.concatenate([x_vdf_batch, x_base], axis=0)
        y_mixed = np.concatenate([y_vdf_batch, y_base], axis=0)
        
        p = np.random.permutation(len(x_mixed))
        yield x_mixed[p], y_mixed[p]

def main():
    print(f"Opal Vanguard: Launching V7.7.2 Bulletproof Mixed Trainer")
    
    # 1. Load the Brain
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        model = tf.keras.models.load_model(CHECKPOINT_PATH)
    else:
        print(f"Loading foundational weights from {BASE_MODEL_PATH}...")
        model = tf.keras.models.load_model(BASE_MODEL_PATH)
    
    for layer in model.layers:
        layer.trainable = True

    # 2. Load Hardware Data into RAM
    x_vdf, y_vdf = load_vdf_to_ram(VDF_DATASET)
    loader_vdf = RadioMLDataLoader(VDF_DATASET) # For indices
    _, val_idx_vdf = loader_vdf.get_train_val_indices(test_size=0.1)
    train_idx_vdf = np.arange(len(x_vdf))
    # Remove validation indices from training set
    train_idx_vdf = np.delete(train_idx_vdf, val_idx_vdf)

    # 3. Setup Base Loader (Disk Stream)
    loader_base = RadioMLDataLoader(BASE_DATASET)
    train_idx_base, val_idx_base = loader_base.get_train_val_indices(test_size=0.1)
    
    # 4. Create Datasets
    train_ds = tf.data.Dataset.from_generator(
        lambda: mixed_generator(x_vdf, y_vdf, train_idx_vdf, loader_base, train_idx_base, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 24), dtype=tf.float32)
        )
    )

    val_ds = tf.data.Dataset.from_generator(
        lambda: mixed_generator(x_vdf, y_vdf, val_idx_vdf, loader_base, val_idx_base, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 24), dtype=tf.float32)
        )
    )

    # 5. Compile & fit
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    ]

    steps = len(train_idx_vdf) // (BATCH_SIZE // 2)
    val_steps = len(val_idx_vdf) // (BATCH_SIZE // 2)
    
    model.fit(train_ds, epochs=50, steps_per_epoch=steps,
              validation_data=val_ds, validation_steps=val_steps,
              callbacks=callbacks)
    
    model.save(NEW_MODEL_PATH)

if __name__ == "__main__":
    main()
