import os
import tensorflow as tf
import numpy as np
import h5py
from data_loader import RadioMLDataLoader

# Configuration
VDF_DATASET = "VDF_SPECTER_GOLDEN.h5"
BASE_DATASET = "2018_01A/GOLD_XYZ_OSC.0001_1024.hdf5"
BASE_MODEL_PATH = 'vanguard_v8_deep_iq.h5'
NEW_MODEL_PATH = 'vanguard_v8_specter_acclimated.keras'
CHECKPOINT_PATH = 'mixed_specter_v8_checkpoint.keras'
BATCH_SIZE = 64
LEARNING_RATE = 0.00002 
NUM_CLASSES = 57 # Expanded V8 Vocabulary

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
    hw_per_batch = batch_size // 2
    sim_per_batch = batch_size - hw_per_batch
    
    gen_base = loader_base.get_generator(indices_base, sim_per_batch)
    
    while True:
        # 1. Get Simulation data from disk
        try:
            x_base, y_base = next(gen_base)
        except StopIteration:
            # Re-initialize generator if it runs out
            gen_base = loader_base.get_generator(indices_base, sim_per_batch)
            x_base, y_base = next(gen_base)
            
        # 2. Get Hardware data from RAM
        vdf_batch_idx = np.random.choice(indices_vdf, hw_per_batch, replace=False)
        x_vdf_batch = x_vdf[vdf_batch_idx]
        y_vdf_batch = y_vdf[vdf_batch_idx]
        
        # 3. Combine and Shuffle
        x_mixed = np.concatenate([x_vdf_batch, x_base], axis=0)
        
        # Ensure simulation labels are also 57-class
        y_base_padded = np.zeros((y_base.shape[0], NUM_CLASSES), dtype=np.float32)
        y_base_padded[:, :y_base.shape[1]] = y_base
        
        y_mixed = np.concatenate([y_vdf_batch, y_base_padded], axis=0)
        
        p = np.random.permutation(len(x_mixed))
        yield x_mixed[p], y_mixed[p]

def main():
    print(f"Opal Vanguard: Launching V8 Specter Hardware Acclimation")
    
    # 1. Build Architecture
    from resnet_opal_vanguard import build_resnet_vanguard
    model = build_resnet_vanguard((1024, 2), NUM_CLASSES)

    # 2. Load the Weights
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        model.load_weights(CHECKPOINT_PATH)
    elif os.path.exists(BASE_MODEL_PATH):
        print(f"Loading high-fidelity weights from {BASE_MODEL_PATH}...")
        model.load_weights(BASE_MODEL_PATH)
    else:
        print(f"WARNING: No base weights found at {BASE_MODEL_PATH}. Starting from scratch.")
    
    for layer in model.layers:
        layer.trainable = True

    # 2. Load Hardware Data into RAM
    x_vdf, y_vdf_raw = load_vdf_to_ram(VDF_DATASET)
    
    # RE-MAP: Pad the 24-class VDF labels to 57 classes for V8 compatibility
    print(f"Adapting {VDF_DATASET} labels (24 -> {NUM_CLASSES} classes)...")
    y_vdf = np.zeros((y_vdf_raw.shape[0], NUM_CLASSES), dtype=np.float32)
    # Ensure we don't overflow if y_vdf_raw has more than 24 classes (unlikely but safe)
    num_hw_classes = min(y_vdf_raw.shape[1], NUM_CLASSES)
    y_vdf[:, :num_hw_classes] = y_vdf_raw[:, :num_hw_classes]
    
    loader_vdf = RadioMLDataLoader(VDF_DATASET, num_classes=num_hw_classes) 
    _, val_idx_vdf = loader_vdf.get_train_val_indices(test_size=0.1)
    train_idx_vdf = np.arange(len(x_vdf))
    train_idx_vdf = np.delete(train_idx_vdf, val_idx_vdf)

    # 3. Setup Base Loader (2018_01A simulation)
    loader_base = RadioMLDataLoader(BASE_DATASET, num_classes=24)
    train_idx_base, val_idx_base = loader_base.get_train_val_indices(test_size=0.1)
    
    # 4. Create Datasets
    train_ds = tf.data.Dataset.from_generator(
        lambda: mixed_generator(x_vdf, y_vdf, train_idx_vdf, loader_base, train_idx_base, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: mixed_generator(x_vdf, y_vdf, val_idx_vdf, loader_base, val_idx_base, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    # 5. Compile & Ignite
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # 6. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger('specter_acclimation_log.csv', append=True)
    ]

    hw_per_epoch = len(train_idx_vdf)
    steps = hw_per_epoch // (BATCH_SIZE // 2)
    val_steps = len(val_idx_vdf) // (BATCH_SIZE // 2)
    
    print(f"Starting Acclimation: {hw_per_epoch} hardware samples mixed with simulation.")
    model.fit(train_ds, epochs=100, steps_per_epoch=steps,
              validation_data=val_ds, validation_steps=val_steps,
              callbacks=callbacks)
    
    model.save(NEW_MODEL_PATH)
    print(f"Mission Success: Acclimated model saved to {NEW_MODEL_PATH}")

if __name__ == "__main__":
    main()
