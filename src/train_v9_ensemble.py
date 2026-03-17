import os
import tensorflow as tf
import numpy as np
import h5py
from src.data_loader import RadioMLDataLoader
from src.resnet_lstm_v9 import build_resnet_lstm_v9

# Configuration
VDF_DATASET = "data/VDF_SPECTER_GOLDEN.h5"
BASE_DATASET = "2018_01A/GOLD_XYZ_OSC.0001_1024.hdf5"
MODEL_PATH = 'models/vanguard_v9_ensemble_acclimated.keras'
CHECKPOINT_PATH = 'models/v9_ensemble_checkpoint.keras'
BATCH_SIZE = 64
LEARNING_RATE = 0.00002 
NUM_CLASSES = 57 # Expanded V8 Vocabulary

def load_vdf_to_ram(file_path):
    print(f"Loading {file_path} into RAM...")
    with h5py.File(file_path, 'r') as f:
        x = f['X'][:]
        y = f['Y'][:]
    print(f"RAM Load Complete: {x.shape[0]} samples cached.")
    return x, y

def mixed_generator(x_vdf, y_vdf, indices_vdf, loader_base, indices_base, batch_size):
    hw_per_batch = batch_size // 2
    sim_per_batch = batch_size - hw_per_batch
    gen_base = loader_base.get_generator(indices_base, sim_per_batch)
    
    while True:
        try:
            x_base, y_base = next(gen_base)
        except StopIteration:
            gen_base = loader_base.get_generator(indices_base, sim_per_batch)
            x_base, y_base = next(gen_base)
            
        vdf_batch_idx = np.random.choice(indices_vdf, hw_per_batch, replace=False)
        x_vdf_batch = x_vdf[vdf_batch_idx]
        y_vdf_batch = y_vdf[vdf_batch_idx]
        
        x_mixed = np.concatenate([x_vdf_batch, x_base], axis=0)
        y_base_padded = np.zeros((y_base.shape[0], NUM_CLASSES), dtype=np.float32)
        y_base_padded[:, :y_base.shape[1]] = y_base
        y_mixed = np.concatenate([y_vdf_batch, y_base_padded], axis=0)
        
        p = np.random.permutation(len(x_mixed))
        yield x_mixed[p], y_mixed[p]

def main():
    print(f"Opal Vanguard: Launching V9.0 CLDNN Ignition (Eyes + Ears)")
    
    # 1. Build V9 Architecture
    model = build_resnet_lstm_v9(input_shape=(1024, 2), num_classes=NUM_CLASSES)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from V9 checkpoint: {CHECKPOINT_PATH}")
        model.load_weights(CHECKPOINT_PATH)
    else:
        print("Starting V9 from scratch.")

    # 2. Load Hardware Data into RAM
    x_vdf, y_vdf_raw = load_vdf_to_ram(VDF_DATASET)
    print(f"Adapting {VDF_DATASET} labels (24 -> {NUM_CLASSES} classes)...")
    y_vdf = np.zeros((y_vdf_raw.shape[0], NUM_CLASSES), dtype=np.float32)
    y_vdf[:, :24] = y_vdf_raw[:, :24]
    
    loader_vdf = RadioMLDataLoader(VDF_DATASET, num_classes=24) 
    _, val_idx_vdf = loader_vdf.get_train_val_indices(test_size=0.1)
    train_idx_vdf = np.arange(len(x_vdf))
    train_idx_vdf = np.delete(train_idx_vdf, val_idx_vdf)

    # 3. Setup Base Loader
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
    # Using from_logits=True as per our Nuclear-Grade stability fix (V6.5)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    # 6. Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger('reports/v9_ensemble_log.csv', append=True)
    ]

    hw_per_epoch = len(train_idx_vdf)
    steps = hw_per_epoch // (BATCH_SIZE // 2)
    val_steps = len(val_idx_vdf) // (BATCH_SIZE // 2)
    
    print(f"Starting V9.0 Ignition: {hw_per_epoch} hardware samples mixed with simulation.")
    model.fit(train_ds, epochs=100, steps_per_epoch=steps,
              validation_data=val_ds, validation_steps=val_steps,
              callbacks=callbacks)
    
    model.save(MODEL_PATH)
    print(f"Mission Success: V9.0 Ensemble saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
