import os
import tensorflow as tf
import numpy as np
import h5py
from src.data_loader import RadioMLDataLoader
from src.resnet_lstm_polar_v9 import build_resnet_lstm_polar_v9

# Configuration: THE BREAKOUT MISSION (V9.8.1 STABILIZED)
VDF_DATASET = "data/VDF_SPECTER_GOLDEN.h5"
BASE_DATASET = "2018_01A/GOLD_XYZ_OSC.0001_1024.hdf5"
NUTRIENT_DATASET = "data/VDF_SPECIALIST_NUTRIENTS.h5" 
MEGA_DATASET = "data/VDF_MEGA_SYNTHETIC_1M.h5"

MODEL_PATH = 'models/vanguard_v9_global_final.keras'
CHECKPOINT_PATH = 'models/v9_global_checkpoint.keras'
RESUME_PATH = 'models/vanguard_v9_specialist_sovereign.keras'

BATCH_SIZE = 64
LEARNING_RATE = 0.00005 
NUM_CLASSES = 57

TARGET_INDICES = [1, 2, 3, 10, 14, 18, 21]

def load_dataset_to_ram(file_path):
    print(f"Loading {file_path} into RAM...")
    with h5py.File(file_path, 'r') as f:
        x = f['X'][:]
        y = f['Y'][:]
    print(f"RAM Load Complete: {x.shape[0]} samples cached.")
    return x, y

def stabilized_breakout_generator(x_vdf, y_vdf, indices_vdf, mega_path, x_nuts, y_nuts, batch_size):
    """
    V9.8.1 Memory-Efficient Generator:
    - 50% Nutrients (RAM)
    - 30% Hardware (RAM)
    - 20% Mega-Synthetic (Disk Streaming)
    """
    nut_size = int(batch_size * 0.5)
    hw_size = int(batch_size * 0.3)
    mega_size = batch_size - nut_size - hw_size
    
    y_vdf_classes = np.argmax(y_vdf, axis=1)
    target_sample_mask = np.isin(y_vdf_classes, TARGET_INDICES)
    target_vdf_indices = np.intersect1d(indices_vdf, np.where(target_sample_mask)[0])
    indices_nuts = np.arange(len(x_nuts))
    
    # Open Mega-Dataset for streaming to save RAM
    with h5py.File(mega_path, 'r') as f_mega:
        x_m_ds = f_mega['X']
        y_m_ds = f_mega['Y']
        n_mega_limit = min(x_m_ds.shape[0], 200000) # Use first 200k as anchor
        
        while True:
            # 1. Nutrients (RAM)
            n_idx = np.random.choice(indices_nuts, nut_size, replace=True)
            x_n, y_n = x_nuts[n_idx], y_nuts[n_idx]
            
            # 2. Hardware (RAM)
            t_idx = np.random.choice(target_vdf_indices, hw_size, replace=True)
            x_h, y_h = x_vdf[t_idx], y_vdf[t_idx]
            
            # 3. Mega-Synthetic (DISK STREAM)
            # We sort indices for faster HDF5 chunk access
            m_idx = sorted(np.random.choice(n_mega_limit, mega_size, replace=False))
            x_m, y_m = x_m_ds[m_idx], y_m_ds[m_idx]
            
            # Combine & Polar
            x_iq_mixed = np.concatenate([x_n, x_h, x_m], axis=0)
            i_comp, q_comp = x_iq_mixed[:, :, 0], x_iq_mixed[:, :, 1]
            amplitude = np.sqrt(i_comp**2 + q_comp**2)
            amplitude = amplitude / (1.0 + np.abs(amplitude))
            phase = np.arctan2(q_comp, i_comp) / np.pi
            x_polar_mixed = np.stack([amplitude, phase], axis=-1)
            
            y_mixed = np.concatenate([y_n, y_h, y_m], axis=0)
            
            p = np.random.permutation(len(x_iq_mixed))
            yield (x_iq_mixed[p], x_polar_mixed[p]), y_mixed[p]

def main():
    print(f"Opal Vanguard: Launching V9.8.1 STABILIZED BREAKOUT Ignition")
    
    model = build_resnet_lstm_polar_v9(iq_shape=(1024, 2), polar_shape=(1024, 2), num_classes=NUM_CLASSES)
    
    if os.path.exists(RESUME_PATH):
        print(f"Resuming from Sovereign Foundation: {RESUME_PATH}")
        model.load_weights(RESUME_PATH)

    # Only load high-priority datasets to RAM
    x_nuts, y_nuts = load_dataset_to_ram(NUTRIENT_DATASET)
    x_vdf, y_vdf_raw = load_dataset_to_ram(VDF_DATASET)
    y_vdf = np.zeros((y_vdf_raw.shape[0], NUM_CLASSES), dtype=np.float32)
    y_vdf[:, :24] = y_vdf_raw[:, :24]
    
    loader_vdf = RadioMLDataLoader(VDF_DATASET, num_classes=24) 
    train_idx_vdf, val_idx_vdf = loader_vdf.get_train_val_indices(test_size=0.1)

    loader_base = RadioMLDataLoader(BASE_DATASET, num_classes=24)
    train_idx_base, val_idx_base = loader_base.get_train_val_indices(test_size=0.1)
    
    train_ds = tf.data.Dataset.from_generator(
        lambda: stabilized_breakout_generator(x_vdf, y_vdf, train_idx_vdf, MEGA_DATASET, x_nuts, y_nuts, BATCH_SIZE),
        output_signature=((tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
                           tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32)),
                          tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32))
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: stabilized_breakout_generator(x_vdf, y_vdf, val_idx_vdf, MEGA_DATASET, x_nuts, y_nuts, BATCH_SIZE),
        output_signature=((tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
                           tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32)),
                          tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32))
    ).prefetch(tf.data.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger('reports/v9_global_log.csv', append=True)
    ]

    print(f"Igniting Stabilized Breakout Marathon...")
    model.fit(train_ds, epochs=100, steps_per_epoch=5000,
              validation_data=val_ds, validation_steps=500,
              callbacks=callbacks)
    
    model.save(MODEL_PATH)
    print(f"Mission Success: Sovereign Stabilized saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
