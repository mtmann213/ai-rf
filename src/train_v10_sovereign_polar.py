import os
import tensorflow as tf
import numpy as np
import h5py
from src.data_loader import RadioMLDataLoader
from src.transformer_polar_v10 import build_sovereign_transformer_polar_v10

# Configuration: THE UNIVERSAL EYE (V10.1)
VDF_DATASET = "data/VDF_SPECTER_GOLDEN.h5"
NUTRIENT_DATASET = "data/VDF_SPECIALIST_NUTRIENTS.h5" 
MEGA_DATASET = "data/VDF_MEGA_SYNTHETIC_1M.h5"

MODEL_PATH = 'models/vanguard_v10_sovereign_polar_final.keras'
CHECKPOINT_PATH = 'models/v10_sovereign_polar_checkpoint.keras'

BATCH_SIZE = 64
LEARNING_RATE = 0.00002 
NUM_CLASSES = 57

TARGET_INDICES = [1, 2, 3, 10, 14, 18, 21]

def load_dataset_to_ram(file_path, limit=None):
    print(f"Loading {file_path} into RAM...")
    with h5py.File(file_path, 'r') as f:
        if limit:
            x = f['X'][:limit]
            y = f['Y'][:limit]
        else:
            x = f['X'][:]
            y = f['Y'][:]
    print(f"RAM Load Complete: {x.shape[0]} samples cached.")
    return x, y

def universal_eye_generator(x_vdf, y_vdf, indices_vdf, mega_path, x_nuts, y_nuts, batch_size):
    """
    V10.1 Quad-Source Attention Generator:
    - 40% Specialist Nutrients (Laptop)
    - 30% Hardware (Realism)
    - 30% Mega-Synthetic (Anchor - Disk Streamed)
    """
    nut_size = int(batch_size * 0.4)
    hw_size = int(batch_size * 0.3)
    mega_size = batch_size - nut_size - hw_size
    
    y_vdf_classes = np.argmax(y_vdf, axis=1)
    target_sample_mask = np.isin(y_vdf_classes, TARGET_INDICES)
    target_vdf_indices = np.intersect1d(indices_vdf, np.where(target_sample_mask)[0])
    indices_nuts = np.arange(len(x_nuts))
    
    with h5py.File(mega_path, 'r') as f_mega:
        x_m_ds = f_mega['X']
        y_m_ds = f_mega['Y']
        n_mega_limit = min(x_m_ds.shape[0], 200000)
        
        while True:
            # 1. Nutrients (RAM)
            n_idx = np.random.choice(indices_nuts, nut_size, replace=True)
            x_n, y_n = x_nuts[n_idx], y_nuts[n_idx]
            
            # 2. Hardware (RAM)
            t_idx = np.random.choice(target_vdf_indices, hw_size, replace=True)
            x_h, y_h = x_vdf[t_idx], y_vdf[t_idx]
            
            # 3. Mega-Synthetic (DISK STREAM)
            m_idx = sorted(np.random.choice(n_mega_limit, mega_size, replace=False))
            x_m, y_m = x_m_ds[m_idx], y_m_ds[m_idx]
            
            # Combine IQ
            x_iq_mixed = np.concatenate([x_n, x_h, x_m], axis=0)
            
            # Generate Polar
            i_comp, q_comp = x_iq_mixed[:, :, 0], x_iq_mixed[:, :, 1]
            amplitude = np.sqrt(i_comp**2 + q_comp**2)
            amplitude = amplitude / (1.0 + np.abs(amplitude))
            phase = np.arctan2(q_comp, i_comp) / np.pi
            x_polar_mixed = np.stack([amplitude, phase], axis=-1)
            
            y_mixed = np.concatenate([y_n, y_h, y_m], axis=0)
            
            p = np.random.permutation(len(x_iq_mixed))
            yield (x_iq_mixed[p], x_polar_mixed[p]), y_mixed[p]

def main():
    print(f"Opal Vanguard: Launching V10.1 SOVEREIGN POLAR TRANSFORMER Ignition")
    
    model = build_sovereign_transformer_polar_v10(iq_shape=(1024, 2), polar_shape=(1024, 2), num_classes=NUM_CLASSES)
    
    # Load Hardware and Nutrients
    x_nuts, y_nuts = load_dataset_to_ram(NUTRIENT_DATASET)
    x_vdf, y_vdf_raw = load_dataset_to_ram(VDF_DATASET)
    y_vdf = np.zeros((y_vdf_raw.shape[0], NUM_CLASSES), dtype=np.float32)
    y_vdf[:, :24] = y_vdf_raw[:, :24]
    
    loader_vdf = RadioMLDataLoader(VDF_DATASET, num_classes=24) 
    train_idx_vdf, val_idx_vdf = loader_vdf.get_train_val_indices(test_size=0.1)

    train_ds = tf.data.Dataset.from_generator(
        lambda: universal_eye_generator(x_vdf, y_vdf, train_idx_vdf, MEGA_DATASET, x_nuts, y_nuts, BATCH_SIZE),
        output_signature=((tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
                           tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32)),
                          tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32))
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: universal_eye_generator(x_vdf, y_vdf, val_idx_vdf, MEGA_DATASET, x_nuts, y_nuts, BATCH_SIZE),
        output_signature=((tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
                           tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32)),
                          tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32))
    ).prefetch(tf.data.AUTOTUNE)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger('reports/v10_sovereign_log.csv', append=True)
    ]

    print(f"Igniting Multi-Modal Attention Marathon...")
    model.fit(train_ds, epochs=100, steps_per_epoch=5000,
              validation_data=val_ds, validation_steps=500,
              callbacks=callbacks)
    
    model.save(MODEL_PATH)
    print(f"Mission Success: Sovereign Polar Transformer saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
