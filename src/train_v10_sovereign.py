import os
import tensorflow as tf
import numpy as np
import h5py
from src.data_loader import RadioMLDataLoader
from src.transformer_v10 import build_sovereign_transformer_v10

# Configuration: THE ATTENTION ERA (V10.0)
VDF_DATASET = "data/VDF_SPECTER_GOLDEN.h5"
MEGA_DATASET = "data/VDF_MEGA_SYNTHETIC_1M.h5"

MODEL_PATH = 'models/vanguard_v10_sovereign_final.keras'
CHECKPOINT_PATH = 'models/v10_sovereign_checkpoint.keras'

BATCH_SIZE = 64
LEARNING_RATE = 0.0001 # Transformers can handle higher LR than LSTMs
NUM_CLASSES = 57

def load_dataset_to_ram(file_path):
    print(f"Loading {file_path} into RAM...")
    with h5py.File(file_path, 'r') as f:
        x = f['X'][:]
        y = f['Y'][:]
    print(f"RAM Load Complete: {x.shape[0]} samples cached.")
    return x, y

def stratified_generator(x_vdf, y_vdf, indices_vdf, mega_path, batch_size):
    """
    V10.0 Stratified Generator:
    - 80% Stratified Synthetic (Novelty)
    - 20% Hardware (Realism Anchor)
    """
    synth_size = int(batch_size * 0.8)
    hw_size = batch_size - synth_size
    
    samples_per_class_in_file = 20000 
    
    with h5py.File(mega_path, 'r') as f_mega:
        x_m_ds = f_mega['X']
        y_m_ds = f_mega['Y']
        
        while True:
            # 1. Stratified Synthetic
            selected_classes = np.random.randint(0, NUM_CLASSES, size=synth_size)
            m_idx = []
            for c in selected_classes:
                start = c * samples_per_class_in_file
                end = start + samples_per_class_in_file
                m_idx.append(np.random.randint(start, end))
            
            m_idx = np.unique(m_idx)
            # Re-pad if unique reduced the size
            while len(m_idx) < synth_size:
                m_idx = np.append(m_idx, np.random.randint(0, 1140000))
            
            m_idx = sorted(m_idx)
            x_m, y_m = x_m_ds[m_idx], y_m_ds[m_idx]
            
            # 2. Hardware
            h_idx = np.random.choice(indices_vdf, hw_size, replace=False)
            x_h, y_h = x_vdf[h_idx], y_vdf[h_idx]
            
            # Combine
            x_mixed = np.concatenate([x_m, x_h], axis=0)
            y_mixed = np.concatenate([y_m, y_h], axis=0)
            
            p = np.random.permutation(len(x_mixed))
            yield x_mixed[p], y_mixed[p]

def main():
    print(f"Opal Vanguard: Launching V10.0 SOVEREIGN TRANSFORMER Ignition")
    print(f"Strategy: Self-Attention over Stratified Novelty.")
    
    model = build_sovereign_transformer_v10(input_shape=(1024, 2), num_classes=NUM_CLASSES)

    # Note: Starting V10 from scratch to build the 'Attention' foundation
    
    # Load Hardware into RAM for speed
    x_vdf, y_vdf_raw = load_dataset_to_ram(VDF_DATASET)
    y_vdf = np.zeros((y_vdf_raw.shape[0], NUM_CLASSES), dtype=np.float32)
    y_vdf[:, :24] = y_vdf_raw[:, :24]
    
    loader_vdf = RadioMLDataLoader(VDF_DATASET, num_classes=24) 
    train_idx_vdf, val_idx_vdf = loader_vdf.get_train_val_indices(test_size=0.1)

    train_ds = tf.data.Dataset.from_generator(
        lambda: stratified_generator(x_vdf, y_vdf, train_idx_vdf, MEGA_DATASET, BATCH_SIZE),
        output_signature=(tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
                          tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32))
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: stratified_generator(x_vdf, y_vdf, val_idx_vdf, MEGA_DATASET, BATCH_SIZE),
        output_signature=(tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
                          tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32))
    ).prefetch(tf.data.AUTOTUNE)

    # Compile
    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.CSVLogger('reports/v10_sovereign_log.csv', append=True),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    ]

    print(f"Igniting Transformer Marathon...")
    model.fit(train_ds, epochs=100, steps_per_epoch=5000,
              validation_data=val_ds, validation_steps=500,
              callbacks=callbacks)
    
    model.save(MODEL_PATH)
    print(f"Mission Success: Sovereign Transformer saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
