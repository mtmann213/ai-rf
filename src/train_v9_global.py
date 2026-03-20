import os
import tensorflow as tf
import numpy as np
import h5py
from src.data_loader import RadioMLDataLoader
from src.resnet_lstm_polar_v9 import build_resnet_lstm_polar_v9

# Configuration: THE SURGICAL NOVELTY (V9.10)
VDF_DATASET = "data/VDF_SPECTER_GOLDEN.h5"
MEGA_DATASET = "data/VDF_MEGA_SYNTHETIC_1M.h5"

MODEL_PATH = 'models/vanguard_v9_global_final.keras'
CHECKPOINT_PATH = 'models/v9_global_checkpoint.keras'
RESUME_PATH = 'models/v9_global_checkpoint.keras'

BATCH_SIZE = 64
LEARNING_RATE = 0.00003 
NUM_CLASSES = 57
# We take 1750 samples per class for a balanced 100k "Expert Chunk"
SAMPLES_PER_CLASS_LIMIT = 1750

def load_dataset_to_ram(file_path):
    print(f"Loading {file_path} into RAM...")
    with h5py.File(file_path, 'r') as f:
        x = f['X'][:]
        y = f['Y'][:]
    print(f"RAM Load Complete: {x.shape[0]} samples cached.")
    return x, y

def stratified_generator(x_vdf, y_vdf, indices_vdf, mega_path, batch_size):
    """
    V9.10 Stratified Generator:
    - Pulls balanced random samples across all 57 classes from the 1.14M set.
    - 80% Stratified Synthetic / 20% Hardware
    """
    synth_size = int(batch_size * 0.8)
    hw_size = batch_size - synth_size
    
    with h5py.File(mega_path, 'r') as f_mega:
        x_m_ds = f_mega['X']
        y_m_ds = f_mega['Y']
        
        # Pre-calculate class indices in the mega dataset for fast stratified access
        # Total samples = 1,140,000 | 57 classes * 20,000 samples each
        samples_per_class_in_file = 20000 
        
        while True:
            # 1. Stratified Synthetic (80%)
            selected_classes = np.random.randint(0, NUM_CLASSES, size=synth_size)
            m_idx = []
            for c in selected_classes:
                start = c * samples_per_class_in_file
                end = start + samples_per_class_in_file
                m_idx.append(np.random.randint(start, end))

            # UNIQUE and SORTED for HDF5 compliance
            m_idx = np.unique(m_idx)
            # Re-calculate actual batch size if unique reduced it
            current_synth_batch_size = len(m_idx)

            x_m, y_m = x_m_ds[m_idx], y_m_ds[m_idx]

            # 2. Hardware (20%)
            h_idx = np.random.choice(indices_vdf, hw_size, replace=False)
            x_h, y_h = x_vdf[h_idx], y_vdf[h_idx]

            # Combine IQ
            x_iq_mixed = np.concatenate([x_m, x_h], axis=0)

            i_comp, q_comp = x_iq_mixed[:, :, 0], x_iq_mixed[:, :, 1]
            amplitude = np.sqrt(i_comp**2 + q_comp**2) / (1.0 + np.abs(np.sqrt(i_comp**2 + q_comp**2)))
            phase = np.arctan2(q_comp, i_comp) / np.pi
            x_polar_mixed = np.stack([amplitude, phase], axis=-1)
            
            y_mixed = np.concatenate([y_m, y_h], axis=0)
            
            p = np.random.permutation(len(x_iq_mixed))
            yield (x_iq_mixed[p], x_polar_mixed[p]), y_mixed[p]

def main():
    print(f"Opal Vanguard: Launching V9.10 STRATIFIED NOVELTY Ignition")
    print(f"Goal: Breaking the 17% plateau via balanced master-class learning.")
    
    model = build_resnet_lstm_polar_v9(iq_shape=(1024, 2), polar_shape=(1024, 2), num_classes=NUM_CLASSES)
    
    if os.path.exists(RESUME_PATH):
        print(f"Resuming from checkpoint: {RESUME_PATH}")
        model.load_weights(RESUME_PATH)

    x_vdf, y_vdf_raw = load_dataset_to_ram(VDF_DATASET)
    y_vdf = np.zeros((y_vdf_raw.shape[0], NUM_CLASSES), dtype=np.float32)
    y_vdf[:, :24] = y_vdf_raw[:, :24]
    
    loader_vdf = RadioMLDataLoader(VDF_DATASET, num_classes=24) 
    train_idx_vdf, val_idx_vdf = loader_vdf.get_train_val_indices(test_size=0.1)

    train_ds = tf.data.Dataset.from_generator(
        lambda: stratified_generator(x_vdf, y_vdf, train_idx_vdf, MEGA_DATASET, BATCH_SIZE),
        output_signature=((tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
                           tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32)),
                          tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32))
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: stratified_generator(x_vdf, y_vdf, val_idx_vdf, MEGA_DATASET, BATCH_SIZE),
        output_signature=((tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
                           tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32)),
                          tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32))
    ).prefetch(tf.data.AUTOTUNE)

    optimizer = tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.CSVLogger('reports/v9_global_log.csv', append=True)
    ]

    print(f"Igniting Stratified Marathon...")
    model.fit(train_ds, epochs=100, steps_per_epoch=5000,
              validation_data=val_ds, validation_steps=500,
              callbacks=callbacks)
    
    model.save(MODEL_PATH)

if __name__ == "__main__":
    main()
