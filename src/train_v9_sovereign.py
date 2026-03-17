import os
import tensorflow as tf
import numpy as np
import h5py
from src.data_loader import RadioMLDataLoader
from src.resnet_lstm_polar_v9 import build_resnet_lstm_polar_v9

# Configuration
VDF_DATASET = "data/VDF_SPECTER_GOLDEN.h5"
BASE_DATASET = "2018_01A/GOLD_XYZ_OSC.0001_1024.hdf5"
MODEL_PATH = 'models/vanguard_v9_sovereign_final.keras'
CHECKPOINT_PATH = 'models/v9_sovereign_checkpoint.keras'
BATCH_SIZE = 64
LEARNING_RATE = 0.00005 
NUM_CLASSES = 57

# Focused Target List: QAM, Analog, and APSK confusions
TARGET_INDICES = [1, 2, 3, 10, 14, 18, 21]

def load_vdf_to_ram(file_path):
    print(f"Loading {file_path} into RAM...")
    with h5py.File(file_path, 'r') as f:
        x = f['X'][:]
        y = f['Y'][:]
    print(f"RAM Load Complete: {x.shape[0]} samples cached.")
    return x, y

def sovereign_generator(x_vdf, y_vdf, indices_vdf, loader_base, indices_base, batch_size):
    """
    Multi-Modal Specialist Generator:
    - Yields: ([IQ_Batch, Polar_Batch], Labels)
    - 40% Targeted Hardware, 30% Random Hardware, 30% Simulation
    """
    hw_target_size = int(batch_size * 0.4)
    hw_random_size = int(batch_size * 0.3)
    sim_size = batch_size - hw_target_size - hw_random_size
    
    y_vdf_classes = np.argmax(y_vdf, axis=1)
    target_sample_mask = np.isin(y_vdf_classes, TARGET_INDICES)
    target_vdf_indices = np.intersect1d(indices_vdf, np.where(target_sample_mask)[0])
    
    gen_base = loader_base.get_generator(indices_base, sim_size)
    
    while True:
        try:
            x_base, y_base = next(gen_base)
        except StopIteration:
            gen_base = loader_base.get_generator(indices_base, sim_size)
            x_base, y_base = next(gen_base)
            
        target_idx = np.random.choice(target_vdf_indices, hw_target_size, replace=True)
        x_target = x_vdf[target_idx]
        y_target = y_vdf[target_idx]
        
        random_idx = np.random.choice(indices_vdf, hw_random_size, replace=False)
        x_random = x_vdf[random_idx]
        y_random = y_vdf[random_idx]
        
        # 1. Combine IQ Stream
        x_iq_mixed = np.concatenate([x_target, x_random, x_base], axis=0)
        
        # 2. Generate Polar Stream (Amplitude & Phase)
        # x_iq_mixed shape: (Batch, 1024, 2) where [:, :, 0] is I and [:, :, 1] is Q
        i_comp = x_iq_mixed[:, :, 0]
        q_comp = x_iq_mixed[:, :, 1]
        
        amplitude = np.sqrt(i_comp**2 + q_comp**2)
        phase = np.arctan2(q_comp, i_comp)
        
        # Normalize Polar components to [0, 1] or [-1, 1] for stability
        amplitude = amplitude / (1.0 + np.abs(amplitude))
        phase = phase / np.pi # Phase is naturally [-pi, pi]
        
        x_polar_mixed = np.stack([amplitude, phase], axis=-1)
        
        # 3. Combine Labels
        y_base_padded = np.zeros((y_base.shape[0], NUM_CLASSES), dtype=np.float32)
        y_base_padded[:, :y_base.shape[1]] = y_base
        y_mixed = np.concatenate([y_target, y_random, y_base_padded], axis=0)
        
        p = np.random.permutation(len(x_iq_mixed))
        # Yield as a tuple of inputs for Keras multi-modal stability
        yield (x_iq_mixed[p], x_polar_mixed[p]), y_mixed[p]

def main():
    print(f"Opal Vanguard: Launching V9.2 SOVEREIGN EYE Multi-Modal Ignition")
    print(f"Goal: Solving QAM/Analog density via Polar Feature Injection.")
    
    # 1. Build Sovereign Architecture
    model = build_resnet_lstm_polar_v9(iq_shape=(1024, 2), polar_shape=(1024, 2), num_classes=NUM_CLASSES)
    
    # 2. Data Load
    x_vdf, y_vdf_raw = load_vdf_to_ram(VDF_DATASET)
    y_vdf = np.zeros((y_vdf_raw.shape[0], NUM_CLASSES), dtype=np.float32)
    y_vdf[:, :24] = y_vdf_raw[:, :24]
    
    loader_vdf = RadioMLDataLoader(VDF_DATASET, num_classes=24) 
    train_idx_vdf, val_idx_vdf = loader_vdf.get_train_val_indices(test_size=0.1)

    loader_base = RadioMLDataLoader(BASE_DATASET, num_classes=24)
    train_idx_base, val_idx_base = loader_base.get_train_val_indices(test_size=0.1)
    
    # 3. Datasets
    train_ds = tf.data.Dataset.from_generator(
        lambda: sovereign_generator(x_vdf, y_vdf, train_idx_vdf, loader_base, train_idx_base, BATCH_SIZE),
        output_signature=(
            (tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: sovereign_generator(x_vdf, y_vdf, val_idx_vdf, loader_base, train_idx_base, BATCH_SIZE),
        output_signature=(
            (tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
             tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32)),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    # 4. Compile & Ignite
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger('reports/v9_sovereign_log.csv', append=True)
    ]

    steps = 5000 
    val_steps = 500
    
    print(f"Starting Sovereign Marathon...")
    model.fit(train_ds, epochs=50, steps_per_epoch=steps,
              validation_data=val_ds, validation_steps=val_steps,
              callbacks=callbacks)
    
    model.save(MODEL_PATH)
    print(f"Mission Success: Sovereign model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
