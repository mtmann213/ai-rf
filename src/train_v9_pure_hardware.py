import os
import tensorflow as tf
import numpy as np
import h5py
from src.data_loader import RadioMLDataLoader
from src.resnet_lstm_polar_v9 import build_resnet_lstm_polar_v9

# Configuration: THE PURE HARDWARE MISSION
VDF_DATASET = "data/VDF_SPECTER_GOLDEN.h5"
NUTRIENT_DATASET = "data/VDF_SPECIALIST_NUTRIENTS.h5" # THE FRESH LAPTOP DATA

MODEL_PATH = 'models/vanguard_v9_pure_hardware_final.keras'
CHECKPOINT_PATH = 'models/v9_pure_hardware_checkpoint.keras'
RESUME_PATH = 'models/vanguard_v9_specialist_sovereign.keras' # Resuming from 61% baseline

BATCH_SIZE = 64
LEARNING_RATE = 0.00001 
NUM_CLASSES = 57

# "Target List" for oversampling within the hardware dataset
TARGET_INDICES = [1, 2, 3, 10, 14, 18, 21]

def load_dataset_to_ram(file_path):
    print(f"Loading {file_path} into RAM...")
    with h5py.File(file_path, 'r') as f:
        x = f['X'][:]
        y = f['Y'][:]
    print(f"RAM Load Complete: {x.shape[0]} samples cached.")
    return x, y

def pure_hardware_generator(x_vdf, y_vdf, indices_vdf, x_nuts, y_nuts, batch_size):
    """
    V9.6 Pure Hardware Generator:
    - 50% Specialized Nutrients (from Laptop)
    - 50% Real Hardware (from Specter Golden)
    - NO SIMULATION DISTRACTIONS
    """
    nut_size = batch_size // 2
    hw_size = batch_size - nut_size
    
    # Prep Hardware targets
    y_vdf_classes = np.argmax(y_vdf, axis=1)
    target_sample_mask = np.isin(y_vdf_classes, TARGET_INDICES)
    target_vdf_indices = np.intersect1d(indices_vdf, np.where(target_sample_mask)[0])
    
    indices_nuts = np.arange(len(x_nuts))
    
    while True:
        # 1. Specialist Nutrients (50%)
        n_idx = np.random.choice(indices_nuts, nut_size, replace=True)
        x_n, y_n = x_nuts[n_idx], y_nuts[n_idx]
        
        # 2. Real Hardware (50% - Weighted)
        hw_target_count = int(hw_size * 0.7)
        hw_random_count = hw_size - hw_target_count
        
        t_idx = np.random.choice(target_vdf_indices, hw_target_count, replace=True)
        r_idx = np.random.choice(indices_vdf, hw_random_count, replace=False)
        
        x_hw = np.concatenate([x_vdf[t_idx], x_vdf[r_idx]], axis=0)
        y_hw = np.concatenate([y_vdf[t_idx], y_vdf[r_idx]], axis=0)
        
        # 3. Combine IQ
        x_iq_mixed = np.concatenate([x_n, x_hw], axis=0)
        
        # 4. Generate Polar
        i_comp, q_comp = x_iq_mixed[:, :, 0], x_iq_mixed[:, :, 1]
        amplitude = np.sqrt(i_comp**2 + q_comp**2)
        phase = np.arctan2(q_comp, i_comp)
        amplitude = amplitude / (1.0 + np.abs(amplitude))
        phase = phase / np.pi
        x_polar_mixed = np.stack([amplitude, phase], axis=-1)
        
        # 5. Combine Labels
        y_mixed = np.concatenate([y_n, y_hw], axis=0)
        
        p = np.random.permutation(len(x_iq_mixed))
        yield (x_iq_mixed[p], x_polar_mixed[p]), y_mixed[p]

def main():
    print(f"Opal Vanguard: Launching V9.6 PURE HARDWARE Refinement")
    print(f"Strategy: Specialized Nutrients + Real Captures (No Simulation).")
    
    model = build_resnet_lstm_polar_v9(iq_shape=(1024, 2), polar_shape=(1024, 2), num_classes=NUM_CLASSES)
    
    if os.path.exists(RESUME_PATH):
        print(f"Loading Sovereign Foundation: {RESUME_PATH}")
        model.load_weights(RESUME_PATH)

    # Load ONLY Hardware Intelligence
    x_vdf, y_vdf_raw = load_dataset_to_ram(VDF_DATASET)
    y_vdf = np.zeros((y_vdf_raw.shape[0], NUM_CLASSES), dtype=np.float32)
    y_vdf[:, :24] = y_vdf_raw[:, :24]
    
    x_nuts, y_nuts = load_dataset_to_ram(NUTRIENT_DATASET)
    
    loader_vdf = RadioMLDataLoader(VDF_DATASET, num_classes=24) 
    train_idx_vdf, val_idx_vdf = loader_vdf.get_train_val_indices(test_size=0.1)

    train_ds = tf.data.Dataset.from_generator(
        lambda: pure_hardware_generator(x_vdf, y_vdf, train_idx_vdf, x_nuts, y_nuts, BATCH_SIZE),
        output_signature=((tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
                           tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32)),
                          tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32))
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: pure_hardware_generator(x_vdf, y_vdf, val_idx_vdf, x_nuts, y_nuts, BATCH_SIZE),
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
        tf.keras.callbacks.CSVLogger('reports/v9_pure_hardware_log.csv', append=True)
    ]

    print(f"Igniting PURE HARDWARE Marathon...")
    model.fit(train_ds, epochs=100, steps_per_epoch=5000,
              validation_data=val_ds, validation_steps=500,
              callbacks=callbacks)
    
    model.save(MODEL_PATH)
    print(f"Mission Success: Sovereign Pure Hardware saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
