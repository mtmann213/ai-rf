import os
import tensorflow as tf
import numpy as np
import h5py
from src.data_loader import RadioMLDataLoader
from src.resnet_lstm_v9 import build_resnet_lstm_v9

# Configuration
VDF_DATASET = "data/VDF_SPECTER_GOLDEN.h5"
BASE_DATASET = "2018_01A/GOLD_XYZ_OSC.0001_1024.hdf5"
MODEL_PATH = 'models/vanguard_v9_specialist_final.keras'
CHECKPOINT_PATH = 'models/v9_specialist_checkpoint.keras'
RESUME_PATH = 'models/v9_ensemble_checkpoint.keras'
BATCH_SIZE = 64
LEARNING_RATE = 0.00001 # Surgical learning rate for fine-tuning
NUM_CLASSES = 57

# "Target List" for oversampling (Mapped to the 24-class Hardware indices)
# 1: 16apsk, 2: 32qam, 3: fm, 10: am-ssb-sc, 14: 128qam, 18: 64qam, 21: am-dsb-wc
TARGET_INDICES = [1, 2, 3, 10, 14, 18, 21]

def load_vdf_to_ram(file_path):
    print(f"Loading {file_path} into RAM...")
    with h5py.File(file_path, 'r') as f:
        x = f['X'][:]
        y = f['Y'][:]
    print(f"RAM Load Complete: {x.shape[0]} samples cached.")
    return x, y

def specialist_generator(x_vdf, y_vdf, indices_vdf, loader_base, indices_base, batch_size):
    """
    Weighted Generator: 
    - 40% Targeted Hardware (QAM/Analog Confusions)
    - 30% Random Hardware (Generalization anchor)
    - 30% Simulation (Physics anchor)
    """
    hw_target_size = int(batch_size * 0.4)
    hw_random_size = int(batch_size * 0.3)
    sim_size = batch_size - hw_target_size - hw_random_size
    
    # Identify samples in VDF that belong to our target classes
    y_vdf_classes = np.argmax(y_vdf, axis=1)
    target_sample_mask = np.isin(y_vdf_classes, TARGET_INDICES)
    target_vdf_indices = np.intersect1d(indices_vdf, np.where(target_sample_mask)[0])
    
    if len(target_vdf_indices) == 0:
        print("CRITICAL ERROR: No target samples found in VDF. Check TARGET_INDICES mapping.")
        # Fallback to random to prevent crash
        target_vdf_indices = indices_vdf
    else:
        print(f"Specialist Engine: {len(target_vdf_indices)} target samples identified for oversampling.")
    
    gen_base = loader_base.get_generator(indices_base, sim_size)
    
    while True:
        # 1. Sim Data
        try:
            x_base, y_base = next(gen_base)
        except StopIteration:
            gen_base = loader_base.get_generator(indices_base, sim_size)
            x_base, y_base = next(gen_base)
            
        # 2. Targeted Hardware Data (Oversampled)
        target_idx = np.random.choice(target_vdf_indices, hw_target_size, replace=True)
        x_target = x_vdf[target_idx]
        y_target = y_vdf[target_idx]
        
        # 3. Random Hardware Data
        random_idx = np.random.choice(indices_vdf, hw_random_size, replace=False)
        x_random = x_vdf[random_idx]
        y_random = y_vdf[random_idx]
        
        # 4. Combine
        x_mixed = np.concatenate([x_target, x_random, x_base], axis=0)
        
        y_base_padded = np.zeros((y_base.shape[0], NUM_CLASSES), dtype=np.float32)
        y_base_padded[:, :y_base.shape[1]] = y_base
        y_mixed = np.concatenate([y_target, y_random, y_base_padded], axis=0)
        
        p = np.random.permutation(len(x_mixed))
        yield x_mixed[p], y_mixed[p]

def main():
    print(f"Opal Vanguard: Launching V9.1 Specialist Refinement")
    print(f"Targeting QAM Density and Analog Blending confusions.")
    
    model = build_resnet_lstm_v9(input_shape=(1024, 2), num_classes=NUM_CLASSES)
    
    # Load the 68.5% Breakthrough Weights
    if os.path.exists(RESUME_PATH):
        print(f"Resuming from V9 Breakthrough: {RESUME_PATH}")
        model.load_weights(RESUME_PATH)
    else:
        print(f"ERROR: {RESUME_PATH} not found. Cannot run specialist refinement.")
        return

    # Load Data
    x_vdf, y_vdf_raw = load_vdf_to_ram(VDF_DATASET)
    y_vdf = np.zeros((y_vdf_raw.shape[0], NUM_CLASSES), dtype=np.float32)
    y_vdf[:, :24] = y_vdf_raw[:, :24]
    
    loader_vdf = RadioMLDataLoader(VDF_DATASET, num_classes=24) 
    train_idx_vdf, val_idx_vdf = loader_vdf.get_train_val_indices(test_size=0.1)

    loader_base = RadioMLDataLoader(BASE_DATASET, num_classes=24)
    train_idx_base, val_idx_base = loader_base.get_train_val_indices(test_size=0.1)
    
    # Create Specialist Datasets
    train_ds = tf.data.Dataset.from_generator(
        lambda: specialist_generator(x_vdf, y_vdf, train_idx_vdf, loader_base, train_idx_base, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: specialist_generator(x_vdf, y_vdf, val_idx_vdf, loader_base, val_idx_base, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    # Compile with Surgical Precision
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger('reports/v9_specialist_log.csv', append=True)
    ]

    # We use fewer steps per epoch but focus more on target classes
    steps = 5000 
    val_steps = 500
    
    print(f"Starting Specialist Refinement...")
    model.fit(train_ds, epochs=50, steps_per_epoch=steps,
              validation_data=val_ds, validation_steps=val_steps,
              callbacks=callbacks)
    
    model.save(MODEL_PATH)
    print(f"Mission Success: Specialist model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
