import os
import tensorflow as tf
import numpy as np
import h5py
from src.data_loader import RadioMLDataLoader
from src.resnet_lstm_polar_v9 import build_resnet_lstm_polar_v9

# Configuration
VDF_DATASET = "data/VDF_SPECTER_GOLDEN.h5"
BASE_DATASET = "2018_01A/GOLD_XYZ_OSC.0001_1024.hdf5"
SYNTH_DATASET = "data/VDF_DEEP_INTELLIGENCE_500K.h5" # THE THIRD SOURCE
MODEL_PATH = 'models/vanguard_v9_specialist_sovereign.keras'
CHECKPOINT_PATH = 'models/v9_specialist_sovereign_checkpoint.keras'
BATCH_SIZE = 64
LEARNING_RATE = 0.00002 
NUM_CLASSES = 57

# "Target List" for oversampling (Mapped to 24-class Hardware indices)
# 1: 16apsk, 2: 32qam, 3: fm, 10: am-ssb-sc, 14: 128qam, 18: 64qam, 21: am-dsb-wc
TARGET_INDICES = [1, 2, 3, 10, 14, 18, 21]

def load_dataset_to_ram(file_path):
    print(f"Loading {file_path} into RAM...")
    with h5py.File(file_path, 'r') as f:
        x = f['X'][:]
        y = f['Y'][:]
    print(f"RAM Load Complete: {x.shape[0]} samples cached.")
    return x, y

def triple_source_generator(x_vdf, y_vdf, indices_vdf, x_synth, y_synth, loader_base, indices_base, batch_size):
    """
    The Ultimate Nutrient Engine:
    - 50% Hardware (High-Targeted Specialist)
    - 25% High-Fidelity Simulation (500k Set)
    - 25% Physics Baseline (2018.01A)
    """
    hw_size = batch_size // 2
    synth_size = batch_size // 4
    base_size = batch_size - hw_size - synth_size
    
    # Prep Hardware targets
    y_vdf_classes = np.argmax(y_vdf, axis=1)
    target_sample_mask = np.isin(y_vdf_classes, TARGET_INDICES)
    target_vdf_indices = np.intersect1d(indices_vdf, np.where(target_sample_mask)[0])
    
    # Prep synthetic indices
    indices_synth = np.arange(len(x_synth))
    
    gen_base = loader_base.get_generator(indices_base, base_size)
    
    while True:
        # 1. Base Sim
        try:
            x_base, y_base = next(gen_base)
        except StopIteration:
            gen_base = loader_base.get_generator(indices_base, base_size)
            x_base, y_base = next(gen_base)
            
        # 2. Hardware (Weighted)
        # 70% of the HW batch is targeted confusions
        hw_target_count = int(hw_size * 0.7)
        hw_random_count = hw_size - hw_target_count
        
        t_idx = np.random.choice(target_vdf_indices, hw_target_count, replace=True)
        r_idx = np.random.choice(indices_vdf, hw_random_count, replace=False)
        
        x_hw = np.concatenate([x_vdf[t_idx], x_vdf[r_idx]], axis=0)
        y_hw = np.concatenate([y_vdf[t_idx], y_vdf[r_idx]], axis=0)
        
        # 3. High-Fidelity Synthetic (25%)
        s_idx = np.random.choice(indices_synth, synth_size, replace=False)
        x_s = x_synth[s_idx]
        y_s = y_synth[s_idx]
        
        # 4. Combine IQ
        x_iq_mixed = np.concatenate([x_hw, x_s, x_base], axis=0)
        
        # 5. Generate Polar
        i_comp, q_comp = x_iq_mixed[:, :, 0], x_iq_mixed[:, :, 1]
        amplitude = np.sqrt(i_comp**2 + q_comp**2)
        phase = np.arctan2(q_comp, i_comp)
        
        amplitude = amplitude / (1.0 + np.abs(amplitude))
        phase = phase / np.pi
        x_polar_mixed = np.stack([amplitude, phase], axis=-1)
        
        # 6. Combine Labels
        y_base_padded = np.zeros((y_base.shape[0], NUM_CLASSES), dtype=np.float32)
        y_base_padded[:, :y_base.shape[1]] = y_base
        y_mixed = np.concatenate([y_hw, y_s, y_base_padded], axis=0)
        
        p = np.random.permutation(len(x_iq_mixed))
        yield (x_iq_mixed[p], x_polar_mixed[p]), y_mixed[p]

def main():
    print(f"Opal Vanguard: Launching V9.3 TRIPLE-SOURCE Sovereign Specialist")
    
    model = build_resnet_lstm_polar_v9(iq_shape=(1024, 2), polar_shape=(1024, 2), num_classes=NUM_CLASSES)
    
    if os.path.exists('models/v9_sovereign_checkpoint.keras'):
        print("Resuming from V9.2 Sovereign weights...")
        model.load_weights('models/v9_sovereign_checkpoint.keras')

    # Load 3-Source Intelligence
    x_vdf, y_vdf_raw = load_dataset_to_ram(VDF_DATASET)
    y_vdf = np.zeros((y_vdf_raw.shape[0], NUM_CLASSES), dtype=np.float32)
    y_vdf[:, :24] = y_vdf_raw[:, :24]
    
    x_synth, y_synth = load_dataset_to_ram(SYNTH_DATASET)
    
    loader_vdf = RadioMLDataLoader(VDF_DATASET, num_classes=24) 
    train_idx_vdf, val_idx_vdf = loader_vdf.get_train_val_indices(test_size=0.1)

    loader_base = RadioMLDataLoader(BASE_DATASET, num_classes=24)
    train_idx_base, val_idx_base = loader_base.get_train_val_indices(test_size=0.1)
    
    train_ds = tf.data.Dataset.from_generator(
        lambda: triple_source_generator(x_vdf, y_vdf, train_idx_vdf, x_synth, y_synth, loader_base, train_idx_base, BATCH_SIZE),
        output_signature=((tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
                           tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32)),
                          tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32))
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: triple_source_generator(x_vdf, y_vdf, val_idx_vdf, x_synth, y_synth, loader_base, val_idx_base, BATCH_SIZE),
        output_signature=((tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
                           tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32)),
                          tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32))
    ).prefetch(tf.data.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger('reports/v9_specialist_sovereign_log.csv', append=True)
    ]

    print(f"Igniting Specialist Triple-Source Marathon...")
    model.fit(train_ds, epochs=100, steps_per_epoch=5000,
              validation_data=val_ds, validation_steps=500,
              callbacks=callbacks)
    
    model.save(MODEL_PATH)
    print(f"Mission Success: Sovereign Specialist Triple-Source saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
