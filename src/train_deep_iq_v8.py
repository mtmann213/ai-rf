import os
import tensorflow as tf
import numpy as np
from data_loader import RadioMLDataLoader
from resnet_opal_vanguard import build_resnet_vanguard

# Configuration
DATASET = "VDF_DEEP_INTELLIGENCE_500K.h5"
STARTING_WEIGHTS = "universal_resnet_v8_weights.h5"
FINAL_WEIGHTS = "vanguard_v8_deep_iq.h5"
CHECKPOINT_PATH = "deep_iq_checkpoint.keras"
BATCH_SIZE = 128 # Higher batch size for 500k samples
LEARNING_RATE = 0.0001
NUM_CLASSES = 57

def main():
    print(f"Opal Vanguard: Launching V8 Deep Intelligence Marathon (500k samples)")
    
    # 1. Build V8 Architecture
    model = build_resnet_vanguard(input_shape=(1024, 2), num_classes=NUM_CLASSES)
    
    # 2. Load the 12% Foundation Brain
    if os.path.exists(STARTING_WEIGHTS):
        print(f"Resuming from V8 foundation: {STARTING_WEIGHTS}")
        model.load_weights(STARTING_WEIGHTS)
    
    # 3. Setup Dataset
    loader = RadioMLDataLoader(DATASET, num_classes=NUM_CLASSES)
    
    # Hardcoded SIGNALS_SHARED_LIST from TorchSig to avoid dependency in training container
    loader.modulations = [
        'ook', '4ask', '8ask', '16ask', '32ask', '64ask', '2fsk', '2gfsk', '2msk', '2gmsk',
        '4fsk', '4gfsk', '4msk', '4gmsk', '8fsk', '8gfsk', '8msk', '8gmsk', '16fsk', '16gfsk',
        '16msk', '16gmsk', 'bpsk', 'qpsk', '8psk', '16psk', '32psk', '64psk', '16qam', '32qam',
        '32qam_cross', '64qam', '128qam_cross', '256qam', '512qam_cross', '1024qam', 'ofdm-64',
        'ofdm-72', 'ofdm-128', 'ofdm-180', 'ofdm-256', 'ofdm-300', 'ofdm-512', 'ofdm-600',
        'ofdm-900', 'ofdm-1024', 'ofdm-1200', 'ofdm-2048', 'fm', 'am-dsb-sc', 'am-dsb',
        'am-lsb', 'am-usb', 'lfm_data', 'lfm_radar', 'chirpss', 'tone'
    ]
    
    train_indices, val_indices = loader.get_train_val_indices(test_size=0.05) # 25k samples for val
    
    train_ds = tf.data.Dataset.from_generator(
        lambda: loader.get_generator(train_indices, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: loader.get_generator(val_indices, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, NUM_CLASSES), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    # 4. Compile & Marathon
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger('deep_iq_marathon.csv', append=True)
    ]

    print(f"\n--- Starting 500k Sample Marathon ---")
    steps = len(train_indices) // BATCH_SIZE
    val_steps = len(val_indices) // BATCH_SIZE
    
    model.fit(train_ds, 
              epochs=50, 
              steps_per_epoch=steps,
              validation_data=val_ds,
              validation_steps=val_steps,
              callbacks=callbacks)
    
    model.save_weights(FINAL_WEIGHTS)
    print(f"Mission Success: Deep IQ model saved to {FINAL_WEIGHTS}")

if __name__ == "__main__":
    main()
