import os
import tensorflow as tf
import numpy as np
from data_loader import RadioMLDataLoader
from resnet_opal_vanguard import build_resnet_vanguard

# Configuration
TORCHSIG_DATASET = "torchsig_v2_industrial.h5"
FOUNDATION_WEIGHTS = "best_resnet_v7_weights.h5"
NEW_MODEL_WEIGHTS = "universal_resnet_v8_weights.h5"
BATCH_SIZE = 64
LEARNING_RATE = 0.0001 # Higher rate for the new classes
NUM_CLASSES = 57 # TorchSig V2 Standard

def main():
    print(f"Opal Vanguard: Launching V8 Universal Brain Expansion ({NUM_CLASSES} Classes)")
    
    # 1. Build the V8 Architecture (57 classes)
    print("Building V8 ResNet architecture...")
    model = build_resnet_vanguard(input_shape=(1024, 2), num_classes=NUM_CLASSES)
    
    # 2. Transfer Weights from V7 Foundation
    if os.path.exists(FOUNDATION_WEIGHTS):
        print(f"Transferring feature extraction weights from {FOUNDATION_WEIGHTS}...")
        # We load by name. The final layer 'dense_head_24' will be ignored 
        # because the new model has 'dense_head_57'.
        model.load_weights(FOUNDATION_WEIGHTS, by_name=True, skip_mismatch=True)
        print("Feature extraction layers initialized from foundation. Classification head is fresh.")
    else:
        print("Warning: Foundation weights not found. Training from scratch.")

    # 3. Setup Dataset
    # We'll use the basic data loader but skip the 2018 reconstruction logic
    # since torchsig_v2_industrial.h5 is fresh and correct.
    loader = RadioMLDataLoader(TORCHSIG_DATASET, num_classes=NUM_CLASSES)
    # We need to manually set the modulations list for TorchSig
    from torchsig.signals.signal_lists import SIGNALS_SHARED_LIST
    loader.modulations = SIGNALS_SHARED_LIST
    
    train_indices, val_indices = loader.get_train_val_indices(test_size=0.1)
    
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

    # 4. Compile & Ignite
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'], jit_compile=False)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(NEW_MODEL_WEIGHTS, save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger('training_log_v8.csv', append=True)
    ]

    print(f"\n--- Starting Universal Training (100,000 samples) ---")
    steps = len(train_indices) // BATCH_SIZE
    val_steps = len(val_indices) // BATCH_SIZE
    
    model.fit(train_ds, 
              epochs=50, 
              steps_per_epoch=steps,
              validation_data=val_ds,
              validation_steps=val_steps,
              callbacks=callbacks)
    
    print(f"Universal model saved: {NEW_MODEL_WEIGHTS}")

if __name__ == "__main__":
    main()
