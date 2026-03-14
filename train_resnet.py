import os
import sys
import tensorflow as tf
from data_loader import RadioMLDataLoader
from resnet_opal_vanguard import build_resnet_vanguard

# Configuration
DATASET_PATH = "2018_01A/GOLD_XYZ_OSC.0001_1024.hdf5"
MODEL_SAVE_PATH = "opal_vanguard_resnet_v7.h5"
INPUT_SHAPE = (1024, 2)
NUM_CLASSES = 24
BATCH_SIZE = 64 
EPOCHS = 50 

def main():
    print(f"\n[V7.0] Event Horizon Engine Engaged. Safe-Mode Active.")
    
    if not os.path.exists(DATASET_PATH):
        print(f"CRITICAL: Dataset missing at {DATASET_PATH}")
        return

    # 1. Init Loader
    loader = RadioMLDataLoader(DATASET_PATH)
    train_indices, val_indices = loader.get_train_val_indices()
    
    # 2. Datasets
    train_ds = tf.data.Dataset.from_generator(
        lambda: loader.get_generator(train_indices, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 24), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_generator(
        lambda: loader.get_generator(val_indices, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 24), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    
    # 3. Model
    checkpoint_path = 'best_resnet_v7.keras'
    initial_epoch = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming V7 from checkpoint: {checkpoint_path}")
        model = tf.keras.models.load_model(checkpoint_path)
    else:
        print("Building fresh Event Horizon ResNet...")
        model = build_resnet_vanguard(INPUT_SHAPE, NUM_CLASSES)
        # DOUBLE-CLIP Optimizer: Combines Norm and Value clipping
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001, 
            clipnorm=1.0,
            clipvalue=0.1
        )
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1)
        
        # Explicitly disable JIT (XLA) inside compile
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'], jit_compile=False)
    
    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
        tf.keras.callbacks.CSVLogger('training_log_v7.csv', append=True)
    ]
    
    # 5. Ignite
    print(f"--- Launching Event Horizon mission ---")
    sys.stdout.flush()
    
    steps = len(train_indices) // BATCH_SIZE
    val_steps = len(val_indices) // BATCH_SIZE

    model.fit(train_ds, 
              epochs=EPOCHS, 
              steps_per_epoch=steps,
              validation_data=val_ds,
              validation_steps=val_steps,
              callbacks=callbacks,
              verbose=1)
    
    model.save(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
