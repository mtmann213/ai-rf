import os
import tensorflow as tf
from data_loader import RadioMLDataLoader
from resnet_opal_vanguard import build_resnet_vanguard

# Configuration
DATASET_PATH = "2018_01A/GOLD_XYZ_OSC.0001_1024.hdf5"
MODEL_SAVE_PATH = "opal_vanguard_resnet_v1.h5"
INPUT_SHAPE = (1024, 2)
NUM_CLASSES = 24
BATCH_SIZE = 64 
EPOCHS = 50 # Increased for real dataset

def main():
    print(f"Opal Vanguard: Training ResNet from local dataset {DATASET_PATH} (Streaming Mode)...")
    
    # 1. Initialize Loader & Get Indices
    loader = RadioMLDataLoader(DATASET_PATH)
    train_indices, val_indices = loader.get_train_val_indices()
    
    # 2. Build tf.data Datasets
    train_dataset = tf.data.Dataset.from_generator(
        lambda: loader.get_generator(train_indices, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 24), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: loader.get_generator(val_indices, BATCH_SIZE),
        output_signature=(
            tf.TensorSpec(shape=(None, 1024, 2), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 24), dtype=tf.float32)
        )
    ).prefetch(tf.data.AUTOTUNE)
    
    # 3. Resume or Build Model
    initial_epoch = 0
    checkpoint_path = 'best_resnet.keras'
    log_path = 'training_log.csv'
    
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint at {checkpoint_path}. Loading model...")
        model = tf.keras.models.load_model(checkpoint_path)
        # Only resume epoch count if we have the weights
        if os.path.exists(log_path):
            try:
                import pandas as pd
                log_df = pd.read_csv(log_path)
                if not log_df.empty:
                    initial_epoch = log_df['epoch'].max() + 1
                    print(f"Resuming from Epoch {initial_epoch}")
            except Exception as e:
                print(f"Could not read logs for epoch tracking: {e}")
    else:
        print("No checkpoint found. Building fresh ResNet model...")
        model = build_resnet_vanguard(INPUT_SHAPE, NUM_CLASSES)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
    
    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        # Save at end of epoch to ensure val_loss is available
        tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True),
        tf.keras.callbacks.CSVLogger(log_path, append=True)
    ]
    # 5. Train
    steps_per_epoch = len(train_indices) // BATCH_SIZE
    validation_steps = len(val_indices) // BATCH_SIZE

    print(f"\n--- Phase 3: Training ResNet on Full RadioML Dataset ---")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Batch Size: {BATCH_SIZE} | Total Steps: {steps_per_epoch}")
    
    model.fit(train_dataset, 
              epochs=EPOCHS, 
              initial_epoch=initial_epoch,
              steps_per_epoch=steps_per_epoch,
              validation_data=val_dataset,
              validation_steps=validation_steps,
              callbacks=callbacks,
              verbose=1)
    
    model.save(MODEL_SAVE_PATH)
    print(f"\nResNet Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
