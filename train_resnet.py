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
    
    # 2. Build tf.data Datasets from Generator
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
    
    # 3. Build ResNet Model
    model = build_resnet_vanguard(INPUT_SHAPE, NUM_CLASSES)
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_resnet.keras', save_best_only=True)
    ]
    
    # 5. Train
    # Need to specify steps_per_epoch since generators are infinite
    steps_per_epoch = len(train_indices) // BATCH_SIZE
    validation_steps = len(val_indices) // BATCH_SIZE

    print("\n--- Phase 3: Training ResNet on Full RadioML Dataset (Streaming) ---")
    model.fit(train_dataset, 
              epochs=EPOCHS, 
              steps_per_epoch=steps_per_epoch,
              validation_data=val_dataset,
              validation_steps=validation_steps,
              callbacks=callbacks,
              verbose=1)
    
    # 6. Save Final Model
    model.save(MODEL_SAVE_PATH)
    print(f"\nResNet Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
