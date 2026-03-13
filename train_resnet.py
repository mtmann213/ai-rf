import os
import tensorflow as tf
from data_loader import RadioMLDataLoader
from resnet_opal_vanguard import build_resnet_vanguard

# Configuration
DATASET_PATH = "GOLD_XYZ_OSC.0001_1024.hdf5"
MODEL_SAVE_PATH = "opal_vanguard_resnet_v1.h5"
INPUT_SHAPE = (1024, 2)
NUM_CLASSES = 24
BATCH_SIZE = 64 # ResNet is more memory intensive
EPOCHS = 10

def main():
    print(f"Opal Vanguard: Training ResNet from local dataset {DATASET_PATH}...")
    
    # 1. Initialize Loader
    loader = RadioMLDataLoader(DATASET_PATH)
    (x_train, y_train, z_train), (x_test, y_test, z_test) = loader.get_train_test_split(test_size=0.2)
    
    # 2. Build ResNet Model
    model = build_resnet_vanguard(INPUT_SHAPE, NUM_CLASSES)
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # 3. Callbacks (Good practice for deeper models)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_resnet.keras', save_best_only=True)
    ]
    
    # 4. Train
    print("\n--- Phase 3: Training ResNet on Synthetic RadioML Dataset ---")
    history = model.fit(x_train, y_train, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_data=(x_test, y_test),
                        callbacks=callbacks,
                        verbose=1)
    
    # 5. Save Final Model
    model.save(MODEL_SAVE_PATH)
    print(f"\nResNet Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
