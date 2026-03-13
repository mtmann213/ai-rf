import os
import tensorflow as tf
import numpy as np
from data_loader import RadioMLDataLoader
from train_opal_vanguard import build_opal_vanguard_model, INPUT_SHAPE, NUM_CLASSES, BATCH_SIZE

# Configuration
DATASET_PATH = "GOLD_XYZ_OSC.0001_1024.hdf5"
MODEL_SAVE_PATH = "opal_vanguard_v2.h5"
EPOCHS = 20

def main():
    print(f"Opal Vanguard: Training from local dataset {DATASET_PATH}...")
    
    # 1. Initialize Loader
    loader = RadioMLDataLoader(DATASET_PATH)
    
    # 2. Get Train/Test Split
    # Split returns: (x_train, y_train, z_train), (x_test, y_test, z_test)
    (x_train, y_train, z_train), (x_test, y_test, z_test) = loader.get_train_test_split(test_size=0.2)
    
    # 3. Build Model
    model = build_opal_vanguard_model(NUM_CLASSES)
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', # Using categorical since RadioML Y is one-hot
                  metrics=['accuracy'])
    
    # 4. Train
    print("\n--- Phase 2: Training on Synthetic RadioML Dataset ---")
    history = model.fit(x_train, y_train, 
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_data=(x_test, y_test),
                        verbose=1)
    
    # 5. Save
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")
    print("Next step: Run benchmark_snr.py to evaluate performance.")

if __name__ == "__main__":
    main()
