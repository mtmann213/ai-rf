import os
import sys
import h5py
import numpy as np
import time
import gc
from tqdm import tqdm
from torchsig.utils.defaults import default_dataset
from torchsig.signals.signal_lists import SIGNALS_SHARED_LIST

# Configuration for Specialist Nutrients
NUM_CLASSES = 57
SAMPLES_PER_CLASS = 10000 
TARGET_CLASSES = ['32qam', '64qam', '128qam_cross', 'am-dsb', 'fm', 'ook', '16qam']
TOTAL_SAMPLES = SAMPLES_PER_CLASS * len(TARGET_CLASSES)
OUTPUT_FILE = "data/VDF_SPECIALIST_NUTRIENTS.h5"
IQ_LENGTH = 1024

def generate_specialist_dataset():
    print(f"Opal Vanguard: Initiating Stabilized Nutrient Factory")
    print(f"Targeting {len(TARGET_CLASSES)} classes | {TOTAL_SAMPLES} samples.")
    sys.stdout.flush()
    
    os.makedirs("data", exist_ok=True)
    class_to_idx = {name: i for i, name in enumerate(SIGNALS_SHARED_LIST)}

    # Open with 'w' to ensure a clean header after the previous crashes
    with h5py.File(OUTPUT_FILE, 'w') as f:
        x_ds = f.create_dataset('X', (TOTAL_SAMPLES, IQ_LENGTH, 2), dtype='float32')
        y_ds = f.create_dataset('Y', (TOTAL_SAMPLES, NUM_CLASSES), dtype='float32')
        current_idx = 0

        level = 2
        
        for class_name in TARGET_CLASSES:
            class_idx = class_to_idx[class_name]
            print(f"\n--- Generating {class_name} ---")
            sys.stdout.flush()
            
            dataset = default_dataset(
                impairment_level=level,
                num_iq_samples_dataset=IQ_LENGTH,
                num_samples_dataset=SAMPLES_PER_CLASS,
                use_class_list=[class_name]
            )
            
            it = iter(dataset)
            pbar = tqdm(total=SAMPLES_PER_CLASS, desc=f" {class_name}", unit="samples")
            
            for _ in range(SAMPLES_PER_CLASS):
                try:
                    try:
                        data, _ = next(it)
                    except ValueError:
                        data = np.zeros(IQ_LENGTH, dtype=np.complex64)
                    
                    if len(data) > IQ_LENGTH: data = data[:IQ_LENGTH]
                    elif len(data) < IQ_LENGTH: data = np.pad(data, (0, IQ_LENGTH - len(data)), 'constant')
                    
                    iq = np.stack([np.real(data), np.imag(data)], axis=-1)
                    iq = iq / (1.0 + np.abs(iq)) # Soft-Clip
                    
                    x_ds[current_idx] = iq
                    
                    one_hot = np.zeros(NUM_CLASSES)
                    one_hot[class_idx] = 1.0 
                    y_ds[current_idx] = one_hot
                    
                    current_idx += 1
                    pbar.update(1)
                except StopIteration:
                    break
            
            pbar.close()
            
            # STABILITY HOOKS:
            print(f" Class {class_name} complete. Flushing to disk and cooling down...")
            f.flush() # Force HDF5 to write buffers to physical disk
            del dataset
            del it
            gc.collect() # Force Python to clear memory
            time.sleep(10) # 10s cooldown to prevent thermal/bus throttling
            sys.stdout.flush()

    print(f"\nMission Success: Specialist Nutrients saved to {OUTPUT_FILE}")
    sys.stdout.flush()

if __name__ == "__main__":
    generate_specialist_dataset()
