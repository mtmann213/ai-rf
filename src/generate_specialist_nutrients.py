import os
import sys
import h5py
import numpy as np
from tqdm import tqdm
from torchsig.utils.defaults import default_dataset
from torchsig.signals.signal_lists import SIGNALS_SHARED_LIST

# Configuration for Specialist Nutrients
NUM_CLASSES = 57
SAMPLES_PER_CLASS = 35000 # ~250k total samples
# The "Fatal 7" from our diagnostic reports
TARGET_CLASSES = ['16apsk', '32qam', '64qam', '128qam_cross', '256qam', 'am-dsb', 'fm']
TOTAL_SAMPLES = SAMPLES_PER_CLASS * len(TARGET_CLASSES)
OUTPUT_FILE = "data/VDF_SPECIALIST_NUTRIENTS.h5"
IQ_LENGTH = 1024

def generate_specialist_dataset():
    print(f"Opal Vanguard: Initiating Laptop Specialist Nutrient Factory")
    print(f"Targeting {len(TARGET_CLASSES)} high-difficulty classes | {TOTAL_SAMPLES} samples.")
    sys.stdout.flush()
    
    os.makedirs("data", exist_ok=True)
    class_to_idx = {name: i for i, name in enumerate(SIGNALS_SHARED_LIST)}

    with h5py.File(OUTPUT_FILE, 'w') as f:
        x_ds = f.create_dataset('X', (TOTAL_SAMPLES, IQ_LENGTH, 2), dtype='float32')
        y_ds = f.create_dataset('Y', (TOTAL_SAMPLES, NUM_CLASSES), dtype='float32')
        current_idx = 0

        # We use Impairment Level 2 (Environmental Hardened) for all nutrient samples
        # to ensure the model learns to separate them even in the presence of noise/fading.
        level = 2
        print(f"\n--- Stage: SPECIALIST NUTRIENTS (Level {level}) ---")
        sys.stdout.flush()
        
        for class_name in TARGET_CLASSES:
            if class_name not in class_to_idx:
                print(f" Skipping {class_name} (Not in master list).")
                continue
                
            class_idx = class_to_idx[class_name]
            print(f" Generating Concentrated {class_name}...")
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
                        data, returned_class_name = next(it)
                    except ValueError:
                        data = np.zeros(IQ_LENGTH, dtype=np.complex64)
                        returned_class_name = class_name
                    
                    if len(data) > IQ_LENGTH: data = data[:IQ_LENGTH]
                    elif len(data) < IQ_LENGTH: data = np.pad(data, (0, IQ_LENGTH - len(data)), 'constant')
                    
                    iq = np.stack([np.real(data), np.imag(data)], axis=-1)
                    # Standard Soft-Clip for project compatibility
                    iq = iq / (1.0 + np.abs(iq)) 
                    
                    x_ds[current_idx] = iq
                    
                    one_hot = np.zeros(NUM_CLASSES)
                    one_hot[class_idx] = 1.0 
                    y_ds[current_idx] = one_hot
                    
                    current_idx += 1
                    pbar.update(1)
                except StopIteration:
                    break
            pbar.close()

    print(f"\nMission Success: Specialist Nutrients saved to {OUTPUT_FILE}")
    sys.stdout.flush()

if __name__ == "__main__":
    generate_specialist_dataset()
