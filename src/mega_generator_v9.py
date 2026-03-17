import os
import sys
import h5py
import numpy as np
from tqdm import tqdm
from torchsig.utils.defaults import default_dataset
from torchsig.signals.signal_lists import SIGNALS_SHARED_LIST

# Configuration
NUM_CLASSES = 57
SAMPLES_PER_CLASS_CLEAN = 10000  # ~570k Clean samples
SAMPLES_PER_CLASS_HARD = 10000   # ~570k Hardened samples
TOTAL_SAMPLES = (SAMPLES_PER_CLASS_CLEAN + SAMPLES_PER_CLASS_HARD) * NUM_CLASSES
OUTPUT_FILE = "data/VDF_MEGA_SYNTHETIC_1M.h5"
IQ_LENGTH = 1024

def generate_dataset():
    print(f"Opal Vanguard: Initiating Mega-TorchSig Expansion (Target: {TOTAL_SAMPLES} samples)")
    sys.stdout.flush()
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Map class names to indices
    class_to_idx = {name: i for i, name in enumerate(SIGNALS_SHARED_LIST)}

    # 2. Initialize HDF5
    with h5py.File(OUTPUT_FILE, 'w') as f:
        x_ds = f.create_dataset('X', (TOTAL_SAMPLES, IQ_LENGTH, 2), dtype='float32')
        y_ds = f.create_dataset('Y', (TOTAL_SAMPLES, NUM_CLASSES), dtype='float32')
        
        current_idx = 0

        # Stage 1: Clean (Level 0)
        # Stage 2: Hardened (Level 2)
        for stage_name, samples_per_class, level in [("CLEAN", SAMPLES_PER_CLASS_CLEAN, 0), 
                                                     ("HARDENED", SAMPLES_PER_CLASS_HARD, 2)]:
            
            print(f"\n--- Launching {stage_name} Generation Stage (Level {level}) ---")
            sys.stdout.flush()
            
            for class_idx, class_name in enumerate(SIGNALS_SHARED_LIST):
                print(f" Generating {class_name} ({class_idx+1}/{NUM_CLASSES})...")
                sys.stdout.flush()
                
                dataset = default_dataset(
                    impairment_level=level,
                    num_iq_samples_dataset=IQ_LENGTH,
                    num_samples_dataset=samples_per_class,
                    use_class_list=[class_name]
                )
                
                it = iter(dataset)
                
                # Add progress bar for this class
                pbar = tqdm(total=samples_per_class, desc=f" {class_name}", unit="samples")
                
                for _ in range(samples_per_class):
                    try:
                        try:
                            data, returned_class_name = next(it)
                        except ValueError:
                            data = np.zeros(IQ_LENGTH, dtype=np.complex64)
                            returned_class_name = class_name
                        
                        if len(data) > IQ_LENGTH:
                            data = data[:IQ_LENGTH]
                        elif len(data) < IQ_LENGTH:
                            data = np.pad(data, (0, IQ_LENGTH - len(data)), 'constant')
                        
                        iq = np.stack([np.real(data), np.imag(data)], axis=-1)
                        iq = iq / (1.0 + np.abs(iq))
                        
                        x_ds[current_idx] = iq
                        
                        one_hot = np.zeros(NUM_CLASSES)
                        if returned_class_name in class_to_idx:
                            one_hot[class_to_idx[returned_class_name]] = 1.0
                        else:
                            one_hot[class_idx] = 1.0
                            
                        y_ds[current_idx] = one_hot
                        current_idx += 1
                        pbar.update(1)
                    except StopIteration:
                        break
                pbar.close()

    print(f"\nMission Success: Mega-Dataset saved to {OUTPUT_FILE}")
    sys.stdout.flush()

if __name__ == "__main__":
    generate_dataset()
