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

    # 2. Initialize or Resume HDF5
    file_mode = 'a' if os.path.exists(OUTPUT_FILE) else 'w'
    with h5py.File(OUTPUT_FILE, file_mode) as f:
        if 'X' not in f:
            x_ds = f.create_dataset('X', (TOTAL_SAMPLES, IQ_LENGTH, 2), dtype='float32')
            y_ds = f.create_dataset('Y', (TOTAL_SAMPLES, NUM_CLASSES), dtype='float32')
            current_idx = 0
        else:
            x_ds = f['X']
            y_ds = f['Y']
            # Find the first zero-row to resume (assuming one-hot labels)
            # This is a heuristic: check every 1000 rows to find the end
            print("Searching for resume point...")
            current_idx = 0
            for i in range(0, TOTAL_SAMPLES, 1000):
                if np.sum(y_ds[i]) == 0:
                    current_idx = i
                    break
            print(f"Resuming generation from index: {current_idx}")

        # Stage 1: Clean (Level 0)
        # Stage 2: Hardened (Level 2)
        for stage_name, samples_per_class, level in [("CLEAN", SAMPLES_PER_CLASS_CLEAN, 0), 
                                                     ("HARDENED", SAMPLES_PER_CLASS_HARD, 2)]:
            
            print(f"\n--- Launching {stage_name} Generation Stage (Level {level}) ---")
            sys.stdout.flush()
            
            for class_idx, class_name in enumerate(SIGNALS_SHARED_LIST):
                # Calculate the index range for this class in this stage
                stage_offset = 0 if stage_name == "CLEAN" else (SAMPLES_PER_CLASS_CLEAN * NUM_CLASSES)
                class_start_idx = stage_offset + (class_idx * samples_per_class)
                class_end_idx = class_start_idx + samples_per_class
                
                # SKIP if entire class is already generated
                if current_idx >= class_end_idx:
                    print(f" Skipping {class_name} (Already complete).")
                    continue
                
                print(f" Generating {class_name} ({class_idx+1}/{NUM_CLASSES})...")
                sys.stdout.flush()
                
                dataset = default_dataset(
                    impairment_level=level,
                    num_iq_samples_dataset=IQ_LENGTH,
                    num_samples_dataset=samples_per_class,
                    use_class_list=[class_name]
                )
                
                it = iter(dataset)
                
                # If we are resuming mid-class, we need to fast-forward the iterator
                samples_to_skip = max(0, current_idx - class_start_idx)
                if samples_to_skip > 0:
                    print(f"  Fast-forwarding {samples_to_skip} samples...")
                    for _ in range(samples_to_skip):
                        try:
                            next(it)
                        except (StopIteration, ValueError):
                            pass
                
                samples_to_generate = samples_per_class - samples_to_skip
                pbar = tqdm(total=samples_per_class, desc=f" {class_name}", unit="samples")
                pbar.update(samples_to_skip)
                
                for _ in range(samples_to_generate):
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
