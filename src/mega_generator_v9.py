import os
import sys
import h5py
import numpy as np
from tqdm import tqdm
from torchsig.utils.defaults import default_dataset
from torchsig.signals.signal_lists import SIGNALS_SHARED_LIST

# Configuration
NUM_CLASSES = 57
SAMPLES_PER_CLASS_CLEAN = 10000  # 570k samples
SAMPLES_PER_CLASS_HARD = 10000   # 570k samples
TOTAL_SAMPLES = (SAMPLES_PER_CLASS_CLEAN + SAMPLES_PER_CLASS_HARD) * NUM_CLASSES
OUTPUT_FILE = "data/VDF_MEGA_SYNTHETIC_1M.h5"
IQ_LENGTH = 1024

def generate_dataset():
    print(f"Opal Vanguard: Initiating full Mega-TorchSig Expansion (Target: {TOTAL_SAMPLES} samples)")
    sys.stdout.flush()
    
    os.makedirs("data", exist_ok=True)
    class_to_idx = {name: i for i, name in enumerate(SIGNALS_SHARED_LIST)}

    # Initialize or Resume
    file_mode = 'a' if os.path.exists(OUTPUT_FILE) else 'w'
    with h5py.File(OUTPUT_FILE, file_mode) as f:
        if 'X' not in f:
            print("Creating fresh HDF5 structure...")
            x_ds = f.create_dataset('X', (TOTAL_SAMPLES, IQ_LENGTH, 2), dtype='float32')
            y_ds = f.create_dataset('Y', (TOTAL_SAMPLES, NUM_CLASSES), dtype='float32')
            current_idx = 0
        else:
            x_ds = f['X']
            y_ds = f['Y']
            print("Searching for resume point...")
            # Heuristic check to find the first empty row
            current_idx = 0
            for i in range(0, TOTAL_SAMPLES, 1000):
                if np.sum(y_ds[i]) == 0:
                    current_idx = i
                    break
            print(f"Resuming generation from global index: {current_idx}")

        for stage_name, samples_per_class, level in [("CLEAN", SAMPLES_PER_CLASS_CLEAN, 0), 
                                                     ("HARDENED", SAMPLES_PER_CLASS_HARD, 2)]:
            print(f"\n--- Stage: {stage_name} (Level {level}) ---")
            sys.stdout.flush()
            
            for class_idx, class_name in enumerate(SIGNALS_SHARED_LIST):
                # Class tracking
                stage_offset = 0 if stage_name == "CLEAN" else (SAMPLES_PER_CLASS_CLEAN * NUM_CLASSES)
                class_start_idx = stage_offset + (class_idx * samples_per_class)
                class_end_idx = class_start_idx + samples_per_class
                
                if current_idx >= class_end_idx:
                    # Skip already completed classes
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
                
                # Fast-forward if resuming mid-class
                samples_to_skip = max(0, current_idx - class_start_idx)
                if samples_to_skip > 0:
                    print(f"  Fast-forwarding {samples_to_skip} samples...")
                    for _ in range(samples_to_skip):
                        try: next(it)
                        except: pass
                
                samples_to_gen = samples_per_class - samples_to_skip
                pbar = tqdm(total=samples_per_class, desc=f" {class_name}", unit="samples")
                pbar.update(samples_to_skip)
                
                for _ in range(samples_to_gen):
                    try:
                        try:
                            data, returned_class_name = next(it)
                        except ValueError:
                            data = np.zeros(IQ_LENGTH, dtype=np.complex64)
                            returned_class_name = class_name
                        
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

    print(f"\nMission Success: Full Mega-Dataset saved to {OUTPUT_FILE}")
    sys.stdout.flush()

if __name__ == "__main__":
    generate_dataset()
